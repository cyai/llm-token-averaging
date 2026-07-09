"""
Unified training script for the Chinchilla FLOPs comparison experiment.

Uses OLM's distributed training utilities for multi-GPU DDP:
  olm.core.dist.setup_distributed()   — init NCCL process group
  olm.core.dist.get_local_rank()      — local GPU index for this process
  olm.core.dist.get_rank()            — global rank
  olm.core.dist.get_world_size()      — total number of GPUs
  olm.core.dist.is_main_process()     — True only on rank 0

See: https://github.com/openlanguagemodel/openlanguagemodel/blob/main/docs/datasets-and-training.md

The training loop itself is kept custom because OLM's DDPTrainer does not
support the averaged-LM forward interface (OLMAveragedLanguageModel) or the
FLOPs-accounting / CSV logging we need.  However all distributed plumbing
delegates to OLM's dist module.

Single-GPU
----------
    python experiments/chinchilla/train.py \
        --model model1_125m --batch_size 16 --device cuda

8-GPU DDP on 8× A6000 (recommended)
-------------------------------------
    torchrun --standalone --nproc_per_node=8 \
        experiments/chinchilla/train.py \
        --model model1_125m --batch_size 16

    # batch_size is per GPU; global effective batch = batch_size × 8

Resume
------
    torchrun --standalone --nproc_per_node=8 \
        experiments/chinchilla/train.py \
        --model model2_500m --batch_size 8 --resume
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Optional

import math

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from transformers import AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.chinchilla.fineweb_loader import build_dataloaders, estimate_total_batches
from experiments.chinchilla.model_configs import get_config, ModelConfig, MODEL_CONFIGS
from experiments.shared.olm_model import (
    OLMTransformerBody,
    OLMAveragedLanguageModel,
    OLMPhasedAveragedLanguageModel,
)
from experiments.shared.averaged_lm import build_method_config


# ---------------------------------------------------------------------------
# Cosine LR schedule with min_lr  (Chinchilla / LLaMA standard: decay to 10%)
# ---------------------------------------------------------------------------

MIN_LR_RATIO = 0.1  # η_min = 0.1 × η_max (Chinchilla 10× decay, LLaMA convention)


def get_cosine_schedule_with_warmup_and_min_lr(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = MIN_LR_RATIO,
) -> LambdaLR:
    """
    Cosine decay from peak LR to min_lr = min_lr_ratio × peak_lr.

    Matches the schedule used by Chinchilla (Hoffmann et al., 2022) and
    LLaMA (Touvron et al., 2023): 10× decay over the full cosine cycle,
    with the cycle length exactly matched to training duration.

    Chinchilla Appendix Figure A1 explicitly shows that overshooting the
    cycle length beyond ~25% hurts final loss — so we keep the cycle
    matched to num_training_steps and decay to a floor instead.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# OLM distributed utilities  (with graceful fallback to raw torch.distributed)
# ---------------------------------------------------------------------------

def _setup_distributed() -> tuple[int, int, int]:
    """
    Initialise the distributed process group using OLM's setup_distributed(),
    falling back to raw torch.distributed if OLM is not available.

    Returns:
        (local_rank, global_rank, world_size)
    """
    local_rank  = int(os.environ.get("LOCAL_RANK",  0))
    global_rank = int(os.environ.get("RANK",        0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))

    if world_size == 1:
        return local_rank, global_rank, world_size

    # Use raw torch.distributed directly — OLM's setup_distributed calls
    # torch.distributed.timedelta which does not exist in any PyTorch version
    # (the correct call is datetime.timedelta).  Bypass OLM entirely.
    import torch.distributed as _dist
    torch.cuda.set_device(local_rank)
    if not _dist.is_initialized():
        _dist.init_process_group(backend="nccl")

    return local_rank, global_rank, world_size


def _cleanup_distributed(world_size: int) -> None:
    # torchrun handles process group teardown; nothing to do here.
    pass


def _get_rank() -> int:
    try:
        from olm.core.dist import get_rank
        return get_rank()
    except ImportError:
        return int(os.environ.get("RANK", 0))


def _get_world_size() -> int:
    try:
        from olm.core.dist import get_world_size
        return get_world_size()
    except ImportError:
        return int(os.environ.get("WORLD_SIZE", 1))


def _is_main_process() -> bool:
    try:
        from olm.core.dist import is_main_process
        return is_main_process()
    except ImportError:
        return int(os.environ.get("RANK", 0)) == 0


def _all_reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Average a scalar tensor across all ranks."""
    if world_size > 1:
        import torch.distributed as dist
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor


def _barrier(world_size: int) -> None:
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.1):
    """OLM AdamW with standard parameter grouping (1-D tensors skip decay)."""
    try:
        from olm.train.optim import AdamW as OLMAdamW
        optimizer_cls = OLMAdamW
    except ImportError:
        optimizer_cls = torch.optim.AdamW

    decay    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    return optimizer_cls(
        [{"params": decay,    "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def _enable_gradient_checkpointing(backbone: OLMTransformerBody) -> None:
    """
    Patch the transformer body to run each layer through gradient checkpointing.

    OLM stores transformer layers inside a ModuleList (which has no forward()).
    We unwrap ModuleList containers one level so we checkpoint individual
    callable layers, not the container itself.
    """
    body = backbone.body

    # HuggingFace-style convenience hook (not present in plain OLM blocks)
    if hasattr(body, "gradient_checkpointing_enable"):
        body.gradient_checkpointing_enable()
        return

    # Collect the actual transformer layers.
    # OLM body structure: Sequential/Block → [ModuleList([layer0, layer1, ...])]
    # We need the individual layers, not the ModuleList wrapper.
    layers: list = []
    for child in body.children():
        if isinstance(child, nn.ModuleList):
            # Unwrap: add each transformer layer individually
            layers.extend(list(child))
        else:
            layers.append(child)

    if not layers:
        return

    # Verify every collected module is callable (has a real forward).
    # If any are still ModuleList, wrap them in Sequential so they become callable.
    callable_layers = []
    for m in layers:
        if isinstance(m, nn.ModuleList):
            callable_layers.append(nn.Sequential(*m))
        else:
            callable_layers.append(m)

    def _ckpt_forward(x: torch.Tensor) -> torch.Tensor:
        for layer in callable_layers:
            x = gradient_checkpoint(layer, x, use_reentrant=False)
        return x

    body.forward = _ckpt_forward


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_loss(
    model: nn.Module,
    eval_dl,
    device: str,
    max_batches: int = 64,
    is_averaged: bool = False,
) -> float:
    """Compute mean cross-entropy loss on the eval dataloader (rank-0 only)."""
    model.eval()
    total_loss = 0.0
    count = 0

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    for i, batch in enumerate(eval_dl):
        if i >= max_batches:
            break
        # OLM DataLoader returns dicts; fallback returns raw tensors
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
        else:
            input_ids = batch.to(device)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                            enabled=(device_type == "cuda")):
            if is_averaged:
                loss, _ = model(input_ids)
            else:
                logits = model(input_ids)
                labels = input_ids[:, 1:].contiguous()
                logits = logits[:, :-1].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )
        total_loss += loss.item()
        count += 1

    model.train()
    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(
    cfg: ModelConfig,
    args: argparse.Namespace,
    results_dir: Optional[Path] = None,
) -> Path:
    """
    Train a single model config, optionally across multiple GPUs via OLM DDP.

    Args:
        cfg         : ModelConfig (architecture + training budget)
        args        : parsed CLI namespace
        results_dir : override output directory

    Returns:
        Path to loss_log.csv (only meaningful on rank 0).
    """
    # --- OLM distributed setup ----------------------------------------------
    local_rank, global_rank, world_size = _setup_distributed()
    is_main = _is_main_process()

    device = f"cuda:{local_rank}" if world_size > 1 else args.device

    # Enable TF32 tensor cores for any FP32 matmuls (Ampere+, e.g. A6000).
    # Free ~2× over strict FP32; combined with bf16 autocast below this is the
    # main throughput fix. No effect on CPU.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    use_amp = device_type == "cuda"

    if is_main:
        print(
            f"[{cfg.name}] Distributed setup: "
            f"world_size={world_size}, local_rank={local_rank}, "
            f"device={device}",
            flush=True,
        )

    # --- directories (rank 0 creates; barrier ensures others see them) ------
    if results_dir is None:
        results_dir = _ROOT / "experiments" / "chinchilla" / "results" / cfg.name
    results_dir = Path(results_dir)
    ckpt_dir = results_dir / "checkpoints"
    if is_main:
        results_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    _barrier(world_size)

    log_path = results_dir / "loss_log.csv"

    # --- tokenizer ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- data  (OLM DataLoader handles DDP sharding via distributed=True) ---
    if is_main:
        print(
            f"[{cfg.name}] Building FineWeb dataloaders "
            f"(distributed={world_size > 1}) …",
            flush=True,
        )

    train_dl, eval_dl = build_dataloaders(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        num_workers=args.num_workers,
        rank=global_rank,
        world_size=world_size,
        distributed=(world_size > 1),
        data_dir=args.data_dir,
        target_tokens=cfg.target_tokens,
    )

    # --- model --------------------------------------------------------------
    is_averaged = cfg.averaging_k > 1

    if is_main:
        print(
            f"[{cfg.name}] Building OLM backbone "
            f"d={cfg.d_model} h={cfg.n_heads} l={cfg.n_layers} …",
            flush=True,
        )

    backbone = OLMTransformerBody(
        vocab_size=len(tokenizer),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        context_length=cfg.context_len,
    )

    if cfg.tie_embeddings:
        backbone.tie_embedding_weights()
        if is_main:
            print(f"[{cfg.name}] Tied input/output embedding weights", flush=True)

    is_phased = cfg.multi_token_phase_ratio > 0

    if is_averaged:
        _method_name = cfg.method_name or f"uniform_k{cfg.averaging_k}"
        method_cfg = build_method_config(_method_name)
        if is_phased:
            model = OLMPhasedAveragedLanguageModel(
                backbone, method_cfg, k=cfg.averaging_k,
            )
        else:
            model = OLMAveragedLanguageModel(backbone, method_cfg)
    else:
        model = backbone

    if cfg.grad_checkpoint:
        if is_main:
            print(f"[{cfg.name}] Enabling gradient checkpointing …", flush=True)
        _enable_gradient_checkpointing(backbone)

    model.to(device)

    # Wrap with PyTorch DDP (OLM's DDPTrainer uses this internally as well)
    ddp_model = model
    if world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(
            f"[{cfg.name}] Parameters: {n_params:,} ({n_params/1e6:.1f}M) | "
            f"GPUs: {world_size} | "
            f"Global batch: {args.batch_size} × {world_size} = "
            f"{args.batch_size * world_size} seqs/step",
            flush=True,
        )

    if is_phased and is_main:
        print(
            f"[{cfg.name}] Token Superposition Training: "
            f"k={cfg.averaging_k}, MCE bag loss | "
            f"Phase 1 = first {cfg.multi_token_phase_ratio:.0%} steps (superposition), "
            f"Phase 2 = remaining (single-token recovery)",
            flush=True,
        )

    # --- OLM AdamW + cosine-warmup scheduler --------------------------------
    # Cosine decay to 10% of peak LR (Chinchilla/LLaMA convention).
    # Cycle length = total training steps (Chinchilla Fig A1: overshooting hurts).
    total_steps = estimate_total_batches(
        cfg.target_tokens, args.seq_len, args.batch_size, world_size
    )
    optimizer = _build_optimizer(model, lr=cfg.lr)
    scheduler = get_cosine_schedule_with_warmup_and_min_lr(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=MIN_LR_RATIO,
    )

    # --- resume -------------------------------------------------------------
    start_step       = 0
    tokens_seen      = 0
    cumulative_flops = 0.0

    if args.resume:
        ckpts = sorted(ckpt_dir.glob("step_*.pt"))
        if ckpts:
            latest = ckpts[-1]
            if is_main:
                print(f"[{cfg.name}] Resuming from {latest} …", flush=True)
            state = torch.load(latest, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_step       = state["step"]
            tokens_seen      = state["tokens_seen"]
            cumulative_flops = state["cumulative_flops"]
            if is_main:
                print(
                    f"[{cfg.name}] Resumed: step={start_step:,} "
                    f"tokens={tokens_seen/1e9:.2f}B",
                    flush=True,
                )
        elif is_main:
            print(
                f"[{cfg.name}] --resume set but no checkpoint found; "
                f"starting from scratch.",
                flush=True,
            )

    # --- CSV log (rank 0 only) ----------------------------------------------
    csv_file   = None
    csv_writer = None
    if is_main:
        csv_mode = "a" if (args.resume and log_path.exists()) else "w"
        csv_file = open(log_path, csv_mode, newline="")
        csv_writer = csv.writer(csv_file)
        if csv_mode == "w":
            csv_writer.writerow(
                ["step", "tokens_seen", "cumulative_flops", "train_loss", "eval_loss"]
            )
            csv_file.flush()

    # --- FLOPs per global optimizer step ------------------------------------
    # Training FLOPs = 3 × forward FLOPs (forward + backward).
    # Forward FLOPs per token per layer:
    #   Attention projections (Q,K,V,Out): 4 × 2d² = 8d²
    #   Attention matmuls (Q@K^T, Attn@V): 2×2×L×d = 4Ld
    #   SwiGLU FFN (up: d→2h, down: h→d, h=2.5d): 2d×2h + 2h×d = 2d×5d + 2×2.5d×d = 15d²
    #   Total forward per token per layer: 8d² + 4Ld + 15d² = 23d² + 4Ld
    # Training (fwd+bwd) per token per layer: 3 × (23d² + 4Ld) = 69d² + 12Ld
    transformer_L          = args.seq_len // cfg.averaging_k
    d                      = cfg.d_model
    fwd_flops_per_token_per_layer = 23 * d * d + 4 * transformer_L * d
    train_flops_per_token_per_layer = 3 * fwd_flops_per_token_per_layer
    flops_per_seq          = cfg.n_layers * transformer_L * train_flops_per_token_per_layer
    seqs_per_global_step   = args.batch_size * world_size
    tokens_per_global_step = seqs_per_global_step * args.seq_len
    flops_per_global_step  = seqs_per_global_step * flops_per_seq

    # --- phased training: compute switch step --------------------------------
    phase_switch_step = 0
    phase_switched    = False
    if is_phased:
        phase_switch_step = int(total_steps * cfg.multi_token_phase_ratio)
        if is_main:
            print(
                f"[{cfg.name}] Phase switch at step {phase_switch_step:,} / {total_steps:,}",
                flush=True,
            )
        # If resuming past the switch point, start in single-token mode
        if start_step >= phase_switch_step:
            raw_model = ddp_model.module if world_size > 1 else ddp_model
            raw_model.set_multi_token(False)
            phase_switched = True

    # --- training loop ------------------------------------------------------
    ddp_model.train()
    step       = start_step
    ema_loss   = None        # GPU tensor; EMA of the per-step loss (no host sync)
    ema_val    = float("nan")  # host float, refreshed (all-reduced) at log steps
    EMA_A      = 0.98
    early_stop = False   # set to True on all ranks when target eval loss is hit

    train_iter = iter(train_dl)

    if is_main:
        print(
            f"[{cfg.name}] Starting: {total_steps:,} steps × "
            f"{tokens_per_global_step:,} tokens/step "
            f"= {cfg.target_tokens/1e9:.1f}B tokens",
            flush=True,
        )

    # Reset peak memory stats for accurate tracking from training start
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        # OLM DataLoader may return dicts; handle both cases
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
        else:
            input_ids = batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                            enabled=use_amp):
            if is_averaged:
                loss, _ = ddp_model(input_ids)
            else:
                logits = ddp_model(input_ids)
                labels = input_ids[:, 1:].contiguous()
                logits = logits[:, :-1].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step             += 1
        tokens_seen      += tokens_per_global_step
        cumulative_flops += flops_per_global_step

        # --- phase switch (superposition → recovery) -----------------------
        if is_phased and not phase_switched and step >= phase_switch_step:
            raw_model = ddp_model.module if world_size > 1 else ddp_model
            raw_model.set_multi_token(False)
            phase_switched = True
            if is_main:
                print(
                    f"[{cfg.name}] Phase switch → single-token recovery "
                    f"at step {step:,} ({tokens_seen/1e9:.3f}B tokens)",
                    flush=True,
                )

        # Update the EMA of the loss entirely on-GPU (no host sync per step).
        # The cross-rank all-reduce is deferred to logging steps: because both
        # EMA and all-reduce-mean are linear, mean_r(EMA_r) == EMA(mean_r(loss_r)),
        # so this is exactly equivalent to averaging the per-step loss across
        # ranks every step — just without the per-step collective + .item() sync.
        loss_detached = loss.detach()
        if ema_loss is None:
            ema_loss = loss_detached.clone()
        else:
            ema_loss.mul_(EMA_A).add_(loss_detached, alpha=1 - EMA_A)

        # --- cross-rank loss sync at log steps (collective: ALL ranks) -------
        if step % args.log_steps == 0:
            ema_t = ema_loss.clone()
            _all_reduce_mean(ema_t, world_size)
            ema_val = ema_t.item()

        # --- logging (rank 0) -----------------------------------------------
        if is_main and step % args.log_steps == 0:
            eval_loss_val = _eval_loss(
                model, eval_dl, device,
                max_batches=args.eval_batches,
                is_averaged=is_averaged,
            )
            csv_writer.writerow([
                step, tokens_seen, f"{cumulative_flops:.6e}",
                f"{ema_val:.6f}", f"{eval_loss_val:.6f}",
            ])
            csv_file.flush()

            elapsed   = time.time() - t0
            tok_s     = tokens_seen / elapsed
            pct       = 100.0 * tokens_seen / cfg.target_tokens
            eta_h     = ((cfg.target_tokens - tokens_seen) / max(tok_s, 1)) / 3600
            lr_now    = scheduler.get_last_lr()[0]
            mem_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            mem_resrv = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

            print(
                f"[{cfg.name}] step={step:>8,} | "
                f"tokens={tokens_seen/1e9:.3f}B/{cfg.target_tokens/1e9:.0f}B ({pct:.1f}%) | "
                f"loss={ema_val:.4f} | eval={eval_loss_val:.4f} | "
                f"lr={lr_now:.2e} | flops={cumulative_flops:.2e} | "
                f"tok/s={tok_s:,.0f} | ETA={eta_h:.1f}h | "
                f"mem={mem_alloc:.1f}GB/{mem_resrv:.1f}GB",
                flush=True,
            )

            if (
                args.early_stop_eval_loss is not None
                and eval_loss_val <= args.early_stop_eval_loss
            ):
                print(
                    f"[{cfg.name}] Early stop: eval_loss={eval_loss_val:.6f} "
                    f"<= target {args.early_stop_eval_loss:.6f} "
                    f"at {tokens_seen/1e9:.3f}B tokens / {cumulative_flops:.3e} FLOPs",
                    flush=True,
                )
                early_ckpt = ckpt_dir / f"early_stop_step_{step:08d}.pt"
                torch.save(
                    {
                        "step":             step,
                        "tokens_seen":      tokens_seen,
                        "cumulative_flops": cumulative_flops,
                        "model":            model.state_dict(),
                        "optimizer":        optimizer.state_dict(),
                        "scheduler":        scheduler.state_dict(),
                    },
                    early_ckpt,
                )
                csv_writer.writerow([
                    step, tokens_seen, f"{cumulative_flops:.6e}",
                    f"{ema_val:.6f}", f"{eval_loss_val:.6f}",
                ])
                csv_file.flush()
                csv_file.close()
                csv_file = None
                early_stop = True

        # Broadcast early_stop flag so all DDP ranks exit the loop together.
        # early_stop can only change inside the rank-0 log-step block above,
        # so we only need this collective (+ host sync) at log steps.
        if world_size > 1 and step % args.log_steps == 0:
            import torch.distributed as dist
            flag = torch.tensor(int(early_stop), dtype=torch.int32,
                                device=device)
            dist.broadcast(flag, src=0)
            early_stop = bool(flag.item())
        if early_stop:
            _cleanup_distributed(world_size)
            return log_path

        # --- checkpoint (rank 0; barrier syncs all ranks) -------------------
        if is_main and step % args.checkpoint_steps == 0:
            ckpt_path = ckpt_dir / f"step_{step:08d}.pt"
            torch.save(
                {
                    "step":             step,
                    "tokens_seen":      tokens_seen,
                    "cumulative_flops": cumulative_flops,
                    "model":            model.state_dict(),
                    "optimizer":        optimizer.state_dict(),
                    "scheduler":        scheduler.state_dict(),
                },
                ckpt_path,
            )
            print(f"[{cfg.name}] Checkpoint → {ckpt_path}", flush=True)

        # No per-step barrier: DDP's backward all-reduce already keeps ranks in
        # lock-step every step. Rank-0-only checkpointing is fine — other ranks
        # naturally wait at the next gradient all-reduce.

    # --- final checkpoint + eval (rank 0) -----------------------------------
    if is_main:
        final_ckpt = ckpt_dir / "final.pt"
        torch.save(
            {
                "step":             step,
                "tokens_seen":      tokens_seen,
                "cumulative_flops": cumulative_flops,
                "model":            model.state_dict(),
                "optimizer":        optimizer.state_dict(),
                "scheduler":        scheduler.state_dict(),
            },
            final_ckpt,
        )
        final_eval = _eval_loss(
            model, eval_dl, device,
            max_batches=args.eval_batches * 2,
            is_averaged=is_averaged,
        )
        csv_writer.writerow([
            step, tokens_seen, f"{cumulative_flops:.6e}",
            f"{ema_val:.6f}", f"{final_eval:.6f}",
        ])
        if csv_file is not None:
            csv_file.flush()
            csv_file.close()
            csv_file = None

        elapsed_total = time.time() - t0
        print(
            f"[{cfg.name}] Done.  "
            f"Final eval loss: {final_eval:.4f}.  "
            f"Total time: {elapsed_total/3600:.1f}h",
            flush=True,
        )

    _cleanup_distributed(world_size)
    return log_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train one Chinchilla comparison model on FineWeb sample-10BT.\n"
            "Single-GPU:  python train.py --model model1_125m\n"
            "8-GPU DDP:   torchrun --standalone --nproc_per_node=8 train.py --model model1_125m"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model",            required=True,
                   choices=list(MODEL_CONFIGS.keys()),
                   help="Any config name from model_configs.py")
    p.add_argument("--batch_size",       type=int, default=16,
                   help="Per-GPU batch size.")
    p.add_argument("--seq_len",          type=int, default=1024)
    p.add_argument("--device",           type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device for single-GPU mode. Ignored by torchrun.")
    p.add_argument("--log_steps",        type=int, default=1_000)
    p.add_argument("--checkpoint_steps", type=int, default=50_000)
    p.add_argument("--eval_batches",     type=int, default=64)
    p.add_argument("--num_workers",      type=int, default=4,
                   help="DataLoader workers per rank. Use 4 with local binary "
                        "cache (fast memmap reads). Overridden to 0 automatically "
                        "when falling back to HF streaming.")
    p.add_argument("--data_dir",         type=str, default=None,
                   help="Path to directory containing pre-tokenised train.bin / "
                        "eval.bin (produced by fineweb_loader.py). If omitted "
                        "or files are missing, falls back to HF streaming.")
    p.add_argument("--tokenizer_name",   type=str,
                   default="EleutherAI/pythia-70m")
    p.add_argument("--resume",           action="store_true")
    p.add_argument("--results_dir",      type=str, default=None)
    p.add_argument("--early_stop_eval_loss", type=float, default=None,
                   help="Stop training early when eval loss falls at or below "
                        "this value (e.g. 4.286754 to match avg_50m_k2_v2).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = get_config(args.model)
    results_dir = Path(args.results_dir) / cfg.name if args.results_dir else None
    log_path = train_model(cfg, args, results_dir=results_dir)
    if _is_main_process():
        print(f"Loss log saved to: {log_path}", flush=True)


if __name__ == "__main__":
    main()
