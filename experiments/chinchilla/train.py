"""
Unified training script for the Chinchilla FLOPs comparison experiment.

Supports both single-GPU and multi-GPU (DDP via torchrun) training.

Single-GPU
----------
    python experiments/chinchilla/train.py \
        --model model1_125m --batch_size 16 --device cuda

8-GPU DDP (recommended for 8× A6000)
--------------------------------------
    torchrun --standalone --nproc_per_node=8 \
        experiments/chinchilla/train.py \
        --model model1_125m --batch_size 16

    # batch_size is per-GPU; global effective batch = batch_size × 8 GPUs

Resume interrupted DDP run
---------------------------
    torchrun --standalone --nproc_per_node=8 \
        experiments/chinchilla/train.py \
        --model model2_500m --batch_size 8 --resume

Notes
-----
* LOCAL_RANK / WORLD_SIZE / RANK are set automatically by torchrun.
* When world_size > 1 the model is wrapped in DistributedDataParallel.
* Each GPU receives a disjoint shard of the FineWeb document stream.
* CSV logging, checkpointing, and stdout progress are rank-0 only.
* FLOPs are accumulated globally across all GPUs (world_size * per-GPU FLOPs).
* With 8× A6000 (48 GB each, ~155 TFLOPS BF16):
    model1_125m  ≈ 3–4 h
    avg_125m_k2  ≈ 1.5–2 h
    model2_500m  ≈ 12–14 h   (no grad-ckpt needed on 48 GB)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.chinchilla.fineweb_loader import build_dataloaders, estimate_total_batches
from experiments.chinchilla.model_configs import get_config, ModelConfig
from experiments.shared.olm_model import OLMTransformerBody, OLMAveragedLanguageModel
from experiments.shared.averaged_lm import build_method_config


# ---------------------------------------------------------------------------
# DDP setup helpers
# ---------------------------------------------------------------------------

def _setup_ddp() -> tuple[int, int, int]:
    """
    Initialise the default process group if torchrun set LOCAL_RANK.

    Returns:
        (local_rank, global_rank, world_size)
    """
    local_rank  = int(os.environ.get("LOCAL_RANK",  0))
    global_rank = int(os.environ.get("RANK",        0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return local_rank, global_rank, world_size


def _cleanup_ddp(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def _all_reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Average a scalar tensor across all ranks."""
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.1):
    """AdamW with OLM-style parameter grouping (1-D tensors skip weight decay)."""
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
    """Wrap each child of the transformer body with torch gradient checkpointing."""
    body = backbone.body
    if hasattr(body, "gradient_checkpointing_enable"):
        body.gradient_checkpointing_enable()
        return
    children = list(body.children())
    if children:
        def _ckpt_forward(x: torch.Tensor) -> torch.Tensor:
            for m in children:
                x = gradient_checkpoint(m, x, use_reentrant=False)
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

    for i, batch in enumerate(eval_dl):
        if i >= max_batches:
            break
        input_ids = batch.to(device)

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
    Train a single model config, optionally across multiple GPUs via DDP.

    Args:
        cfg         : ModelConfig (architecture + training budget)
        args        : parsed CLI namespace
        results_dir : override output directory

    Returns:
        Path to loss_log.csv (only meaningful on rank 0).
    """
    # --- DDP init -----------------------------------------------------------
    local_rank, global_rank, world_size = _setup_ddp()
    is_main = global_rank == 0

    # Determine device: torchrun sets LOCAL_RANK → use that GPU
    if world_size > 1:
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    # --- directories (rank 0 creates them; others wait) ---------------------
    if results_dir is None:
        results_dir = _ROOT / "experiments" / "chinchilla" / "results" / cfg.name
    results_dir = Path(results_dir)
    ckpt_dir = results_dir / "checkpoints"
    if is_main:
        results_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    log_path = results_dir / "loss_log.csv"

    # --- tokenizer ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- data (each rank gets its own shard) --------------------------------
    if is_main:
        print(f"[{cfg.name}] Building FineWeb dataloaders "
              f"(rank {global_rank}/{world_size}) …", flush=True)

    train_dl, eval_dl = build_dataloaders(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,   # per-GPU batch size
        eval_batches=args.eval_batches,
        num_workers=args.num_workers,
        rank=global_rank,
        world_size=world_size,
    )

    # --- model --------------------------------------------------------------
    is_averaged = cfg.averaging_k > 1

    if is_main:
        print(f"[{cfg.name}] Building OLM backbone "
              f"d={cfg.d_model} h={cfg.n_heads} l={cfg.n_layers} …", flush=True)

    backbone = OLMTransformerBody(
        vocab_size=len(tokenizer),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        context_length=cfg.context_len,
    )

    if is_averaged:
        method_cfg = build_method_config(f"uniform_k{cfg.averaging_k}")
        model = OLMAveragedLanguageModel(backbone, method_cfg)
    else:
        model = backbone   # OLMTransformerBody is itself a valid nn.Module

    # Gradient checkpointing: on A6000 (48 GB) the 500M model fits without it,
    # but keep it conditional on the config flag for flexibility.
    if cfg.grad_checkpoint:
        if is_main:
            print(f"[{cfg.name}] Enabling gradient checkpointing …", flush=True)
        _enable_gradient_checkpointing(backbone)

    model.to(device)

    # Wrap with DDP (only after moving to GPU)
    ddp_model = model
    if world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"[{cfg.name}] Parameters: {n_params:,} ({n_params/1e6:.1f}M) | "
              f"World size: {world_size} GPU(s)", flush=True)

    # --- optimiser + scheduler (on the underlying model, not DDP wrapper) --
    total_steps = estimate_total_batches(
        cfg.target_tokens, args.seq_len, args.batch_size, world_size
    )
    optimizer = _build_optimizer(model, lr=cfg.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    # --- resume -------------------------------------------------------------
    start_step       = 0
    tokens_seen      = 0        # global tokens across all GPUs
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
                print(f"[{cfg.name}] Resumed at step={start_step:,} "
                      f"tokens={tokens_seen:,}", flush=True)
        elif is_main:
            print(f"[{cfg.name}] --resume set but no checkpoint found; "
                  f"starting from scratch.", flush=True)

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

    # --- FLOPs accounting ---------------------------------------------------
    # Per global step (all GPUs combined):
    #   standard  : 6 · N · (batch_per_gpu · world_size) · seq_len
    #   averaging : 6 · N · (batch_per_gpu · world_size) · (seq_len / k)
    effective_seq       = args.seq_len / cfg.averaging_k
    tokens_per_gpu_step = args.batch_size * args.seq_len
    tokens_per_global_step = tokens_per_gpu_step * world_size
    flops_per_global_step  = (
        6.0 * n_params * args.batch_size * world_size * effective_seq
    )

    # --- training loop ------------------------------------------------------
    ddp_model.train()
    step      = start_step
    ema_loss  = None
    ema_alpha = 0.98

    train_iter = iter(train_dl)

    if is_main:
        print(
            f"[{cfg.name}] Starting training: "
            f"{total_steps:,} steps × {tokens_per_global_step:,} tokens/step "
            f"= {cfg.target_tokens/1e9:.1f}B total tokens  "
            f"(batch {args.batch_size}/GPU × {world_size} GPU = "
            f"{args.batch_size * world_size} global batch)",
            flush=True,
        )
    t0 = time.time()

    while step < total_steps:
        # Fetch a batch for this rank
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        input_ids = batch.to(device)

        optimizer.zero_grad(set_to_none=True)

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

        # Average the loss across all ranks for accurate logging
        loss_tensor = loss.detach().clone()
        _all_reduce_mean(loss_tensor, world_size)
        loss_val = loss_tensor.item()

        if ema_loss is None:
            ema_loss = loss_val
        else:
            ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * loss_val

        # --- logging (rank 0 only) ------------------------------------------
        if is_main and step % args.log_steps == 0:
            eval_loss_val = _eval_loss(
                model, eval_dl, device,
                max_batches=args.eval_batches,
                is_averaged=is_averaged,
            )
            csv_writer.writerow([
                step, tokens_seen, f"{cumulative_flops:.6e}",
                f"{ema_loss:.6f}", f"{eval_loss_val:.6f}",
            ])
            csv_file.flush()

            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / elapsed
            pct = 100.0 * tokens_seen / cfg.target_tokens
            eta_h = ((cfg.target_tokens - tokens_seen) / max(tok_per_sec, 1)) / 3600

            print(
                f"[{cfg.name}] step={step:>8,} | "
                f"tokens={tokens_seen/1e9:.3f}B/{cfg.target_tokens/1e9:.0f}B "
                f"({pct:.1f}%) | "
                f"loss={ema_loss:.4f} | eval={eval_loss_val:.4f} | "
                f"flops={cumulative_flops:.2e} | "
                f"tok/s={tok_per_sec:,.0f} | ETA={eta_h:.1f}h",
                flush=True,
            )

        # --- checkpoint (rank 0 only) ----------------------------------------
        if is_main and step % args.checkpoint_steps == 0:
            ckpt_path = ckpt_dir / f"step_{step:08d}.pt"
            torch.save(
                {
                    "step":              step,
                    "tokens_seen":       tokens_seen,
                    "cumulative_flops":  cumulative_flops,
                    "model":             model.state_dict(),
                    "optimizer":         optimizer.state_dict(),
                    "scheduler":         scheduler.state_dict(),
                },
                ckpt_path,
            )
            print(f"[{cfg.name}] Checkpoint → {ckpt_path}", flush=True)

        # Sync all ranks at checkpoint boundary to keep data streams aligned
        if world_size > 1 and step % args.checkpoint_steps == 0:
            dist.barrier()

    # --- final checkpoint + eval (rank 0 only) ------------------------------
    if is_main:
        final_ckpt = ckpt_dir / "final.pt"
        torch.save(
            {
                "step":              step,
                "tokens_seen":       tokens_seen,
                "cumulative_flops":  cumulative_flops,
                "model":             model.state_dict(),
                "optimizer":         optimizer.state_dict(),
                "scheduler":         scheduler.state_dict(),
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
            f"{ema_loss:.6f}", f"{final_eval:.6f}",
        ])
        csv_file.flush()
        csv_file.close()

        elapsed_total = time.time() - t0
        print(
            f"[{cfg.name}] Training complete.  "
            f"Final eval loss: {final_eval:.4f}.  "
            f"Total time: {elapsed_total/3600:.1f}h",
            flush=True,
        )

    _cleanup_ddp(world_size)
    return log_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train one Chinchilla comparison model on FineWeb sample-10BT.\n"
            "Launch with torchrun for multi-GPU: "
            "torchrun --standalone --nproc_per_node=8 train.py --model model1_125m"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        choices=["model1_125m", "model2_500m", "avg_125m_k2"],
        help="Which model config to train.",
    )
    p.add_argument("--batch_size",       type=int, default=16,
                   help="Per-GPU batch size. Global batch = batch_size × num_GPUs.")
    p.add_argument("--seq_len",          type=int, default=1024)
    p.add_argument("--device",           type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device for single-GPU mode. Ignored when torchrun is used.")
    p.add_argument("--log_steps",        type=int, default=1_000,
                   help="Log to CSV + print every N optimizer steps.")
    p.add_argument("--checkpoint_steps", type=int, default=50_000,
                   help="Save checkpoint every N optimizer steps.")
    p.add_argument("--eval_batches",     type=int, default=64,
                   help="Eval batches per evaluation pass (rank-0 only).")
    p.add_argument("--num_workers",      type=int, default=2,
                   help="DataLoader worker processes per rank.")
    p.add_argument("--tokenizer_name",   type=str,
                   default="EleutherAI/pythia-70m")
    p.add_argument("--resume",           action="store_true",
                   help="Resume from the latest checkpoint.")
    p.add_argument("--results_dir",      type=str, default=None,
                   help="Override results directory.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = get_config(args.model)

    results_dir = Path(args.results_dir) / cfg.name if args.results_dir else None
    log_path = train_model(cfg, args, results_dir=results_dir)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(f"Loss log saved to: {log_path}", flush=True)


if __name__ == "__main__":
    main()
