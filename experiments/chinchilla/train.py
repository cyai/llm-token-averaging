"""
Unified training script for the Chinchilla FLOPs comparison experiment.

Trains any one of the three models (model1_125m, model2_500m, avg_125m_k2)
on FineWeb sample-10BT for 10B tokens, logging loss and cumulative FLOPs
at regular intervals so that the empirical training curves can be plotted
against theoretical Chinchilla scaling predictions.

Features
--------
* Single function `train_model(cfg, args)` handles all three model variants
* FLOPs accounting: standard models count 6·N·D, averaging model counts 6·N·D/k
* CSV log: step, tokens_seen, cumulative_flops, train_loss, eval_loss
* Gradient checkpointing for model2_500m to fit in 24 GB VRAM
* Checkpoint every `args.checkpoint_steps` steps, resumable via --resume
* nohup-safe single-line stdout progress every `args.log_steps` steps

Usage
-----
    python experiments/chinchilla/train.py \
        --model model1_125m \
        --batch_size 8 \
        --device cuda

    python experiments/chinchilla/train.py \
        --model model2_500m \
        --batch_size 4 \
        --device cuda \
        --resume
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
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
# Helpers
# ---------------------------------------------------------------------------

def _build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.1):
    """AdamW with OLM-style parameter grouping (1-D params skip weight decay)."""
    try:
        from olm.train.optim import AdamW as OLMAdamW
        optimizer_cls = OLMAdamW
    except ImportError:
        optimizer_cls = torch.optim.AdamW

    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    return optimizer_cls(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def _enable_gradient_checkpointing(model: OLMTransformerBody) -> None:
    """
    Enable gradient checkpointing on the transformer body by monkey-patching
    its forward method.  OLM's body block is a sequential chain of sub-blocks;
    we wrap each child's forward call with torch.utils.checkpoint.checkpoint.
    """
    body = model.body

    if hasattr(body, "gradient_checkpointing_enable"):
        body.gradient_checkpointing_enable()
        return

    # Wrap each sequential child
    original_forward = body.forward

    def _checkpointed_forward(x: torch.Tensor) -> torch.Tensor:
        for module in body.children():
            x = gradient_checkpoint(module, x, use_reentrant=False)
        return x

    # Only replace if the body has children (i.e. is nn.Sequential or similar)
    children = list(body.children())
    if children:
        body.forward = _checkpointed_forward


@torch.no_grad()
def _eval_loss(
    model: nn.Module,
    eval_dl,
    device: str,
    max_batches: int = 64,
    is_averaged: bool = False,
) -> float:
    """Compute mean cross-entropy loss on a finite eval dataloader."""
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
            # Standard model: causal LM loss over the full sequence
            logits = model(input_ids)                        # [B, T, V]
            labels = input_ids[:, 1:].contiguous()          # [B, T-1]
            logits = logits[:, :-1].contiguous()            # [B, T-1, V]
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
    Train a single model config end-to-end.

    Args:
        cfg         : ModelConfig describing the model architecture and budget
        args        : parsed CLI namespace (batch_size, device, seq_len, …)
        results_dir : root directory for results; defaults to
                      experiments/chinchilla/results/<cfg.name>

    Returns:
        Path to the loss_log.csv file.
    """
    # --- directories --------------------------------------------------------
    if results_dir is None:
        results_dir = _ROOT / "experiments" / "chinchilla" / "results" / cfg.name
    results_dir = Path(results_dir)
    ckpt_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "loss_log.csv"

    # --- tokenizer ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- data ---------------------------------------------------------------
    print(f"[{cfg.name}] Building FineWeb dataloaders …", flush=True)
    train_dl, eval_dl = build_dataloaders(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        num_workers=args.num_workers,
    )

    # --- model --------------------------------------------------------------
    is_averaged = cfg.averaging_k > 1

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
        model = backbone

    model.to(args.device)

    if cfg.grad_checkpoint:
        print(f"[{cfg.name}] Enabling gradient checkpointing …", flush=True)
        _enable_gradient_checkpointing(backbone)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{cfg.name}] Trainable parameters: {n_params:,} ({n_params/1e6:.1f}M)",
          flush=True)

    # --- optimiser + scheduler ---------------------------------------------
    total_steps = estimate_total_batches(cfg.target_tokens, args.seq_len, args.batch_size)
    optimizer = _build_optimizer(model, lr=cfg.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    # --- resume logic -------------------------------------------------------
    start_step = 0
    tokens_seen = 0
    cumulative_flops = 0.0

    if args.resume:
        ckpts = sorted(ckpt_dir.glob("step_*.pt"))
        if ckpts:
            latest = ckpts[-1]
            print(f"[{cfg.name}] Resuming from {latest} …", flush=True)
            state = torch.load(latest, map_location=args.device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_step = state["step"]
            tokens_seen = state["tokens_seen"]
            cumulative_flops = state["cumulative_flops"]
            print(f"[{cfg.name}] Resumed at step={start_step:,} "
                  f"tokens={tokens_seen:,}", flush=True)
        else:
            print(f"[{cfg.name}] --resume set but no checkpoint found; "
                  f"starting from scratch.", flush=True)

    # --- CSV log setup ------------------------------------------------------
    csv_mode = "a" if (args.resume and log_path.exists()) else "w"
    csv_file = open(log_path, csv_mode, newline="")
    csv_writer = csv.writer(csv_file)
    if csv_mode == "w":
        csv_writer.writerow(
            ["step", "tokens_seen", "cumulative_flops", "train_loss", "eval_loss"]
        )
        csv_file.flush()

    # FLOPs per training step (forward + backward, standard rule-of-thumb = 6·N)
    # For averaging k=2, the transformer processes seq_len/k positions per sequence
    effective_seq = args.seq_len / cfg.averaging_k
    flops_per_step = 6.0 * n_params * args.batch_size * effective_seq

    # --- training loop ------------------------------------------------------
    model.train()
    step = start_step
    ema_loss = None
    ema_alpha = 0.98

    tokens_per_step = args.batch_size * args.seq_len
    train_iter = iter(train_dl)

    print(
        f"[{cfg.name}] Starting training: "
        f"{total_steps:,} steps × {tokens_per_step:,} tokens/step "
        f"= {cfg.target_tokens/1e9:.1f}B tokens",
        flush=True,
    )
    t0 = time.time()

    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        input_ids = batch.to(args.device)

        optimizer.zero_grad(set_to_none=True)

        if is_averaged:
            loss, _ = model(input_ids)
        else:
            logits = model(input_ids)                      # [B, T, V]
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

        step += 1
        tokens_seen += tokens_per_step
        cumulative_flops += flops_per_step

        loss_val = loss.item()
        if ema_loss is None:
            ema_loss = loss_val
        else:
            ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * loss_val

        # --- logging every log_steps steps ----------------------------------
        if step % args.log_steps == 0:
            eval_loss_val = _eval_loss(
                model, eval_dl, args.device,
                max_batches=args.eval_batches,
                is_averaged=is_averaged,
            )
            csv_writer.writerow([
                step, tokens_seen, f"{cumulative_flops:.6e}",
                f"{ema_loss:.6f}", f"{eval_loss_val:.6f}",
            ])
            csv_file.flush()

            elapsed = time.time() - t0
            tokens_per_sec = tokens_seen / elapsed
            progress_pct = 100.0 * tokens_seen / cfg.target_tokens
            eta_h = ((cfg.target_tokens - tokens_seen) / max(tokens_per_sec, 1)) / 3600

            print(
                f"[{cfg.name}] step={step:>8,} | "
                f"tokens={tokens_seen/1e9:.3f}B/{cfg.target_tokens/1e9:.0f}B "
                f"({progress_pct:.1f}%) | "
                f"loss={ema_loss:.4f} | eval={eval_loss_val:.4f} | "
                f"flops={cumulative_flops:.2e} | "
                f"tok/s={tokens_per_sec:,.0f} | ETA={eta_h:.1f}h",
                flush=True,
            )

        # --- checkpoint every checkpoint_steps steps -----------------------
        if step % args.checkpoint_steps == 0:
            ckpt_path = ckpt_dir / f"step_{step:08d}.pt"
            torch.save(
                {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "cumulative_flops": cumulative_flops,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                ckpt_path,
            )
            print(f"[{cfg.name}] Checkpoint saved → {ckpt_path}", flush=True)

    # --- final checkpoint + eval -------------------------------------------
    final_ckpt = ckpt_dir / "final.pt"
    torch.save(
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "cumulative_flops": cumulative_flops,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        final_ckpt,
    )

    final_eval = _eval_loss(
        model, eval_dl, args.device,
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
        f"[{cfg.name}] Training complete. "
        f"Final eval loss: {final_eval:.4f}. "
        f"Total time: {elapsed_total/3600:.1f}h",
        flush=True,
    )
    return log_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train one Chinchilla comparison model on FineWeb sample-10BT."
    )
    p.add_argument(
        "--model",
        required=True,
        choices=["model1_125m", "model2_500m", "avg_125m_k2"],
        help="Which model config to train.",
    )
    p.add_argument("--batch_size",    type=int, default=8)
    p.add_argument("--seq_len",       type=int, default=1024)
    p.add_argument("--device",        type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_steps",     type=int, default=1_000,
                   help="Log to CSV and print every N steps.")
    p.add_argument("--checkpoint_steps", type=int, default=50_000,
                   help="Save checkpoint every N steps.")
    p.add_argument("--eval_batches",  type=int, default=64,
                   help="Number of eval batches per evaluation pass.")
    p.add_argument("--num_workers",   type=int, default=0,
                   help="DataLoader worker processes.")
    p.add_argument("--tokenizer_name", type=str,
                   default="EleutherAI/pythia-70m",
                   help="HuggingFace tokenizer identifier.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest checkpoint in the results dir.")
    p.add_argument("--results_dir", type=str, default=None,
                   help="Override output directory (default: experiments/chinchilla/results/<model>).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.model)

    results_dir = Path(args.results_dir) / cfg.name if args.results_dir else None
    log_path = train_model(cfg, args, results_dir=results_dir)
    print(f"Loss log saved to: {log_path}", flush=True)


if __name__ == "__main__":
    main()
