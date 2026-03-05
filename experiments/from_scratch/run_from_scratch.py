"""
Experiment 2 — Train from scratch with token averaging (OLM architecture).

Builds a fresh transformer using OLM (OpenLanguageModel) blocks, wraps it
with OLMAveragedLanguageModel for the chosen averaging configuration, trains
on WikiText-103, then evaluates perplexity on the test split.

Model architecture (via OLM's LM block):
  d_model  = config.EXPERIMENT_OLM_D_MODEL   (default 512)
  n_heads  = config.EXPERIMENT_OLM_N_HEADS   (default 8)
  n_layers = config.EXPERIMENT_OLM_N_LAYERS  (default 6)
  context  = config.MAX_SEQUENCE_LENGTH      (default 512)
  ≈ 70M parameters — comparable to Pythia-70m

Tokeniser: EleutherAI/pythia-70m (GPT-NeoX BPE, 50 257 vocab tokens) is
used for consistency with the other two experiments.  Only the tokeniser is
loaded from HuggingFace — no pretrained weights.

Optimiser: OLM's AdamW (from olm.train.optim) with cosine-warmup scheduling.
Training loop: custom step-based loop (OLM Trainer is bypassed because our
averaging label construction requires a non-standard forward pass).

Usage
-----
# Representative subset (default)
python experiments/from_scratch/run_from_scratch.py --device cuda

# Specific configs
python experiments/from_scratch/run_from_scratch.py \\
    --configs baseline_k1 uniform_k2 uniform_k4 \\
              weighted_exponential_k2 overlap_w2s2 learnable_k2 \\
    --train_steps 5000 --lr 5e-4 --device cuda

# All configs
python experiments/from_scratch/run_from_scratch.py --all_configs --device cuda
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from utils.data_loader import get_data_iterator
from experiments.shared.averaged_lm import (
    build_method_config,
    get_all_config_names,
)
from experiments.shared.olm_model import OLMTransformerBody, OLMAveragedLanguageModel
from experiments.shared.eval_utils import (
    compute_perplexity,
    make_result_row,
    save_results,
    save_checkpoint,
    setup_exp_logging,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train from scratch with token averaging (OLM architecture)"
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Method config names to train (default: representative subset)",
    )
    p.add_argument(
        "--all_configs",
        action="store_true",
        help="Train every config in the registry",
    )
    p.add_argument(
        "--tokenizer_name",
        default=config.EXPERIMENT_MODEL_SCRATCH,
        help="HuggingFace tokeniser to use (weights are NOT loaded — architecture "
             "is built with OLM from scratch)",
    )
    # OLM architecture
    p.add_argument("--d_model",  type=int, default=config.EXPERIMENT_OLM_D_MODEL)
    p.add_argument("--n_heads",  type=int, default=config.EXPERIMENT_OLM_N_HEADS)
    p.add_argument("--n_layers", type=int, default=config.EXPERIMENT_OLM_N_LAYERS)
    # Training
    p.add_argument("--train_steps",   type=int,   default=config.EXPERIMENT_TRAIN_STEPS)
    p.add_argument("--lr",            type=float, default=config.EXPERIMENT_LR_SCRATCH)
    p.add_argument("--warmup_steps",  type=int,   default=config.EXPERIMENT_WARMUP_STEPS)
    p.add_argument("--grad_clip",     type=float, default=config.EXPERIMENT_GRAD_CLIP)
    p.add_argument("--batch_size",    type=int,   default=config.EXPERIMENT_BATCH_SIZE)
    p.add_argument("--max_length",    type=int,   default=config.MAX_SEQUENCE_LENGTH)
    p.add_argument(
        "--train_sequences",
        type=int,
        default=config.EXPERIMENT_TRAIN_SEQUENCES,
        help="Number of training sequences per data-iterator pass",
    )
    p.add_argument("--eval_sequences",  type=int, default=config.EXPERIMENT_EVAL_SEQUENCES)
    p.add_argument("--checkpoint_every", type=int, default=1000)
    p.add_argument(
        "--output_dir",
        default=os.path.join(config.EXPERIMENT_OUTPUT_DIR, "from_scratch"),
    )
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--seed",   type=int, default=config.RANDOM_SEED)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Default representative subset
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS = [
    "baseline_k1",
    "uniform_k2",
    "uniform_k4",
    "uniform_k8",
    "weighted_exponential_k2",
    "weighted_exponential_k4",
    "overlap_w2s2",
    "overlap_w4s2",
    "dynamic_alt23",
    "learnable_k2",
    "learnable_k4",
]


# ---------------------------------------------------------------------------
# OLM AdamW helper
# ---------------------------------------------------------------------------

def _build_optimizer(model: OLMAveragedLanguageModel, lr: float, weight_decay: float = 0.1):
    """
    Build an AdamW optimiser using OLM's implementation where available,
    falling back to torch.optim.AdamW.

    OLM's AdamW is designed to be passed to the Trainer as a class, but we
    can also instantiate it directly since it extends torch.optim.AdamW.
    The parameter grouping (exclude 1-D params from weight decay) mirrors
    OLM's Trainer._configure_optimizer logic.
    """
    try:
        from olm.train.optim import AdamW as OLMAdamW
        optimizer_cls = OLMAdamW
    except ImportError:
        optimizer_cls = torch.optim.AdamW

    # Separate weight-decay and no-decay parameter groups (OLM best practice)
    decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and p.dim() >= 2   # matrices — apply decay
    ]
    no_decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and p.dim() < 2    # biases, LayerNorm weights — no decay
    ]
    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return optimizer_cls(param_groups, lr=lr)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_config(
    cfg_name: str,
    args,
    tokenizer,
    log,
) -> dict:
    """
    Build a fresh OLM model for one averaging configuration, train it, and
    return a result row dict.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Config  : {cfg_name}")
    log.info(f"OLM arch: d_model={args.d_model}  n_heads={args.n_heads}  n_layers={args.n_layers}")
    log.info(f"{'='*60}")

    # Per-config seeding for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    # --- Build OLM architecture ---
    log.info("Building OLM transformer from scratch ...")
    backbone = OLMTransformerBody(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_length=args.max_length,
    )
    backbone.to(args.device)

    # --- Wrap with averaging ---
    method_cfg = build_method_config(cfg_name)
    model = OLMAveragedLanguageModel(backbone, method_cfg)
    model.to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    # --- Optimiser + cosine-warmup scheduler (OLM training best practice) ---
    optimizer = _build_optimizer(model, lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )

    # Checkpoint dir for this config
    ckpt_dir = os.path.join(args.output_dir, "checkpoints", cfg_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Training ---
    model.train()
    step = 0
    losses: list[float] = []

    log.info(f"Training for {args.train_steps} steps ...")

    while step < args.train_steps:
        train_iter = get_data_iterator(
            tokenizer=tokenizer,
            num_sequences=args.train_sequences,
            max_length=args.max_length,
            batch_size=args.batch_size,
            split="train",
        )

        with tqdm(train_iter, desc=f"  step {step}", leave=False) as pbar:
            for batch in pbar:
                if step >= args.train_steps:
                    break

                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)

                try:
                    loss, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                except Exception as exc:
                    log.warning(f"  Step {step}: skipping batch — {exc}")
                    continue

                if not torch.isfinite(loss):
                    log.warning(f"  Step {step}: non-finite loss, skipping")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                losses.append(loss.item())
                step += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": step})

                if step % args.checkpoint_every == 0:
                    ckpt_path = os.path.join(ckpt_dir, f"step_{step}.pt")
                    save_checkpoint(model, ckpt_path, step)
                    mean_recent = sum(losses[-100:]) / len(losses[-100:])
                    log.info(f"  Step {step:>6}  loss={mean_recent:.4f}")

    # Save final checkpoint
    save_checkpoint(model, os.path.join(ckpt_dir, "final.pt"), step)

    # --- Evaluate ---
    log.info("Evaluating perplexity on test split ...")
    ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        num_sequences=args.eval_sequences,
        device=args.device,
        split="test",
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    log.info(f"  PPL = {ppl:.4f}")

    result = make_result_row(
        experiment="from_scratch",
        config_name=cfg_name,
        method_family=method_cfg.method_family,
        nominal_k=method_cfg.nominal_k,
        compression_ratio=method_cfg.compression_ratio,
        ppl=ppl,
        model_name=f"OLM-d{args.d_model}h{args.n_heads}l{args.n_layers}",
        num_sequences=args.eval_sequences,
        train_steps=step,
        extra={
            "final_train_loss": round(
                sum(losses[-100:]) / max(len(losses[-100:]), 1), 4
            ),
            "olm_d_model":  args.d_model,
            "olm_n_heads":  args.n_heads,
            "olm_n_layers": args.n_layers,
        },
    )

    # Release GPU memory before next config
    del model, backbone
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log = setup_exp_logging(
        os.path.join(config.OUTPUT_DIR, "logs"),
        prefix="from_scratch",
    )
    log.info("=" * 60)
    log.info("From-scratch training experiment (OLM architecture)")
    log.info(f"  tokenizer  : {args.tokenizer_name}")
    log.info(f"  d_model    : {args.d_model}")
    log.info(f"  n_heads    : {args.n_heads}")
    log.info(f"  n_layers   : {args.n_layers}")
    log.info(f"  train_steps: {args.train_steps}")
    log.info(f"  lr         : {args.lr}")
    log.info(f"  device     : {args.device}")
    log.info(f"  output_dir : {args.output_dir}")
    log.info("=" * 60)

    if args.all_configs:
        configs_to_run = get_all_config_names()
    elif args.configs:
        configs_to_run = args.configs
    else:
        configs_to_run = DEFAULT_CONFIGS

    log.info(f"Configs to train ({len(configs_to_run)}): {configs_to_run}")

    # Load tokeniser once (GPT-NeoX BPE from Pythia family)
    log.info(f"Loading tokeniser: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    csv_path = os.path.join(args.output_dir, "results.csv")
    rows = []

    for cfg_name in configs_to_run:
        try:
            row = train_one_config(cfg_name, args, tokenizer, log)
            rows.append(row)
            save_results(rows, csv_path)
        except Exception as exc:
            log.error(f"Config {cfg_name} failed: {exc}", exc_info=True)
            continue

    log.info(f"\nAll done.  Results saved to {csv_path}")

    log.info("\nSummary:")
    log.info(f"{'Config':<30} {'Method':<12} {'k':>4} {'PPL':>10} {'Loss':>10}")
    log.info("-" * 72)
    for r in sorted(rows, key=lambda x: (x["method"], x["nominal_k"])):
        loss_str = str(r.get("final_train_loss", "N/A"))
        log.info(
            f"{r['config_name']:<30} {r['method']:<12} {r['nominal_k']:>4} "
            f"{r['ppl']:>10.2f} {loss_str:>10}"
        )


if __name__ == "__main__":
    main()
