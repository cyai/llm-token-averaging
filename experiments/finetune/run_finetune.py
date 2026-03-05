"""
Experiment 3 — Finetune a pretrained model with token averaging.

Loads a pretrained Pythia model (default: pythia-410m), records its baseline
perplexity at k=1 (standard LM), then for each averaging configuration:
  1. Wraps the pretrained model with AveragedLanguageModel
  2. Records pre-finetune PPL (zero-shot, no weight updates)
  3. Finetunes for --finetune_steps steps on WikiText-103 train split
  4. Records post-finetune PPL on the test split

The result CSV contains both ppl_before and ppl_after so you can measure
how much the model adapts from zero-shot performance.

Usage
-----
# Default subset
python experiments/finetune/run_finetune.py --device cuda

# Specific configs
python experiments/finetune/run_finetune.py \
    --configs baseline_k1 uniform_k2 uniform_k4 uniform_k8 \
              weighted_exponential_k2 overlap_w2s2 learnable_k2 \
    --finetune_steps 2000 --lr 5e-5 --device cuda

# All configs
python experiments/finetune/run_finetune.py --all_configs --device cuda
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from utils.data_loader import get_data_iterator
from experiments.shared.averaged_lm import (
    AveragedLanguageModel,
    build_method_config,
    get_all_config_names,
)
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
        description="Finetune a pretrained model with token averaging"
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Method config names to finetune (default: representative subset)",
    )
    p.add_argument(
        "--all_configs",
        action="store_true",
        help="Finetune every config in the registry",
    )
    p.add_argument(
        "--model_name",
        default=config.EXPERIMENT_MODEL_FINETUNE,
        help="HuggingFace model identifier (pretrained weights loaded)",
    )
    p.add_argument(
        "--finetune_steps",
        type=int,
        default=config.EXPERIMENT_FINETUNE_STEPS,
    )
    p.add_argument(
        "--lr",
        type=float,
        default=config.EXPERIMENT_LR_FINETUNE,
    )
    p.add_argument(
        "--warmup_steps",
        type=int,
        default=config.EXPERIMENT_WARMUP_STEPS,
    )
    p.add_argument(
        "--grad_clip",
        type=float,
        default=config.EXPERIMENT_GRAD_CLIP,
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=config.EXPERIMENT_BATCH_SIZE,
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=config.MAX_SEQUENCE_LENGTH,
    )
    p.add_argument(
        "--train_sequences",
        type=int,
        default=config.EXPERIMENT_TRAIN_SEQUENCES,
    )
    p.add_argument(
        "--eval_sequences",
        type=int,
        default=config.EXPERIMENT_EVAL_SEQUENCES,
    )
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join(config.EXPERIMENT_OUTPUT_DIR, "finetune"),
    )
    p.add_argument(
        "--learnable_checkpoint_dir",
        default=os.path.join(config.OUTPUT_DIR, "learnable", "checkpoints"),
        help="Directory with averager_k{k}.pt files (initial weights for learnable configs)",
    )
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Default subset
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS = [
    "baseline_k1",
    "uniform_k2",
    "uniform_k4",
    "uniform_k8",
    "weighted_exponential_k2",
    "weighted_exponential_k4",
    "weighted_exponential_k8",
    "overlap_w2s2",
    "overlap_w4s2",
    "dynamic_alt23",
    "learnable_k2",
    "learnable_k4",
    "learnable_k8",
]


# ---------------------------------------------------------------------------
# Single config finetuning
# ---------------------------------------------------------------------------

def finetune_one_config(
    cfg_name: str,
    args,
    tokenizer,
    log,
) -> dict:
    """
    Finetune one averaging configuration and return a result row.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Config: {cfg_name}")
    log.info(f"{'='*60}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    # Build method config (load learnable checkpoint for initial weights)
    method_cfg = build_method_config(
        cfg_name,
        learnable_checkpoint_dir=args.learnable_checkpoint_dir,
    )

    # Load pretrained model
    log.info(f"Loading pretrained {args.model_name} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    base_model.to(args.device)

    # Wrap with averaging
    model = AveragedLanguageModel(base_model, method_cfg)
    model.to(args.device)

    # Move learnable module to device if present
    if method_cfg.learnable_module is not None:
        method_cfg.learnable_module.to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    # ---- Step 1: Record zero-shot (pre-finetune) PPL ----
    log.info("Recording pre-finetune PPL ...")
    ppl_before = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        num_sequences=args.eval_sequences,
        device=args.device,
        split="test",
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    log.info(f"  PPL before finetune = {ppl_before:.4f}")

    # ---- Step 2: Finetune ----
    optimizer = torch.optim.AdamW(
        model.parameters_to_train(),
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.finetune_steps,
    )

    ckpt_dir = os.path.join(args.output_dir, "checkpoints", cfg_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    model.train()
    step = 0
    losses = []

    log.info(f"Finetuning for {args.finetune_steps} steps ...")

    while step < args.finetune_steps:
        train_iter = get_data_iterator(
            tokenizer=tokenizer,
            num_sequences=args.train_sequences,
            max_length=args.max_length,
            batch_size=args.batch_size,
            split="train",
        )

        with tqdm(train_iter, desc=f"  step {step}", leave=False) as pbar:
            for batch in pbar:
                if step >= args.finetune_steps:
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
                    mean_recent = sum(losses[-50:]) / len(losses[-50:])
                    log.info(f"  Step {step:>6}  loss={mean_recent:.4f}")

    # Save final checkpoint
    save_checkpoint(model, os.path.join(ckpt_dir, "final.pt"), step)

    # ---- Step 3: Post-finetune PPL ----
    log.info("Evaluating post-finetune PPL ...")
    ppl_after = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        num_sequences=args.eval_sequences,
        device=args.device,
        split="test",
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    log.info(f"  PPL after finetune  = {ppl_after:.4f}")
    log.info(f"  PPL delta           = {ppl_after - ppl_before:+.4f}")

    result = make_result_row(
        experiment="finetune",
        config_name=cfg_name,
        method_family=method_cfg.method_family,
        nominal_k=method_cfg.nominal_k,
        compression_ratio=method_cfg.compression_ratio,
        ppl=ppl_after,
        model_name=args.model_name,
        num_sequences=args.eval_sequences,
        train_steps=step,
        ppl_before=ppl_before,
        extra={"final_train_loss": round(sum(losses[-50:]) / max(len(losses[-50:]), 1), 4)},
    )

    del model
    del base_model
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
        prefix="finetune",
    )
    log.info("=" * 60)
    log.info("Finetune experiment")
    log.info(f"  model          : {args.model_name}")
    log.info(f"  finetune_steps : {args.finetune_steps}")
    log.info(f"  lr             : {args.lr}")
    log.info(f"  device         : {args.device}")
    log.info(f"  output_dir     : {args.output_dir}")
    log.info("=" * 60)

    if args.all_configs:
        configs_to_run = get_all_config_names()
    elif args.configs:
        configs_to_run = args.configs
    else:
        configs_to_run = DEFAULT_CONFIGS

    log.info(f"Configs to finetune ({len(configs_to_run)}): {configs_to_run}")

    log.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    csv_path = os.path.join(args.output_dir, "results.csv")
    rows = []

    for cfg_name in configs_to_run:
        try:
            row = finetune_one_config(cfg_name, args, tokenizer, log)
            rows.append(row)
            save_results(rows, csv_path)
        except Exception as exc:
            log.error(f"Config {cfg_name} failed: {exc}", exc_info=True)
            continue

    log.info(f"\nAll done.  Results saved to {csv_path}")

    # Summary table
    log.info("\nSummary:")
    log.info(f"{'Config':<30} {'k':>4} {'PPL before':>12} {'PPL after':>10} {'Delta':>8}")
    log.info("-" * 72)
    for r in sorted(rows, key=lambda x: (x["method"], x["nominal_k"])):
        before_str = f"{r.get('ppl_before', 'N/A'):>12}" if isinstance(r.get("ppl_before"), float) else f"{'N/A':>12}"
        delta_str = f"{r.get('ppl_delta', 0.0):>+8.2f}" if isinstance(r.get("ppl_delta"), float) else f"{'N/A':>8}"
        log.info(
            f"{r['config_name']:<30} {r['nominal_k']:>4} "
            f"{before_str} {r['ppl']:>10.2f} {delta_str}"
        )


if __name__ == "__main__":
    main()
