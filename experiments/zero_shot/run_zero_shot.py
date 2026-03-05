"""
Experiment 1 — Zero-shot perplexity evaluation.

Loads a pretrained Pythia model (default: pythia-410m) with frozen weights,
inserts each averaging configuration between the embedding and the first
transformer block via AveragedLanguageModel, and evaluates perplexity on
the WikiText-103 test split.

No weight updates occur — this purely measures how well the pretrained model
handles averaged token embeddings it was never trained on.

Usage
-----
python experiments/zero_shot/run_zero_shot.py \
    --configs baseline_k1 uniform_k2 uniform_k4 uniform_k8 \
              weighted_exponential_k2 weighted_exponential_k4 \
              overlap_w2s2 overlap_w2s1 overlap_w4s2 \
              dynamic_alt23 learnable_k2 learnable_k4 \
    --num_sequences 500 \
    --device cuda

python experiments/zero_shot/run_zero_shot.py --all_configs --device cuda
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from experiments.shared.averaged_lm import (
    AveragedLanguageModel,
    build_method_config,
    get_all_config_names,
)
from experiments.shared.eval_utils import (
    compute_perplexity,
    make_result_row,
    save_results,
    setup_exp_logging,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Zero-shot perplexity evaluation with token averaging"
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Method config names to evaluate (default: a representative subset)",
    )
    p.add_argument(
        "--all_configs",
        action="store_true",
        help="Evaluate every config in the registry",
    )
    p.add_argument(
        "--model_name",
        default=config.EXPERIMENT_MODEL_ZEROSHOT,
        help="HuggingFace model identifier",
    )
    p.add_argument(
        "--num_sequences",
        type=int,
        default=config.EXPERIMENT_EVAL_SEQUENCES,
        help="Number of WikiText-103 test sequences to evaluate on",
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
        "--output_dir",
        default=os.path.join(config.EXPERIMENT_OUTPUT_DIR, "zero_shot"),
    )
    p.add_argument(
        "--learnable_checkpoint_dir",
        default=os.path.join(config.OUTPUT_DIR, "learnable", "checkpoints"),
        help="Directory with averager_k{k}.pt files (for learnable_* configs)",
    )
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Default config subset (representative, not exhaustive)
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS = [
    "baseline_k1",
    "uniform_k2",
    "uniform_k4",
    "uniform_k8",
    "weighted_exponential_k2",
    "weighted_exponential_k4",
    "weighted_exponential_k8",
    "overlap_w2s1",
    "overlap_w2s2",
    "overlap_w4s2",
    "dynamic_alt23",
    "dynamic_rnd24",
    "learnable_k2",
    "learnable_k4",
    "learnable_k8",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log = setup_exp_logging(
        os.path.join(config.OUTPUT_DIR, "logs"),
        prefix="zero_shot",
    )
    log.info("=" * 60)
    log.info("Zero-shot perplexity experiment")
    log.info(f"  model      : {args.model_name}")
    log.info(f"  device     : {args.device}")
    log.info(f"  sequences  : {args.num_sequences}")
    log.info(f"  output_dir : {args.output_dir}")
    log.info("=" * 60)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    # Decide which configs to run
    if args.all_configs:
        configs_to_run = get_all_config_names()
    elif args.configs:
        configs_to_run = args.configs
    else:
        configs_to_run = DEFAULT_CONFIGS

    log.info(f"Configs to evaluate ({len(configs_to_run)}): {configs_to_run}")

    # Load tokenizer once
    log.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    csv_path = os.path.join(args.output_dir, "results.csv")

    # Reload the base model fresh for each config to avoid state bleed
    for cfg_name in configs_to_run:
        log.info(f"\n--- Config: {cfg_name} ---")

        # Build method config
        try:
            method_cfg = build_method_config(
                cfg_name,
                learnable_checkpoint_dir=args.learnable_checkpoint_dir,
            )
        except ValueError as e:
            log.error(f"Skipping {cfg_name}: {e}")
            continue

        # Load fresh pretrained model (frozen)
        log.info(f"Loading {args.model_name} ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        base_model.eval()
        base_model.to(args.device)
        for param in base_model.parameters():
            param.requires_grad_(False)

        # Wrap with averaging
        model = AveragedLanguageModel(base_model, method_cfg)
        model.to(args.device)

        # If learnable: move the averager to device but keep frozen
        if method_cfg.learnable_module is not None:
            method_cfg.learnable_module.to(args.device)
            for param in method_cfg.learnable_module.parameters():
                param.requires_grad_(False)

        # Evaluate
        log.info(f"Evaluating PPL (method={method_cfg.method_family}, k={method_cfg.nominal_k}) ...")
        ppl = compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            num_sequences=args.num_sequences,
            device=args.device,
            split="test",
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        log.info(f"  PPL = {ppl:.4f}")

        row = make_result_row(
            experiment="zero_shot",
            config_name=cfg_name,
            method_family=method_cfg.method_family,
            nominal_k=method_cfg.nominal_k,
            compression_ratio=method_cfg.compression_ratio,
            ppl=ppl,
            model_name=args.model_name,
            num_sequences=args.num_sequences,
            train_steps=0,
        )
        rows.append(row)

        # Save incrementally so partial results are preserved on interruption
        save_results(rows, csv_path)

        # Release GPU memory before next config
        del model
        del base_model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    log.info(f"\nAll done.  Results saved to {csv_path}")

    # Print a quick summary table
    log.info("\nSummary:")
    log.info(f"{'Config':<30} {'Method':<12} {'k':>4} {'Compression':>12} {'PPL':>10}")
    log.info("-" * 72)
    for r in sorted(rows, key=lambda x: (x["method"], x["nominal_k"])):
        log.info(
            f"{r['config_name']:<30} {r['method']:<12} {r['nominal_k']:>4} "
            f"{r['compression_ratio']:>11.1%} {r['ppl']:>10.2f}"
        )


if __name__ == "__main__":
    main()
