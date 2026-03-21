"""
Dynamic K Averaging Analysis
=============================
Sweeps over three grouping strategies — alternating, random, adaptive — and
measures how each affects variance, norm, information content, spectral energy
and embedding rank compared to the original (uncompressed) embeddings.

Usage examples
--------------
# All strategies with defaults
python run_dynamic_analysis.py

# Only alternating and random, custom pattern
python run_dynamic_analysis.py --strategies alternating random --pattern 2 3 4

# Override num_sequences for a quick smoke-test
python run_dynamic_analysis.py --num_sequences 50
"""

import os
import sys
import argparse
import logging
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import (
    load_pythia_model,
    setup_plot_style,
    setup_logging,
    collect_embeddings,
    run_analyses_for_averaged,
    flatten_results_to_rows,
    export_results_to_csv,
    export_results_to_json,
    create_summary_report,
)
from utils.averaging_methods.dynamic import (
    apply_dynamic_averaging,
    get_group_stats,
    DYNAMIC_STRATEGIES,
)


# ---------------------------------------------------------------------------
# Strategy configurations
# ---------------------------------------------------------------------------

def build_strategy_configs(args) -> List[Dict]:
    """
    Build the list of strategy configurations to sweep over.

    Each config dict has keys: strategy, label, kwargs.
    """
    configs = []

    for strategy in args.strategies:
        if strategy == "alternating":
            pattern = args.pattern
            label = f"alternating_{'_'.join(map(str, pattern))}"
            configs.append(
                dict(
                    strategy="alternating",
                    label=label,
                    kwargs=dict(pattern=pattern),
                )
            )

        elif strategy == "random":
            for k_min, k_max in [(2, 4), (2, 8)]:
                label = f"random_{k_min}_{k_max}"
                configs.append(
                    dict(
                        strategy="random",
                        label=label,
                        kwargs=dict(k_min=k_min, k_max=k_max, seed=config.RANDOM_SEED),
                    )
                )

        elif strategy == "adaptive":
            for threshold in [0.75, 0.85, 0.95]:
                label = f"adaptive_sim{int(threshold * 100)}"
                configs.append(
                    dict(
                        strategy="adaptive",
                        label=label,
                        kwargs=dict(k_min=2, k_max=6, high_sim_threshold=threshold),
                    )
                )

    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_method(
    original_embeddings: Dict[str, np.ndarray],
    output_dir: str,
    logger: logging.Logger,
    args,
) -> Dict:
    """
    Core logic: apply each dynamic strategy and run all 5 analyses.
    Returns nested results dict keyed by strategy label.
    """
    all_rows = []
    all_results = {}
    strategy_configs = build_strategy_configs(args)

    for cfg in strategy_configs:
        strategy = cfg["strategy"]
        label = cfg["label"]
        kwargs = cfg["kwargs"]

        logger.info(f"\n{'='*70}")
        logger.info(f"Dynamic strategy: {label}")
        logger.info(f"{'='*70}")

        averaged_embeddings: Dict[str, np.ndarray] = {}

        for layer_name, orig_emb in original_embeddings.items():
            t = torch.from_numpy(orig_emb.astype(np.float32))
            try:
                averaged_t, groups = apply_dynamic_averaging(t, strategy=strategy, **kwargs)
            except Exception as exc:
                logger.warning(f"  Skipping {layer_name}: {exc}")
                continue

            averaged_embeddings[layer_name] = averaged_t.numpy()

            if layer_name == "embedding":
                stats = get_group_stats(groups)
                logger.info(f"  Group stats: {stats}")

        if not averaged_embeddings:
            logger.error(f"No layers could be averaged for strategy {label}. Skipping.")
            continue

        layer_results = run_analyses_for_averaged(
            original_embeddings,
            averaged_embeddings,
            k_label=label,
            output_dir=output_dir,
            logger=logger,
        )
        all_results[label] = layer_results

        extra = {"strategy": strategy, "label": label}
        extra.update(kwargs)
        rows = flatten_results_to_rows(layer_results, method_name="dynamic", extra_meta=extra)
        all_rows.extend(rows)

    # Export
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    export_results_to_csv(
        all_rows,
        os.path.join(metrics_dir, "dynamic_metrics.csv"),
        logger,
    )
    export_results_to_json(
        all_results,
        os.path.join(metrics_dir, "dynamic_metrics.json"),
        logger,
    )
    create_summary_report(
        all_results,
        method_name="Dynamic K",
        output_path=os.path.join(output_dir, "dynamic_summary.md"),
        logger=logger,
    )

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Dynamic K Averaging Analysis")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DYNAMIC_STRATEGIES,
        choices=DYNAMIC_STRATEGIES,
        help="Which strategies to run",
    )
    parser.add_argument(
        "--pattern",
        nargs="+",
        type=int,
        default=[2, 3],
        help="Group-size pattern for the alternating strategy (default: 2 3)",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=config.NUM_SEQUENCES,
        help="Number of sequences to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(config.OUTPUT_DIR, "dynamic"),
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        help="Device to use (cuda/cpu)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(config.LOGS_DIR, prefix="dynamic")

    logger.info("=" * 70)
    logger.info("Dynamic K Averaging Analysis")
    logger.info(f"  strategies  : {args.strategies}")
    logger.info(f"  pattern     : {args.pattern}")
    logger.info(f"  sequences   : {args.num_sequences}")
    logger.info(f"  output_dir  : {args.output_dir}")
    logger.info(f"  device      : {args.device}")
    logger.info("=" * 70)

    setup_plot_style(config.PLOT_STYLE)

    logger.info("Loading model …")
    model, tokenizer = load_pythia_model(config.MODEL_NAME, args.device)

    original_embeddings = collect_embeddings(
        model=model,
        tokenizer=tokenizer,
        num_sequences=args.num_sequences,
        max_length=config.MAX_SEQUENCE_LENGTH,
        batch_size=config.BATCH_SIZE,
        device=args.device,
        logger=logger,
    )

    run_method(original_embeddings, args.output_dir, logger, args)

    logger.info("\nDynamic analysis complete.")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
