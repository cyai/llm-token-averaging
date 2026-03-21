"""
Weighted Average Analysis
==========================
Applies five static weight schemes (uniform, linear, exponential, gaussian,
triangular) over non-overlapping windows of configurable sizes and measures
how each scheme preserves embedding structure.

Also reports the weight entropy for each scheme — a measure of how
concentrated the weighting is.  Lower entropy means the scheme is closer
to selecting a single token; higher entropy approaches uniform averaging.

Usage examples
--------------
# All schemes, default k values
python run_weighted_analysis.py

# Specific schemes and k values
python run_weighted_analysis.py --schemes linear exponential --k_values 4 8 16

# Quick smoke-test
python run_weighted_analysis.py --num_sequences 50 --k_values 4
"""

import os
import sys
import argparse
import logging
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

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
from utils.averaging_methods.weighted import (
    apply_weighted_averaging,
    compute_weights,
    compute_weight_entropy,
    WEIGHT_SCHEMES,
)

DEFAULT_K_VALUES = [2, 4, 8, 16]


def plot_weight_profiles(
    k_values: List[int],
    schemes: List[str],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """
    Save a grid plot showing the weight vector for each (k, scheme) combination.
    """
    fig, axes = plt.subplots(
        len(k_values),
        len(schemes),
        figsize=(3 * len(schemes), 2.5 * len(k_values)),
        squeeze=False,
    )
    for ri, k in enumerate(k_values):
        for ci, scheme in enumerate(schemes):
            w = compute_weights(k, scheme)
            ax = axes[ri][ci]
            ax.bar(range(k), w)
            ax.set_title(f"{scheme} k={k}", fontsize=8)
            ax.set_ylim(0, w.max() * 1.2 + 0.01)
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=6)

    plt.suptitle("Weight profiles per (k, scheme)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "weight_profiles.png")
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Weight profiles saved to {path}")


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
    Apply each (k, scheme) combination and run all 5 analyses.
    """
    all_rows: List[Dict] = []
    all_results: Dict = {}

    # Save weight profile visualisation first
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_weight_profiles(args.k_values, args.schemes, plots_dir, logger)

    # Log weight entropies
    logger.info("\nWeight entropies:")
    for scheme in args.schemes:
        for k in args.k_values:
            w = compute_weights(k, scheme)
            H = compute_weight_entropy(w)
            H_max = float(np.log(k)) if k > 1 else 0.0
            logger.info(
                f"  scheme={scheme:12s}  k={k:3d}  H={H:.4f}  H_max={H_max:.4f}"
                f"  normalised={H/H_max:.3f}" if H_max > 0 else f"  H={H:.4f}"
            )

    for k in args.k_values:
        for scheme in args.schemes:
            label = f"{scheme}_k{k}"
            weights = compute_weights(k, scheme)
            H = compute_weight_entropy(weights)

            logger.info(f"\n{'='*60}")
            logger.info(f"scheme={scheme}, k={k}, weight_entropy={H:.4f}")
            logger.info(f"  weights: {np.round(weights, 4).tolist()}")
            logger.info(f"{'='*60}")

            averaged_embeddings: Dict[str, np.ndarray] = {}
            for layer_name, orig_emb in original_embeddings.items():
                t = torch.from_numpy(orig_emb.astype(np.float32))
                averaged_t = apply_weighted_averaging(t, k=k, weights=weights)
                averaged_embeddings[layer_name] = averaged_t.numpy()

            layer_results = run_analyses_for_averaged(
                original_embeddings,
                averaged_embeddings,
                k_label=label,
                output_dir=output_dir,
                logger=logger,
            )
            all_results[label] = layer_results

            extra = {
                "scheme": scheme,
                "k": k,
                "weight_entropy": round(H, 6),
                "weight_entropy_normalised": round(
                    H / float(np.log(k)) if k > 1 else 1.0, 4
                ),
            }
            rows = flatten_results_to_rows(
                layer_results, method_name="weighted", extra_meta=extra
            )
            all_rows.extend(rows)

    # Export
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    export_results_to_csv(
        all_rows,
        os.path.join(metrics_dir, "weighted_metrics.csv"),
        logger,
    )
    export_results_to_json(
        all_results,
        os.path.join(metrics_dir, "weighted_metrics.json"),
        logger,
    )
    create_summary_report(
        all_results,
        method_name="Weighted Average",
        output_path=os.path.join(output_dir, "weighted_summary.md"),
        logger=logger,
        extra_info=(
            "Weight entropy measures how concentrated the weighting is.  "
            "Lower entropy ≈ closer to selecting a single token.  "
            "Higher entropy ≈ closer to uniform averaging (maximum H = ln k)."
        ),
    )

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Weighted Average Analysis")
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=WEIGHT_SCHEMES,
        choices=WEIGHT_SCHEMES,
        help="Weight schemes to run",
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=DEFAULT_K_VALUES,
        help="Window sizes to analyse (default: 2 4 8 16)",
    )
    parser.add_argument("--num_sequences", type=int, default=config.NUM_SEQUENCES)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(config.OUTPUT_DIR, "weighted"),
    )
    parser.add_argument("--device", type=str, default=config.DEVICE)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(config.LOGS_DIR, prefix="weighted")

    logger.info("=" * 70)
    logger.info("Weighted Average Analysis")
    logger.info(f"  schemes    : {args.schemes}")
    logger.info(f"  k_values   : {args.k_values}")
    logger.info(f"  sequences  : {args.num_sequences}")
    logger.info(f"  output_dir : {args.output_dir}")
    logger.info(f"  device     : {args.device}")
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

    logger.info("\nWeighted analysis complete.")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
