"""
Overlapping Window Averaging Analysis
======================================
Sweeps over combinations of (window_size, stride) to study how overlapping
windows affect embedding structure compared to non-overlapping (uniform)
averaging.

Each (window_size, stride) pair has an effective compression ratio = stride /
window_size.  A 2-D heatmap of key metrics vs (window, stride) is written to
the output directory.

Usage examples
--------------
# All defaults
python run_overlapping_analysis.py

# Custom windows and strides
python run_overlapping_analysis.py --window_sizes 4 8 --strides 1 2 4

# Quick smoke-test
python run_overlapping_analysis.py --num_sequences 50 --window_sizes 4
"""

import os
import sys
import argparse
import logging
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

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
from utils.averaging_methods.overlapping import (
    apply_overlapping_averaging,
    get_output_length,
    get_compression_ratio,
)

DEFAULT_WINDOW_SIZES = [2, 4, 8]
# Default strides: for each window_size we sweep [1, window_size//2, window_size]
# but they are generated dynamically below if not overridden


def build_sweep(window_sizes: List[int], strides: List[int]) -> List[Tuple[int, int]]:
    """
    Return all valid (window_size, stride) pairs from the provided lists.
    Pairs where stride > window_size are skipped (undefined).
    """
    pairs = []
    for ws in window_sizes:
        for st in strides:
            if 1 <= st <= ws:
                pairs.append((ws, st))
    return pairs


def plot_metric_heatmap(
    window_sizes: List[int],
    strides: List[int],
    values: Dict[Tuple[int, int], float],
    metric_name: str,
    layer_name: str,
    output_path: str,
) -> None:
    """
    Draw a 2-D heatmap of `metric_name` as a function of (window_size, stride).
    Cells for which no data exists are shown as NaN (blank).
    """
    matrix = np.full((len(window_sizes), len(strides)), np.nan)
    for wi, ws in enumerate(window_sizes):
        for si, st in enumerate(strides):
            if (ws, st) in values:
                matrix[wi, si] = values[(ws, st)]

    fig, ax = plt.subplots(figsize=(max(6, len(strides) + 1), max(4, len(window_sizes))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label=metric_name)
    ax.set_xticks(range(len(strides)))
    ax.set_xticklabels([str(s) for s in strides])
    ax.set_yticks(range(len(window_sizes)))
    ax.set_yticklabels([str(w) for w in window_sizes])
    ax.set_xlabel("Stride")
    ax.set_ylabel("Window size")
    ax.set_title(f"{metric_name} — {layer_name}")

    for wi in range(len(window_sizes)):
        for si in range(len(strides)):
            val = matrix[wi, si]
            if not np.isnan(val):
                ax.text(si, wi, f"{val:.3f}", ha="center", va="center",
                        color="white", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


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
    Apply each (window_size, stride) pair to every layer and run all 5 analyses.
    """
    # Generate strides list: if not supplied, build from window_sizes
    if args.strides:
        strides = sorted(set(args.strides))
    else:
        strides_set = set()
        for ws in args.window_sizes:
            strides_set.add(1)
            if ws > 2:
                strides_set.add(ws // 2)
            strides_set.add(ws)
        strides = sorted(strides_set)

    pairs = build_sweep(args.window_sizes, strides)

    if not pairs:
        logger.error("No valid (window_size, stride) pairs found. Exiting.")
        return {}

    logger.info(f"Sweeping {len(pairs)} (window_size, stride) pairs: {pairs}")

    all_rows: List[Dict] = []
    all_results: Dict = {}

    # For heatmap plotting we collect one scalar metric per pair per layer
    # key: layer_name → metric_name → (ws, st) → value
    heatmap_data: Dict[str, Dict[str, Dict]] = {}

    for ws, st in pairs:
        label = f"w{ws}_s{st}"
        compression = get_compression_ratio(ws, st)
        logger.info(f"\n{'='*60}")
        logger.info(f"window_size={ws}, stride={st}, compression={compression:.3f}")
        logger.info(f"{'='*60}")

        averaged_embeddings: Dict[str, np.ndarray] = {}
        for layer_name, orig_emb in original_embeddings.items():
            t = torch.from_numpy(orig_emb.astype(np.float32))
            batch, seq_len, dim = t.shape

            out_len = get_output_length(seq_len, ws, st)
            if out_len < 1:
                logger.warning(f"  {layer_name}: seq_len={seq_len} too short for w={ws}. Skipping.")
                continue

            averaged_t = apply_overlapping_averaging(t, window_size=ws, stride=st)
            averaged_embeddings[layer_name] = averaged_t.numpy()

        if not averaged_embeddings:
            logger.warning(f"No layers processed for w={ws}, s={st}. Skipping.")
            continue

        layer_results = run_analyses_for_averaged(
            original_embeddings,
            averaged_embeddings,
            k_label=label,
            output_dir=output_dir,
            logger=logger,
        )
        all_results[label] = layer_results

        extra = {
            "window_size": ws,
            "stride": st,
            "compression_ratio": round(compression, 4),
        }
        rows = flatten_results_to_rows(layer_results, method_name="overlapping", extra_meta=extra)
        all_rows.extend(rows)

        # Harvest scalar metrics for heatmap
        for layer_name, lr in layer_results.items():
            heatmap_data.setdefault(layer_name, {})
            for metric_key, nested_key, col in [
                ("variance",          "shrinkage",     "shrinkage_factor"),
                ("norm",              "shrinkage",     "shrinkage_factor"),
                ("information_theory","retention",     "retention_ratio"),
                ("spectral",          "energy_loss",   "total_energy_loss_percentage"),
                ("rank",              "rank_reduction","rank_reduction"),
            ]:
                val = (
                    lr.get(metric_key, {})
                      .get(nested_key, {})
                      .get(col, np.nan)
                )
                full_key = f"{metric_key}_{col}"
                heatmap_data[layer_name].setdefault(full_key, {})
                heatmap_data[layer_name][full_key][(ws, st)] = val

    # Export metrics
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    export_results_to_csv(
        all_rows,
        os.path.join(metrics_dir, "overlapping_metrics.csv"),
        logger,
    )
    export_results_to_json(
        all_results,
        os.path.join(metrics_dir, "overlapping_metrics.json"),
        logger,
    )
    create_summary_report(
        all_results,
        method_name="Overlapping Windows",
        output_path=os.path.join(output_dir, "overlapping_summary.md"),
        logger=logger,
    )

    # Save heatmap plots for the first layer only (embedding)
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    target_layer = "embedding" if "embedding" in heatmap_data else next(iter(heatmap_data), None)
    if target_layer:
        for metric_name, values in heatmap_data[target_layer].items():
            path = os.path.join(heatmap_dir, f"heatmap_{target_layer}_{metric_name}.png")
            try:
                plot_metric_heatmap(
                    args.window_sizes, strides, values,
                    metric_name, target_layer, path
                )
                logger.info(f"Heatmap saved: {path}")
            except Exception as exc:
                logger.warning(f"Could not save heatmap for {metric_name}: {exc}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Overlapping Window Averaging Analysis")
    parser.add_argument(
        "--window_sizes",
        nargs="+",
        type=int,
        default=DEFAULT_WINDOW_SIZES,
        help="Window sizes to sweep (default: 2 4 8)",
    )
    parser.add_argument(
        "--strides",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Stride values to test.  If omitted, strides are generated automatically "
            "as [1, window_size//2, window_size] for each window_size."
        ),
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=config.NUM_SEQUENCES,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(config.OUTPUT_DIR, "overlapping"),
    )
    parser.add_argument("--device", type=str, default=config.DEVICE)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(config.LOGS_DIR, prefix="overlapping")

    logger.info("=" * 70)
    logger.info("Overlapping Window Averaging Analysis")
    logger.info(f"  window_sizes : {args.window_sizes}")
    logger.info(f"  strides      : {args.strides or 'auto'}")
    logger.info(f"  sequences    : {args.num_sequences}")
    logger.info(f"  output_dir   : {args.output_dir}")
    logger.info(f"  device       : {args.device}")
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

    logger.info("\nOverlapping analysis complete.")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
