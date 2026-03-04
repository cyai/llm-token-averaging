"""
Token Averaging — Master Orchestration Script
==============================================
Loads the model and collects embeddings exactly once, then runs every
averaging method (uniform-k, dynamic, overlapping, weighted, learnable)
in sequence.  Produces per-method outputs plus a unified cross-method
comparison report.

Methods
-------
  uniform     – the existing analysis from run_all_analyses.py (k = powers of 2)
  dynamic     – alternating / random / adaptive group sizes
  overlapping – fixed window with variable stride
  weighted    – static weight schemes (uniform, linear, exponential, …)
  learnable   – content-dependent weights trained via reconstruction loss

Usage examples
--------------
# Run all methods (learnable included) with default config
python run_all_methods.py

# Run only fast methods (skip learnable which requires training)
python run_all_methods.py --skip_learnable

# Select specific methods
python run_all_methods.py --methods uniform dynamic weighted

# Quick smoke-test: small dataset, minimal config
python run_all_methods.py --num_sequences 50 --skip_learnable
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
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
)
from utils.embedding_extractor import apply_averaging
from utils.averaging_methods.dynamic import apply_dynamic_averaging, get_group_stats, DYNAMIC_STRATEGIES
from utils.averaging_methods.overlapping import apply_overlapping_averaging, get_output_length, get_compression_ratio
from utils.averaging_methods.weighted import apply_weighted_averaging, compute_weights, WEIGHT_SCHEMES
from utils.averaging_methods.learnable import train_learnable_averager, apply_trained_averager

AVAILABLE_METHODS = ["uniform", "dynamic", "overlapping", "weighted", "learnable"]


# ===========================================================================
# Per-method runners
# ===========================================================================

def run_uniform(
    original_embeddings: Dict[str, np.ndarray],
    output_dir: str,
    logger: logging.Logger,
    k_values: List[int],
) -> Dict:
    """Run the standard uniform-k averaging for a list of k values."""
    all_rows: List[Dict] = []
    all_results: Dict = {}

    for k in k_values:
        label = f"uniform_k{k}"
        logger.info(f"[uniform] k={k}")

        averaged: Dict[str, np.ndarray] = {}
        for layer_name, orig in original_embeddings.items():
            t = torch.from_numpy(orig.astype(np.float32))
            averaged[layer_name] = apply_averaging(t, k).numpy()

        layer_results = run_analyses_for_averaged(
            original_embeddings, averaged, k_label=label,
            output_dir=output_dir, logger=logger,
        )
        all_results[label] = layer_results
        all_rows.extend(
            flatten_results_to_rows(layer_results, "uniform", {"k": k})
        )

    return all_results, all_rows


def run_dynamic(
    original_embeddings: Dict[str, np.ndarray],
    output_dir: str,
    logger: logging.Logger,
) -> Dict:
    """Run all dynamic-k strategies."""
    from run_dynamic_analysis import build_strategy_configs

    all_rows: List[Dict] = []
    all_results: Dict = {}

    dummy_args = SimpleNamespace(
        strategies=DYNAMIC_STRATEGIES,
        pattern=[2, 3],
    )
    strategy_configs = build_strategy_configs(dummy_args)

    for cfg in strategy_configs:
        strategy = cfg["strategy"]
        label = cfg["label"]
        kwargs = cfg["kwargs"]
        logger.info(f"[dynamic] {label}")

        averaged: Dict[str, np.ndarray] = {}
        for layer_name, orig in original_embeddings.items():
            t = torch.from_numpy(orig.astype(np.float32))
            try:
                averaged_t, _ = apply_dynamic_averaging(t, strategy=strategy, **kwargs)
                averaged[layer_name] = averaged_t.numpy()
            except Exception as exc:
                logger.warning(f"  {layer_name} skipped: {exc}")

        if not averaged:
            continue

        layer_results = run_analyses_for_averaged(
            original_embeddings, averaged, k_label=label,
            output_dir=output_dir, logger=logger,
        )
        all_results[label] = layer_results
        extra = {"strategy": strategy, "label": label}
        extra.update(kwargs)
        all_rows.extend(flatten_results_to_rows(layer_results, "dynamic", extra))

    return all_results, all_rows


def run_overlapping(
    original_embeddings: Dict[str, np.ndarray],
    output_dir: str,
    logger: logging.Logger,
    window_sizes: List[int] = None,
) -> Dict:
    """Run overlapping-window averaging."""
    if window_sizes is None:
        window_sizes = [2, 4, 8]

    all_rows: List[Dict] = []
    all_results: Dict = {}

    for ws in window_sizes:
        strides = sorted({1, ws // 2, ws} - {0})
        for st in strides:
            if st > ws:
                continue
            label = f"w{ws}_s{st}"
            compression = get_compression_ratio(ws, st)
            logger.info(f"[overlapping] {label}  compression={compression:.3f}")

            averaged: Dict[str, np.ndarray] = {}
            for layer_name, orig in original_embeddings.items():
                t = torch.from_numpy(orig.astype(np.float32))
                _, seq_len, _ = t.shape
                if get_output_length(seq_len, ws, st) < 1:
                    continue
                averaged[layer_name] = apply_overlapping_averaging(
                    t, window_size=ws, stride=st
                ).numpy()

            if not averaged:
                continue

            layer_results = run_analyses_for_averaged(
                original_embeddings, averaged, k_label=label,
                output_dir=output_dir, logger=logger,
            )
            all_results[label] = layer_results
            all_rows.extend(
                flatten_results_to_rows(
                    layer_results, "overlapping",
                    {"window_size": ws, "stride": st, "compression_ratio": round(compression, 4)},
                )
            )

    return all_results, all_rows


def run_weighted(
    original_embeddings: Dict[str, np.ndarray],
    output_dir: str,
    logger: logging.Logger,
    k_values: List[int] = None,
) -> Dict:
    """Run all static weight schemes."""
    if k_values is None:
        k_values = [2, 4, 8, 16]

    all_rows: List[Dict] = []
    all_results: Dict = {}

    for k in k_values:
        for scheme in WEIGHT_SCHEMES:
            label = f"{scheme}_k{k}"
            weights = compute_weights(k, scheme)
            logger.info(f"[weighted] {label}")

            averaged: Dict[str, np.ndarray] = {}
            for layer_name, orig in original_embeddings.items():
                t = torch.from_numpy(orig.astype(np.float32))
                averaged[layer_name] = apply_weighted_averaging(t, k=k, weights=weights).numpy()

            layer_results = run_analyses_for_averaged(
                original_embeddings, averaged, k_label=label,
                output_dir=output_dir, logger=logger,
            )
            all_results[label] = layer_results
            all_rows.extend(
                flatten_results_to_rows(layer_results, "weighted", {"k": k, "scheme": scheme})
            )

    return all_results, all_rows


def run_learnable(
    original_embeddings: Dict[str, np.ndarray],
    output_dir: str,
    logger: logging.Logger,
    k_values: List[int] = None,
    n_epochs: int = None,
    lr: float = None,
    device: str = "cpu",
) -> Dict:
    """Train LearnableAverager and run all analyses."""
    if k_values is None:
        k_values = [2, 4, 8, 16]
    if n_epochs is None:
        n_epochs = config.LEARNABLE_EPOCHS
    if lr is None:
        lr = config.LEARNABLE_LR

    all_rows: List[Dict] = []
    all_results: Dict = {}

    first_layer = next(iter(original_embeddings))
    hidden_dim = original_embeddings[first_layer].shape[-1]

    all_layer_embs = np.concatenate(list(original_embeddings.values()), axis=0)

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for k in k_values:
        logger.info(f"[learnable] k={k} — training …")

        averager, loss_history = train_learnable_averager(
            embeddings=all_layer_embs,
            k=k,
            hidden_dim=hidden_dim,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=config.LEARNABLE_BATCH_SIZE,
            device=device,
            logger=logger,
        )
        logger.info(f"  Final loss: {loss_history[-1]:.6f}")

        learned_averaged: Dict[str, np.ndarray] = {}
        for layer_name, orig in original_embeddings.items():
            learned_averaged[layer_name] = apply_trained_averager(
                orig, averager, device=device, batch_size=32
            )

        label = f"learned_k{k}"
        layer_results = run_analyses_for_averaged(
            original_embeddings, learned_averaged, k_label=label,
            output_dir=output_dir, logger=logger,
        )
        all_results[label] = layer_results
        all_rows.extend(
            flatten_results_to_rows(
                layer_results, "learnable",
                {"k": k, "n_epochs": n_epochs, "final_mse_loss": round(loss_history[-1], 8)},
            )
        )

    return all_results, all_rows


# ===========================================================================
# Cross-method comparison
# ===========================================================================

def create_comparison_report(
    all_method_rows: Dict[str, List[Dict]],
    all_method_results: Dict[str, Dict],
    output_path: str,
    logger: logging.Logger,
) -> None:
    """
    Write a unified Markdown report comparing all methods.

    Includes:
    - Configuration summary
    - Per-method row counts
    - Top-level comparison table (mean of key metrics across layers, first config per method)
    - Links to per-method subdirectory reports
    """
    all_rows: List[Dict] = []
    for rows in all_method_rows.values():
        all_rows.extend(rows)

    if not all_rows:
        logger.warning("No data rows available for comparison report.")
        return

    df = pd.DataFrame(all_rows)

    with open(output_path, "w") as fh:
        fh.write("# Token Averaging — Cross-Method Comparison Report\n\n")
        fh.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        fh.write("## Configuration\n\n")
        fh.write(f"- Model: `{config.MODEL_NAME}`\n")
        fh.write(f"- Dataset: `{config.DATASET_NAME}` (`{config.DATASET_CONFIG}`)\n\n")

        fh.write("## Methods run\n\n")
        for method, rows in all_method_rows.items():
            n_configs = len(rows) // max(len(set(r.get("layer", "x") for r in rows)), 1)
            fh.write(f"- **{method}**: {len(rows)} rows  ({n_configs} configurations)\n")
        fh.write("\n")

        # Summarise by method: mean over all layers and all configs
        fh.write("## Summary table (mean across all layers and configurations)\n\n")
        numeric_cols = [
            "variance_shrinkage_factor",
            "norm_shrinkage_factor",
            "info_retention_ratio",
            "spectral_total_energy_loss_pct",
            "rank_reduction",
        ]
        available_cols = [c for c in numeric_cols if c in df.columns]

        if available_cols and "method" in df.columns:
            summary = (
                df.groupby("method")[available_cols]
                .mean()
                .round(4)
                .reset_index()
            )
            fh.write(summary.to_markdown(index=False))
            fh.write("\n\n")

        fh.write("## Per-method output directories\n\n")
        for method in all_method_rows:
            fh.write(f"- `outputs/{method}/`\n")

        fh.write(
            "\n## Metric descriptions\n\n"
            "| Metric | Ideal value | Interpretation |\n"
            "|--------|-------------|----------------|\n"
            "| `variance_shrinkage_factor` | close to 1 | Little variance lost |\n"
            "| `norm_shrinkage_factor`     | close to 1 | Norm preserved |\n"
            "| `info_retention_ratio`      | close to 1 | Information retained |\n"
            "| `spectral_total_energy_loss_pct` | close to 0 | Little spectral energy lost |\n"
            "| `rank_reduction`            | close to 0 | Intrinsic dimensionality preserved |\n\n"
        )

    logger.info(f"Comparison report written to {output_path}")


def plot_method_comparison(
    all_method_rows: Dict[str, List[Dict]],
    output_path: str,
    logger: logging.Logger,
) -> None:
    """
    Bar chart comparing mean key metrics across methods.
    """
    all_rows: List[Dict] = []
    for rows in all_method_rows.values():
        all_rows.extend(rows)
    if not all_rows:
        return

    df = pd.DataFrame(all_rows)
    if "method" not in df.columns:
        return

    metrics = {
        "info_retention_ratio":           "Information retention",
        "variance_shrinkage_factor":       "Variance preservation",
        "norm_shrinkage_factor":           "Norm preservation",
        "spectral_total_energy_loss_pct":  "Spectral energy loss %",
        "rank_reduction":                  "Rank reduction",
    }
    available = {k: v for k, v in metrics.items() if k in df.columns}
    if not available:
        return

    methods = df["method"].unique().tolist()
    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), squeeze=False)

    for col_idx, (col, title) in enumerate(available.items()):
        ax = axes[0][col_idx]
        means = [df[df["method"] == m][col].mean() for m in methods]
        ax.bar(methods, means)
        ax.set_title(title, fontsize=9)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Cross-method metric comparison (mean across all layers / configs)", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Comparison bar chart saved to {output_path}")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Token Averaging — master script for all averaging methods"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=AVAILABLE_METHODS,
        choices=AVAILABLE_METHODS,
        help=(
            "Which methods to run.  Default: all methods.  "
            "Example: --methods uniform dynamic weighted"
        ),
    )
    parser.add_argument(
        "--skip_learnable",
        action="store_true",
        help="Shortcut to exclude the learnable method (equivalent to removing it from --methods)",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=config.NUM_SEQUENCES,
        help="Number of sequences for embedding extraction (default from config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.OUTPUT_DIR,
        help="Root output directory (sub-dirs are created per method)",
    )
    parser.add_argument("--device", type=str, default=config.DEVICE)
    # Fine-grained overrides
    parser.add_argument(
        "--uniform_k_max",
        type=int,
        default=16,
        help="Max k for uniform sweep (powers of 2 up to this value)",
    )
    parser.add_argument(
        "--learnable_epochs",
        type=int,
        default=config.LEARNABLE_EPOCHS,
    )
    parser.add_argument(
        "--learnable_lr",
        type=float,
        default=config.LEARNABLE_LR,
    )
    args = parser.parse_args()

    methods = list(args.methods)
    if args.skip_learnable and "learnable" in methods:
        methods.remove("learnable")

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(config.LOGS_DIR, prefix="all_methods")

    logger.info("=" * 70)
    logger.info("Token Averaging — Master Script")
    logger.info(f"  methods      : {methods}")
    logger.info(f"  num_sequences: {args.num_sequences}")
    logger.info(f"  output_dir   : {args.output_dir}")
    logger.info(f"  device       : {args.device}")
    logger.info("=" * 70)

    setup_plot_style(config.PLOT_STYLE)

    logger.info("\nLoading model …")
    model, tokenizer = load_pythia_model(config.MODEL_NAME, args.device)

    logger.info("\nCollecting embeddings (once for all methods) …")
    original_embeddings = collect_embeddings(
        model=model,
        tokenizer=tokenizer,
        num_sequences=args.num_sequences,
        max_length=config.MAX_SEQUENCE_LENGTH,
        batch_size=config.BATCH_SIZE,
        device=args.device,
        logger=logger,
    )

    all_method_rows: Dict[str, List[Dict]] = {}
    all_method_results: Dict[str, Dict] = {}

    # ---- Uniform ----------------------------------------------------------
    if "uniform" in methods:
        logger.info("\n" + "=" * 70)
        logger.info("METHOD: uniform")
        k_values = []
        k = 1
        while k <= args.uniform_k_max:
            k_values.append(k)
            k *= 2
        method_dir = os.path.join(args.output_dir, "uniform")
        os.makedirs(method_dir, exist_ok=True)
        results, rows = run_uniform(original_embeddings, method_dir, logger, k_values)
        all_method_results["uniform"] = results
        all_method_rows["uniform"] = rows
        export_results_to_csv(rows, os.path.join(method_dir, "metrics", "uniform_metrics.csv"), logger)
        export_results_to_json(results, os.path.join(method_dir, "metrics", "uniform_metrics.json"), logger)

    # ---- Dynamic ----------------------------------------------------------
    if "dynamic" in methods:
        logger.info("\n" + "=" * 70)
        logger.info("METHOD: dynamic")
        method_dir = os.path.join(args.output_dir, "dynamic")
        os.makedirs(method_dir, exist_ok=True)
        results, rows = run_dynamic(original_embeddings, method_dir, logger)
        all_method_results["dynamic"] = results
        all_method_rows["dynamic"] = rows
        export_results_to_csv(rows, os.path.join(method_dir, "metrics", "dynamic_metrics.csv"), logger)
        export_results_to_json(results, os.path.join(method_dir, "metrics", "dynamic_metrics.json"), logger)

    # ---- Overlapping ------------------------------------------------------
    if "overlapping" in methods:
        logger.info("\n" + "=" * 70)
        logger.info("METHOD: overlapping")
        method_dir = os.path.join(args.output_dir, "overlapping")
        os.makedirs(method_dir, exist_ok=True)
        results, rows = run_overlapping(original_embeddings, method_dir, logger)
        all_method_results["overlapping"] = results
        all_method_rows["overlapping"] = rows
        export_results_to_csv(rows, os.path.join(method_dir, "metrics", "overlapping_metrics.csv"), logger)
        export_results_to_json(results, os.path.join(method_dir, "metrics", "overlapping_metrics.json"), logger)

    # ---- Weighted ---------------------------------------------------------
    if "weighted" in methods:
        logger.info("\n" + "=" * 70)
        logger.info("METHOD: weighted")
        method_dir = os.path.join(args.output_dir, "weighted")
        os.makedirs(method_dir, exist_ok=True)
        results, rows = run_weighted(original_embeddings, method_dir, logger)
        all_method_results["weighted"] = results
        all_method_rows["weighted"] = rows
        export_results_to_csv(rows, os.path.join(method_dir, "metrics", "weighted_metrics.csv"), logger)
        export_results_to_json(results, os.path.join(method_dir, "metrics", "weighted_metrics.json"), logger)

    # ---- Learnable --------------------------------------------------------
    if "learnable" in methods:
        logger.info("\n" + "=" * 70)
        logger.info("METHOD: learnable")
        method_dir = os.path.join(args.output_dir, "learnable")
        os.makedirs(method_dir, exist_ok=True)
        results, rows = run_learnable(
            original_embeddings, method_dir, logger,
            k_values=[2, 4, 8],
            n_epochs=args.learnable_epochs,
            lr=args.learnable_lr,
            device=args.device,
        )
        all_method_results["learnable"] = results
        all_method_rows["learnable"] = rows
        export_results_to_csv(rows, os.path.join(method_dir, "metrics", "learnable_metrics.csv"), logger)
        export_results_to_json(results, os.path.join(method_dir, "metrics", "learnable_metrics.json"), logger)

    # ---- Cross-method comparison -----------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Generating cross-method comparison …")

    comparison_report_path = os.path.join(args.output_dir, "comparison_report.md")
    create_comparison_report(all_method_rows, all_method_results, comparison_report_path, logger)

    comparison_chart_path = os.path.join(args.output_dir, "comparison_chart.png")
    plot_method_comparison(all_method_rows, comparison_chart_path, logger)

    all_combined_rows: List[Dict] = []
    for rows in all_method_rows.values():
        all_combined_rows.extend(rows)
    export_results_to_csv(
        all_combined_rows,
        os.path.join(args.output_dir, "all_methods_metrics.csv"),
        logger,
    )

    logger.info("\n" + "=" * 70)
    logger.info("All methods complete.")
    logger.info(f"  Comparison report : {comparison_report_path}")
    logger.info(f"  Comparison chart  : {comparison_chart_path}")
    logger.info(f"  Combined CSV      : {os.path.join(args.output_dir, 'all_methods_metrics.csv')}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
