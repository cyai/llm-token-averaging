"""
Learnable Weighted Average Analysis
=====================================
Trains a small content-dependent scoring network (LearnableAverager) for each
k value and measures how the learned averaging compares to uniform averaging.

Training objective
------------------
For each k, a LearnableAverager + ReconstructionDecoder are jointly trained to
minimise MSE reconstruction loss.  The averager learns to assign importance
weights to each token within its k-token window so that the weighted average
retains maximum information.  Training runs entirely in embedding space —
no re-running of the base LM is required.

Outputs per k
-------------
  • All 5 standard analysis metrics (variance / norm / info / spectral / rank)
  • Per-epoch training loss curve (PNG + CSV)
  • Learned weight profile: mean attention weight per window position
  • Comparison table: learned averaging vs. uniform averaging

Usage examples
--------------
# All k values, all layers (slow)
python run_learnable_analysis.py

# Quick smoke-test: k=2 only, 50 sequences, 1 epoch
python run_learnable_analysis.py --k_values 2 --num_sequences 50 --n_epochs 1

# Use GPU
python run_learnable_analysis.py --device cuda
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional

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
from utils.averaging_methods.learnable import (
    train_learnable_averager,
    apply_trained_averager,
)
from utils.embedding_extractor import apply_averaging

DEFAULT_K_VALUES = [2, 4, 8, 16]


def plot_loss_curve(
    loss_history: List[float],
    k: int,
    output_path: str,
) -> None:
    """Save a training loss curve PNG."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"LearnableAverager training loss  (k={k})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_weight_profile(
    mean_weights: np.ndarray,
    k: int,
    output_path: str,
) -> None:
    """
    Plot the mean attention weight assigned to each window position.
    Also shows the uniform baseline (1/k) as a dashed line.
    """
    fig, ax = plt.subplots(figsize=(max(5, k // 2 + 2), 4))
    ax.bar(range(k), mean_weights, label="learned")
    ax.axhline(1.0 / k, color="red", linestyle="--", alpha=0.8, label=f"uniform (1/{k})")
    ax.set_xlabel("Position within window")
    ax.set_ylabel("Mean attention weight")
    ax.set_title(f"Learned weight profile  (k={k})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def build_comparison_table(
    results_learned: Dict,
    results_uniform: Dict,
    k: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Build a table comparing key scalar metrics between learned and uniform averaging.
    One row per layer, columns: layer | metric_learned | metric_uniform | delta.
    """
    records = []
    metrics = [
        ("variance",           "shrinkage",      "shrinkage_factor"),
        ("norm",               "shrinkage",      "shrinkage_factor"),
        ("information_theory", "retention",      "retention_ratio"),
        ("spectral",           "energy_loss",    "total_energy_loss_percentage"),
        ("rank",               "rank_reduction", "rank_reduction"),
    ]
    for layer in results_learned:
        if layer not in results_uniform:
            continue
        row = {"layer": layer, "k": k}
        for mod, nested, col in metrics:
            v_learned = (
                results_learned[layer]
                .get(mod, {})
                .get(nested, {})
                .get(col, np.nan)
            )
            v_uniform = (
                results_uniform[layer]
                .get(mod, {})
                .get(nested, {})
                .get(col, np.nan)
            )
            key = f"{mod}_{col}"
            row[f"{key}_learned"] = v_learned
            row[f"{key}_uniform"] = v_uniform
            try:
                row[f"{key}_delta"] = float(v_learned) - float(v_uniform)
            except (TypeError, ValueError):
                row[f"{key}_delta"] = np.nan
        records.append(row)

    df = pd.DataFrame(records)
    logger.info(f"\nComparison table (k={k}):\n{df.to_string(index=False)}\n")
    return df


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
    For each k: train LearnableAverager, apply it, run 5 analyses,
    then compare with uniform averaging.
    """
    all_rows: List[Dict] = []
    all_results: Dict = {}
    comparison_dfs: List[pd.DataFrame] = []

    plots_dir = os.path.join(output_dir, "plots")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Determine hidden_dim from the first layer
    first_layer = next(iter(original_embeddings))
    hidden_dim = original_embeddings[first_layer].shape[-1]
    logger.info(f"Detected hidden_dim = {hidden_dim}")

    # Concatenate all layers for training (shared averager per k)
    all_layer_embs = np.concatenate(
        list(original_embeddings.values()), axis=0
    )  # [n_layers * n_seq, seq_len, dim]
    logger.info(
        f"Training data shape (all layers combined): {all_layer_embs.shape}"
    )

    for k in args.k_values:
        logger.info(f"\n{'='*70}")
        logger.info(f"k = {k}")
        logger.info(f"{'='*70}")

        # ---- Train --------------------------------------------------------
        logger.info(f"Training LearnableAverager (k={k}) …")
        averager, loss_history = train_learnable_averager(
            embeddings=all_layer_embs,
            k=k,
            hidden_dim=hidden_dim,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=config.LEARNABLE_BATCH_SIZE,
            device=args.device,
            logger=logger,
        )

        # Save loss curve
        loss_png = os.path.join(plots_dir, f"loss_curve_k{k}.png")
        plot_loss_curve(loss_history, k, loss_png)

        loss_csv = os.path.join(metrics_dir, f"loss_curve_k{k}.csv")
        pd.DataFrame(
            {"epoch": range(1, len(loss_history) + 1), "mse_loss": loss_history}
        ).to_csv(loss_csv, index=False)
        logger.info(f"  Loss curve saved: {loss_png}")

        # ---- Learned weight profile (from embedding layer) ----------------
        emb_layer_data = original_embeddings.get(
            "embedding", original_embeddings[first_layer]
        )
        mean_weights = averager.get_effective_weights(
            emb_layer_data, device=args.device
        )
        weight_png = os.path.join(plots_dir, f"weight_profile_k{k}.png")
        plot_weight_profile(mean_weights, k, weight_png)
        logger.info(f"  Mean learned weights: {np.round(mean_weights, 4).tolist()}")

        # ---- Apply learned averager per layer ----------------------------
        learned_averaged: Dict[str, np.ndarray] = {}
        for layer_name, orig_emb in original_embeddings.items():
            learned_averaged[layer_name] = apply_trained_averager(
                orig_emb, averager, device=args.device, batch_size=32
            )

        label_learned = f"learned_k{k}"
        layer_results_learned = run_analyses_for_averaged(
            original_embeddings,
            learned_averaged,
            k_label=label_learned,
            output_dir=output_dir,
            logger=logger,
        )
        all_results[label_learned] = layer_results_learned

        # ---- Uniform baseline for the same k -----------------------------
        uniform_averaged: Dict[str, np.ndarray] = {}
        for layer_name, orig_emb in original_embeddings.items():
            t = torch.from_numpy(orig_emb.astype(np.float32))
            uniform_averaged[layer_name] = apply_averaging(t, k).numpy()

        label_uniform = f"uniform_k{k}"
        layer_results_uniform = run_analyses_for_averaged(
            original_embeddings,
            uniform_averaged,
            k_label=label_uniform,
            output_dir=output_dir,
            logger=logger,
        )
        all_results[label_uniform] = layer_results_uniform

        # ---- Comparison table --------------------------------------------
        cmp_df = build_comparison_table(
            layer_results_learned, layer_results_uniform, k, logger
        )
        comparison_dfs.append(cmp_df)
        cmp_df.to_csv(
            os.path.join(metrics_dir, f"comparison_k{k}.csv"), index=False
        )

        # ---- Accumulate rows ---------------------------------------------
        for label, layer_results, extra_method in [
            (label_learned, layer_results_learned, "learned"),
            (label_uniform, layer_results_uniform, "uniform_baseline"),
        ]:
            rows = flatten_results_to_rows(
                layer_results,
                method_name=f"learnable_{extra_method}",
                extra_meta={
                    "k": k,
                    "variant": extra_method,
                    "final_mse_loss": round(loss_history[-1], 8),
                },
            )
            all_rows.extend(rows)

    # ---- Export -----------------------------------------------------------
    export_results_to_csv(
        all_rows,
        os.path.join(metrics_dir, "learnable_metrics.csv"),
        logger,
    )
    export_results_to_json(
        all_results,
        os.path.join(metrics_dir, "learnable_metrics.json"),
        logger,
    )

    if comparison_dfs:
        combined_cmp = pd.concat(comparison_dfs, ignore_index=True)
        combined_cmp.to_csv(
            os.path.join(metrics_dir, "comparison_all_k.csv"), index=False
        )
        logger.info("Combined comparison table saved.")

    create_summary_report(
        all_results,
        method_name="Learnable Weighted Average",
        output_path=os.path.join(output_dir, "learnable_summary.md"),
        logger=logger,
        extra_info=(
            "The LearnableAverager uses a shared linear scoring head (dim → 1) "
            "applied to every token within a k-token window.  Scores are passed "
            "through softmax to produce normalised attention weights.  Training "
            "minimises MSE reconstruction loss using a lightweight decoder.\n\n"
            "Comparison tables (learned vs. uniform) are in `metrics/comparison_k*.csv`."
        ),
    )

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Learnable Weighted Average Analysis")
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=DEFAULT_K_VALUES,
        help="Window sizes to train and analyse (default: 2 4 8 16)",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=config.LEARNABLE_EPOCHS,
        help="Training epochs per k value",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.LEARNABLE_LR,
        help="Learning rate for AdamW",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=config.LEARNABLE_TRAIN_SEQUENCES,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(config.OUTPUT_DIR, "learnable"),
    )
    parser.add_argument("--device", type=str, default=config.DEVICE)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(config.LOGS_DIR, prefix="learnable")

    logger.info("=" * 70)
    logger.info("Learnable Weighted Average Analysis")
    logger.info(f"  k_values   : {args.k_values}")
    logger.info(f"  n_epochs   : {args.n_epochs}")
    logger.info(f"  lr         : {args.lr}")
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

    logger.info("\nLearnable analysis complete.")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
