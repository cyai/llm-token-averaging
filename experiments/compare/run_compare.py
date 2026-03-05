"""
Experiment 4 — Unified comparison across all three experiment types.

Reads:
  outputs/experiments/zero_shot/results.csv
  outputs/experiments/from_scratch/results.csv
  outputs/experiments/finetune/results.csv

Produces:
  outputs/experiments/comparison/comparison_table.md   — wide-format Markdown table
  outputs/experiments/comparison/ppl_by_k_{method}.png — PPL vs k per method, 3 lines
  outputs/experiments/comparison/ppl_by_method_k{k}.png — PPL vs method per k, grouped bars

Usage
-----
python experiments/compare/run_compare.py

python experiments/compare/run_compare.py \
    --results_dir outputs/experiments/ \
    --output_dir  outputs/experiments/comparison/
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Aggregate experiment results into comparison plots")
    p.add_argument(
        "--results_dir",
        default=config.EXPERIMENT_OUTPUT_DIR,
        help="Directory containing zero_shot/, finetune/, from_scratch/ subdirs",
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join(config.EXPERIMENT_OUTPUT_DIR, "comparison"),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=config.FIGURE_DPI,
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_csv(path: str, experiment: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"[WARN] Not found: {path} — skipping {experiment}")
        return None
    df = pd.read_csv(path)
    df["experiment"] = experiment
    return df


def load_all(results_dir: str) -> pd.DataFrame:
    frames = []
    for exp in ("zero_shot", "finetune", "from_scratch"):
        csv_path = os.path.join(results_dir, exp, "results.csv")
        df = load_csv(csv_path, exp)
        if df is not None:
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No results CSVs found under {results_dir}. "
            "Run at least one experiment first."
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

EXPERIMENT_LABEL = {
    "zero_shot":    "Zero-shot",
    "finetune":     "Finetune",
    "from_scratch": "From scratch",
}

def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide-format table: rows = (method, nominal_k), columns = PPL per experiment.
    Also includes compression_ratio and ppl_before/ppl_after for finetune.
    """
    pivot_ppl = df.pivot_table(
        index=["method", "nominal_k", "compression_ratio"],
        columns="experiment",
        values="ppl",
        aggfunc="mean",
    ).reset_index()

    # Rename experiment columns
    pivot_ppl.columns.name = None
    rename_map = {e: f"ppl_{e}" for e in ("zero_shot", "finetune", "from_scratch")}
    pivot_ppl = pivot_ppl.rename(columns=rename_map)

    # Add finetune ppl_before if available
    ft_df = df[df["experiment"] == "finetune"]
    if "ppl_before" in ft_df.columns:
        before = ft_df.groupby(["method", "nominal_k"])["ppl_before"].mean().reset_index()
        before = before.rename(columns={"ppl_before": "ppl_finetune_before"})
        pivot_ppl = pivot_ppl.merge(before, on=["method", "nominal_k"], how="left")

    # Compute PPL ratio vs zero-shot baseline (k=1)
    baseline_rows = df[(df["experiment"] == "zero_shot") & (df["nominal_k"] == 1)]
    if not baseline_rows.empty:
        baseline_ppl = baseline_rows["ppl"].mean()
        for col in ("ppl_zero_shot", "ppl_finetune", "ppl_from_scratch"):
            if col in pivot_ppl.columns:
                pivot_ppl[col.replace("ppl_", "ratio_")] = (
                    pivot_ppl[col] / baseline_ppl
                ).round(4)

    pivot_ppl = pivot_ppl.sort_values(["method", "nominal_k"]).reset_index(drop=True)
    return pivot_ppl


def write_markdown_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    cols = list(df.columns)
    lines = []
    lines.append(f"| {' | '.join(cols)} |")
    lines.append(f"| {' | '.join(['---'] * len(cols))} |")

    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append(f"| {' | '.join(cells)} |")

    with open(path, "w") as fh:
        fh.write("# Cross-Experiment PPL Comparison\n\n")
        fh.write(
            "PPL measured on WikiText-103 test split.  "
            "`ratio_*` columns are normalised by the zero-shot k=1 baseline.\n\n"
        )
        fh.write("\n".join(lines))
        fh.write("\n")

    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

EXP_COLORS = {
    "zero_shot":    "#e15759",   # red
    "finetune":     "#4e79a7",   # blue
    "from_scratch": "#59a14f",   # green
}
EXP_MARKERS = {
    "zero_shot":    "o",
    "finetune":     "s",
    "from_scratch": "^",
}


def plot_ppl_by_k(df: pd.DataFrame, method: str, output_dir: str, dpi: int) -> None:
    """
    For a given method, plot PPL vs nominal_k with one line per experiment type.
    A dashed horizontal line shows the zero-shot k=1 baseline.
    """
    method_df = df[df["method"] == method].copy()
    if method_df.empty:
        return

    experiments = [e for e in ("zero_shot", "finetune", "from_scratch")
                   if e in method_df["experiment"].unique()]
    if not experiments:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    # Baseline reference
    baseline = method_df[
        (method_df["experiment"] == "zero_shot") & (method_df["nominal_k"] == 1)
    ]
    if not baseline.empty:
        bppl = baseline["ppl"].values[0]
        ax.axhline(bppl, color="grey", linestyle="--", linewidth=1.2,
                   label=f"Baseline k=1 ({bppl:.1f})", zorder=1)

    for exp in experiments:
        exp_df = method_df[method_df["experiment"] == exp].sort_values("nominal_k")
        k_vals = exp_df["nominal_k"].tolist()
        ppl_vals = exp_df["ppl"].tolist()
        if not k_vals:
            continue
        label = EXPERIMENT_LABEL.get(exp, exp)
        ax.plot(
            k_vals, ppl_vals,
            marker=EXP_MARKERS.get(exp, "o"),
            color=EXP_COLORS.get(exp, "#333"),
            linewidth=2,
            markersize=7,
            label=label,
            zorder=2,
        )
        for x, y in zip(k_vals, ppl_vals):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)

    ax.set_xlabel("Nominal k (window size)", fontsize=12)
    ax.set_ylabel("Perplexity (↓ better)", fontsize=12)
    ax.set_title(f"PPL vs k  —  method: {method}", fontsize=13)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"ppl_by_k_{method}.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_ppl_by_method(df: pd.DataFrame, k: int, output_dir: str, dpi: int) -> None:
    """
    For a given k value, plot PPL by method with grouped bars (one per experiment).
    """
    k_df = df[df["nominal_k"] == k].copy()
    if k_df.empty:
        return

    methods = sorted(k_df["method"].unique())
    experiments = [e for e in ("zero_shot", "finetune", "from_scratch")
                   if e in k_df["experiment"].unique()]
    if not experiments:
        return

    n_exp = len(experiments)
    bar_width = 0.7 / n_exp
    x = list(range(len(methods)))

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.8), 5))

    for i, exp in enumerate(experiments):
        exp_df = k_df[k_df["experiment"] == exp]
        ppl_vals = []
        for m in methods:
            row = exp_df[exp_df["method"] == m]
            ppl_vals.append(row["ppl"].values[0] if not row.empty else float("nan"))

        offsets = [xi + (i - n_exp / 2 + 0.5) * bar_width for xi in x]
        label = EXPERIMENT_LABEL.get(exp, exp)
        ax.bar(
            offsets, ppl_vals,
            width=bar_width,
            color=EXP_COLORS.get(exp, "#333"),
            label=label,
            alpha=0.85,
        )

    # Baseline reference
    baseline = df[(df["experiment"] == "zero_shot") & (df["nominal_k"] == 1)]
    if not baseline.empty:
        bppl = baseline["ppl"].mean()
        ax.axhline(bppl, color="grey", linestyle="--", linewidth=1.2,
                   label=f"Baseline k=1 ({bppl:.1f})")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=10)
    ax.set_xlabel("Averaging method", fontsize=12)
    ax.set_ylabel("Perplexity (↓ better)", fontsize=12)
    ax.set_title(f"PPL by method  —  k = {k}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"ppl_by_method_k{k}.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_ppl_recovery(df: pd.DataFrame, output_dir: str, dpi: int) -> None:
    """
    Scatter plot: x=zero_shot PPL, y=finetune PPL or from_scratch PPL.
    Shows how much each config improves from zero-shot baseline.
    """
    # Need both zero_shot and at least one other experiment
    zs = df[df["experiment"] == "zero_shot"][["config_name", "ppl"]].rename(
        columns={"ppl": "ppl_zero_shot"}
    )
    ft = df[df["experiment"] == "finetune"][["config_name", "ppl"]].rename(
        columns={"ppl": "ppl_finetune"}
    )
    fs = df[df["experiment"] == "from_scratch"][["config_name", "ppl"]].rename(
        columns={"ppl": "ppl_scratch"}
    )

    merged = zs.copy()
    if not ft.empty:
        merged = merged.merge(ft, on="config_name", how="outer")
    if not fs.empty:
        merged = merged.merge(fs, on="config_name", how="outer")

    if merged.empty or "ppl_zero_shot" not in merged.columns:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    for col, color, marker, label in [
        ("ppl_finetune", EXP_COLORS["finetune"], "s", "Finetune"),
        ("ppl_scratch", EXP_COLORS["from_scratch"], "^", "From scratch"),
    ]:
        sub = merged.dropna(subset=["ppl_zero_shot", col]) if col in merged.columns else pd.DataFrame()
        if not sub.empty:
            ax.scatter(
                sub["ppl_zero_shot"], sub[col],
                color=color, marker=marker, s=60, alpha=0.8, label=label, zorder=2,
            )

    # y=x diagonal (no improvement)
    if not merged["ppl_zero_shot"].dropna().empty:
        lims = [merged["ppl_zero_shot"].min() * 0.9, merged["ppl_zero_shot"].max() * 1.1]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="No change (y=x)", zorder=1)
        ax.set_xlim(lims)

    ax.set_xlabel("Zero-shot PPL", fontsize=12)
    ax.set_ylabel("Trained PPL", fontsize=12)
    ax.set_title("PPL recovery: trained vs zero-shot", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "ppl_recovery_scatter.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    df = load_all(args.results_dir)
    print(f"Total rows loaded: {len(df)}")
    print(f"Experiments present: {df['experiment'].unique().tolist()}")
    print(f"Methods present:     {df['method'].unique().tolist()}")
    print(f"k values present:    {sorted(df['nominal_k'].unique().tolist())}")

    # Build and write wide-format comparison table
    comparison = build_comparison_table(df)
    md_path = os.path.join(args.output_dir, "comparison_table.md")
    write_markdown_table(comparison, md_path)
    csv_path = md_path.replace(".md", ".csv")
    comparison.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    # Print to terminal
    print("\nComparison table preview:")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(comparison.to_string(index=False))

    # Per-method PPL vs k plots
    for method in sorted(df["method"].unique()):
        plot_ppl_by_k(df, method, args.output_dir, args.dpi)

    # Per-k PPL by method plots
    for k in sorted(df["nominal_k"].unique()):
        if k == 1:
            continue   # baseline only — no grouped bar makes sense
        plot_ppl_by_method(df, k, args.output_dir, args.dpi)

    # Recovery scatter
    plot_ppl_recovery(df, args.output_dir, args.dpi)

    print(f"\nAll comparison outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
