"""
Visualization for the Chinchilla FLOPs comparison experiment.

Reads loss_log.csv files produced by train.py for all three models and generates:
  1. loss_vs_flops.png          — matplotlib, dark background
  2. loss_vs_flops_interactive.html — Plotly, dark theme

Both plots overlay:
  - Solid empirical training curves (EMA-smoothed train loss)
  - Dashed empirical eval curves at checkpoint steps
  - Faint dotted Chinchilla theoretical L(N, D(C)) curves for each model size
  - Marker dots at the Chinchilla-optimal compute point per model

Chinchilla constants from Hoffmann et al. (2022), as fitted in chinchilla_analysis.ipynb:
  L(N, D) = A / N^α  +  B / D^β  +  E
  A = 406.4,  α = 0.3392
  B = 410.7,  β = 0.2849
  E = 1.6934

Usage
-----
    python experiments/chinchilla/visualize.py \
        --results_dir experiments/chinchilla/results \
        --plots_dir   experiments/chinchilla/plots
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.chinchilla.model_configs import MODEL_CONFIGS, ModelConfig

# ---------------------------------------------------------------------------
# Chinchilla constants  (Hoffmann et al. 2022 / chinchilla_analysis.ipynb)
# ---------------------------------------------------------------------------

A     = 406.4
ALPHA = 0.3392
B     = 410.7
BETA  = 0.2849
E     = 1.6934


def chinchilla_loss(N: float, D: float) -> float:
    """Predicted cross-entropy loss for N parameters trained on D tokens."""
    return A / N ** ALPHA + B / D ** BETA + E


def chinchilla_optimal_D(N: float, budget_C: float) -> float:
    """
    Optimal token count D* for a given model size N and compute budget C.
    From Chinchilla optimal allocation: D* = (C / (6·N))  (rule-of-thumb).
    """
    return budget_C / (6.0 * N)


def theoretical_curve(
    N: float,
    flops_range: np.ndarray,
    flops_per_token_divisor: float = 6.0,
) -> np.ndarray:
    """
    Theoretical Chinchilla loss at each compute budget in `flops_range`.

    For a standard model: D = C / (6·N)
    For the averaging model (k=2): transformer sees D/2, but we use
    flops_per_token = 6·N/2, so D_effective = C / (6·N/2) = C / (3·N).
    The token count seen by the model is also D_effective, which is what
    matters for the language modeling difficulty.
    """
    losses = np.zeros_like(flops_range, dtype=float)
    for i, C in enumerate(flops_range):
        D = C / (flops_per_token_divisor * N)
        losses[i] = chinchilla_loss(N, max(D, 1e6))
    return losses


def _optimal_point(N: float, flops_per_token_divisor: float = 6.0) -> Tuple[float, float]:
    """Return (C_opt, L_opt) at the Chinchilla-optimal compute budget."""
    # Chinchilla optimal: D* = 20·N (rule of thumb); C* = 6·N·D*
    D_opt = 20.0 * N
    C_opt = flops_per_token_divisor * N * D_opt
    L_opt = chinchilla_loss(N, D_opt)
    return C_opt, L_opt


def _ema_smooth(values: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    smoothed = np.zeros_like(values)
    if len(values) == 0:
        return smoothed
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * values[i]
    return smoothed


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_logs(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all available loss_log.csv files from results_dir/<model_name>/.
    Returns a dict mapping model name → DataFrame.
    """
    logs: Dict[str, pd.DataFrame] = {}
    for name in MODEL_CONFIGS:
        csv_path = results_dir / name / "loss_log.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["cumulative_flops"] = df["cumulative_flops"].astype(float)
            logs[name] = df
            print(f"Loaded {name}: {len(df)} rows")
        else:
            print(f"[warn] {csv_path} not found — skipping {name}")
    return logs


# ---------------------------------------------------------------------------
# Matplotlib static plot
# ---------------------------------------------------------------------------

def plot_matplotlib(
    logs: Dict[str, pd.DataFrame],
    plots_dir: Path,
    smooth_alpha: float = 0.9,
) -> Path:
    """Generate loss_vs_flops.png with a dark matplotlib style."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    # Build a shared FLOPs axis for theoretical curves
    all_flops: List[float] = []
    for df in logs.values():
        all_flops.extend(df["cumulative_flops"].tolist())

    if not all_flops:
        print("[warn] No empirical data found; plotting theoretical curves only.")
        flops_min = 1e15
        flops_max = 4e19
    else:
        flops_min = min(all_flops) * 0.5
        flops_max = max(all_flops) * 2.0

    flops_axis = np.geomspace(flops_min, flops_max, 400)

    # --- theoretical Chinchilla curves ---
    for name, cfg in MODEL_CONFIGS.items():
        N = cfg.n_params_approx
        divisor = 6.0 / cfg.averaging_k   # 6 for standard, 3 for k=2
        theory = theoretical_curve(N, flops_axis, flops_per_token_divisor=divisor)
        ax.plot(
            flops_axis, theory,
            linestyle=":",
            color=cfg.color,
            alpha=0.35,
            linewidth=1.5,
            label=f"Chinchilla theory ({cfg.label})",
        )

        # Chinchilla-optimal point
        C_opt, L_opt = _optimal_point(N, divisor)
        if flops_min <= C_opt <= flops_max * 10:
            ax.scatter(
                [C_opt], [L_opt],
                color=cfg.color,
                s=100,
                zorder=5,
                edgecolors="white",
                linewidths=1.0,
            )

    # --- empirical curves ---
    for name, df in logs.items():
        cfg = MODEL_CONFIGS[name]
        flops = df["cumulative_flops"].values
        train_loss = df["train_loss"].values
        eval_loss = df["eval_loss"].values

        # Smooth train loss for visibility
        train_smooth = _ema_smooth(train_loss, smooth_alpha)

        ax.plot(
            flops, train_smooth,
            color=cfg.color,
            linewidth=2.0,
            label=f"{cfg.label} (train)",
        )

        # Eval at every logged point
        eval_mask = ~np.isnan(eval_loss)
        if eval_mask.any():
            ax.plot(
                flops[eval_mask], eval_loss[eval_mask],
                color=cfg.color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.75,
                label=f"{cfg.label} (eval)",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Cumulative FLOPs", fontsize=13)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=13)
    ax.set_title(
        "Token Averaging vs. Standard OLM: Loss vs. FLOPs\n"
        "(dashed = eval, dotted = Chinchilla theory, ● = Chinchilla-optimal point)",
        fontsize=14,
        pad=14,
    )

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0e}".replace("e+0", "e").replace("e+", "e")
    ))
    ax.grid(True, which="both", color="#21262d", linewidth=0.7)

    legend = ax.legend(
        loc="upper right",
        fontsize=10,
        framealpha=0.3,
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="#c9d1d9",
    )

    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "loss_vs_flops.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"Saved static plot → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Plotly interactive plot
# ---------------------------------------------------------------------------

def plot_plotly(
    logs: Dict[str, pd.DataFrame],
    plots_dir: Path,
    smooth_alpha: float = 0.9,
) -> Path:
    """Generate loss_vs_flops_interactive.html with Plotly dark theme."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Build FLOPs range
    all_flops: List[float] = []
    for df in logs.values():
        all_flops.extend(df["cumulative_flops"].tolist())

    flops_min = min(all_flops) * 0.5 if all_flops else 1e15
    flops_max = max(all_flops) * 2.0 if all_flops else 4e19
    flops_axis = np.geomspace(flops_min, flops_max, 400)

    # --- theoretical curves ---
    for name, cfg in MODEL_CONFIGS.items():
        N = cfg.n_params_approx
        divisor = 6.0 / cfg.averaging_k
        theory = theoretical_curve(N, flops_axis, flops_per_token_divisor=divisor)

        fig.add_trace(go.Scatter(
            x=flops_axis.tolist(),
            y=theory.tolist(),
            mode="lines",
            name=f"Chinchilla theory ({cfg.label})",
            line=dict(color=cfg.color, width=1.5, dash="dot"),
            opacity=0.4,
            hovertemplate="C = %{x:.3e}<br>L = %{y:.4f}<extra></extra>",
        ))

        # Optimal point
        C_opt, L_opt = _optimal_point(N, divisor)
        fig.add_trace(go.Scatter(
            x=[C_opt],
            y=[L_opt],
            mode="markers",
            name=f"Chinchilla-opt ({cfg.label})",
            marker=dict(
                color=cfg.color,
                size=12,
                line=dict(color="white", width=1.5),
                symbol="circle",
            ),
            hovertemplate=f"Chinchilla-opt {cfg.label}<br>C = %{{x:.3e}}<br>L = %{{y:.4f}}<extra></extra>",
        ))

    # --- empirical curves ---
    for name, df in logs.items():
        cfg = MODEL_CONFIGS[name]
        flops = df["cumulative_flops"].values
        train_loss = _ema_smooth(df["train_loss"].values, smooth_alpha)
        eval_loss = df["eval_loss"].values

        fig.add_trace(go.Scatter(
            x=flops.tolist(),
            y=train_loss.tolist(),
            mode="lines",
            name=f"{cfg.label} train",
            line=dict(color=cfg.color, width=2.5),
            hovertemplate="C = %{x:.3e}<br>train loss = %{y:.4f}<extra></extra>",
        ))

        eval_mask = ~np.isnan(eval_loss)
        if eval_mask.any():
            fig.add_trace(go.Scatter(
                x=flops[eval_mask].tolist(),
                y=eval_loss[eval_mask].tolist(),
                mode="lines+markers",
                name=f"{cfg.label} eval",
                line=dict(color=cfg.color, width=1.5, dash="dash"),
                marker=dict(size=5),
                hovertemplate="C = %{x:.3e}<br>eval loss = %{y:.4f}<extra></extra>",
            ))

    # --- Chinchilla irreducible floor annotation ---
    fig.add_hline(
        y=E,
        line=dict(color="#8b949e", width=1, dash="longdash"),
        annotation_text=f"irreducible floor  E = {E}",
        annotation_position="bottom right",
        annotation_font=dict(color="#8b949e", size=11),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        title=dict(
            text=(
                "Token Averaging vs. Standard OLM: Loss vs. FLOPs<br>"
                "<sub>Solid = train (EMA-smoothed), Dashed = eval, "
                "Dotted = Chinchilla theory, ● = Chinchilla-optimal</sub>"
            ),
            font=dict(size=18, color="#e6edf3"),
        ),
        xaxis=dict(
            title=dict(text="Cumulative FLOPs", font=dict(color="#c9d1d9")),
            type="log",
            gridcolor="#21262d",
            tickfont=dict(color="#c9d1d9"),
        ),
        yaxis=dict(
            title=dict(text="Cross-Entropy Loss", font=dict(color="#c9d1d9")),
            gridcolor="#21262d",
            tickfont=dict(color="#c9d1d9"),
        ),
        legend=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(color="#c9d1d9", size=11),
        ),
        hovermode="x unified",
        width=1100,
        height=650,
    )

    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "loss_vs_flops_interactive.html"
    fig.write_html(str(out_path))
    print(f"Saved interactive plot → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_plots(
    results_dir: Path,
    plots_dir: Path,
    smooth_alpha: float = 0.9,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Load available training logs and generate both plot files.
    Returns (static_path, interactive_path), either may be None on failure.
    """
    logs = load_logs(results_dir)
    if not logs:
        print("[warn] No loss_log.csv files found. Plots will show theory only.")

    static_path = interactive_path = None

    try:
        static_path = plot_matplotlib(logs, plots_dir, smooth_alpha)
    except Exception as exc:
        print(f"[error] matplotlib plot failed: {exc}")

    try:
        interactive_path = plot_plotly(logs, plots_dir, smooth_alpha)
    except Exception as exc:
        print(f"[error] Plotly plot failed: {exc}")

    return static_path, interactive_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize Chinchilla FLOPs comparison.")
    p.add_argument(
        "--results_dir",
        type=str,
        default=str(_ROOT / "experiments" / "chinchilla" / "results"),
        help="Directory containing model subdirectories with loss_log.csv files.",
    )
    p.add_argument(
        "--plots_dir",
        type=str,
        default=str(_ROOT / "experiments" / "chinchilla" / "plots"),
        help="Output directory for generated plots.",
    )
    p.add_argument(
        "--smooth_alpha",
        type=float,
        default=0.9,
        help="EMA smoothing coefficient for train loss curves.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    generate_plots(
        results_dir=Path(args.results_dir),
        plots_dir=Path(args.plots_dir),
        smooth_alpha=args.smooth_alpha,
    )


if __name__ == "__main__":
    main()
