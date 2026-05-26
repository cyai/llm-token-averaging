"""
Plot loss vs tokens-seen and loss vs FLOPs for the 50M model comparison.

FLOPs are computed from scratch using:
    FLOPs/sequence = N_layers * L * (24*d_model^2 + 4*L*d_model)
where L = context_len (the sequence length the transformer actually processes).
The CSV's cumulative_flops column is ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chinchilla.model_configs import MODEL_CONFIGS

# Chinchilla loss constants (Hoffmann et al. 2022)
_A, _ALPHA = 406.4, 0.3392
_B, _BETA  = 410.7, 0.2849
_E         = 1.6934


def chinchilla_loss(N: float, D: float) -> float:
    return _A / N ** _ALPHA + _B / D ** _BETA + _E


def chinchilla_optimal(cfg) -> tuple[float, float, float]:
    """Return (tokens_seen_opt, flops_opt, loss_opt) at the Chinchilla D*=20N point."""
    N = cfg.n_params_approx
    D_opt = 20.0 * N                        # transformer tokens
    tokens_seen_opt = cfg.averaging_k * D_opt  # original tokens consumed
    flops_divisor = 6.0 / cfg.averaging_k
    flops_opt = flops_divisor * N * D_opt
    loss_opt = chinchilla_loss(N, D_opt)
    return tokens_seen_opt, flops_opt, loss_opt

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"

TARGET_MODELS = [
    # "model1_8m",
    # "avg_8m_k2",
    # "model2_8m_ctx2n",
    # "avg_8m_k4",
    # "model2_8m_ctx4n",
    "model1_50m",
    "model2_50m_ctx2n",
    "avg_50m_k2",
]

# Color scheme: warm tones for standard models, cool tones for averaging models
COLOR_OVERRIDE = {
    "model1_50m":       "#4e9de0",   # steel blue   – standard baseline (n=512)
    "model2_50m_ctx2n": "#f0a500",   # amber        – standard 2n context
    "avg_50m_k2":       "#3fb950",   # green        – 2× averaging
}


def flops_per_sequence(cfg) -> float:
    """FLOPs for one forward pass through one sequence of length context_len."""
    L = cfg.context_len
    d = cfg.d_model
    return cfg.n_layers * L * (24 * d * d + 4 * L * d)


def compute_cumulative_flops(df: pd.DataFrame, cfg) -> np.ndarray:
    """
    Derive batch_size from the CSV (tokens_seen / step / effective_tokens_per_seq),
    then compute cumulative FLOPs using the analytic formula.

    For standard models:   effective_tokens_per_seq = context_len
    For averaging models:  effective_tokens_per_seq = k * context_len
      (each transformer call consumes k original tokens per position)
    """
    tokens_per_step = df["tokens_seen"].values / df["step"].values
    effective_tokens_per_seq = cfg.averaging_k * cfg.context_len
    batch_size = tokens_per_step / effective_tokens_per_seq
    # batch_size should be ~constant; use per-row value for accuracy
    fps = flops_per_sequence(cfg)
    return df["step"].values * batch_size * fps


def ema_smooth(values: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
    return out


def _style_ax(ax, xlabel: str, title: str, xlog: bool = False) -> None:
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#c9d1d9", labelsize=10)
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, which="both", color="#21262d", linewidth=0.7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    if xlog:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:.1e}")
        )


def make_plots() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load available logs
    logs: dict[str, pd.DataFrame] = {}
    for name in TARGET_MODELS:
        csv_path = RESULTS_DIR / name / "loss_log.csv"
        if csv_path.exists():
            logs[name] = pd.read_csv(csv_path)
            print(f"  loaded {name}: {len(logs[name])} steps")
        else:
            print(f"  [skip] {name} — no loss_log.csv")

    if not logs:
        raise RuntimeError("No CSVs found in " + str(RESULTS_DIR))

    # ------------------------------------------------------------------ #
    #  Figure 1: loss vs tokens seen                                       #
    # ------------------------------------------------------------------ #
    fig1, ax1 = plt.subplots(figsize=(11, 6), facecolor="#0d1117")
    _style_ax(ax1, "Tokens Seen", "Loss vs Tokens Seen  (50M model comparison)")

    for name, df in logs.items():
        cfg = MODEL_CONFIGS[name]
        color = COLOR_OVERRIDE.get(name, cfg.color)
        tokens = df["tokens_seen"].values
        train_s = ema_smooth(df["train_loss"].values)
        eval_v = df["eval_loss"].values

        mask = ~np.isnan(eval_v)
        if mask.any():
            ax1.plot(tokens[mask], eval_v[mask], color=color, linewidth=2.2,
                     label=cfg.label)

    # Chinchilla-optimal markers
    for name in logs:
        cfg = MODEL_CONFIGS[name]
        color = COLOR_OVERRIDE.get(name, cfg.color)
        tok_opt, _, loss_opt = chinchilla_optimal(cfg)
        ax1.scatter([tok_opt], [loss_opt], color=color, s=120, zorder=6,
                    edgecolors="white", linewidths=1.2, marker="*")

    _add_legend(ax1)
    fig1.tight_layout()
    out1 = PLOTS_DIR / "50m_loss_vs_tokens.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig1)
    print(f"  → {out1}")

    # ------------------------------------------------------------------ #
    #  Figure 2: loss vs FLOPs (computed analytically)                    #
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots(figsize=(11, 6), facecolor="#0d1117")
    _style_ax(ax2, "Cumulative FLOPs  (N_layers·L·(24d²+4Ld) formula)",
              "Loss vs FLOPs  (50M model comparison)", xlog=True)

    for name, df in logs.items():
        cfg = MODEL_CONFIGS[name]
        color = COLOR_OVERRIDE.get(name, cfg.color)
        cum_flops = compute_cumulative_flops(df, cfg)
        train_s = ema_smooth(df["train_loss"].values)
        eval_v = df["eval_loss"].values

        fps = flops_per_sequence(cfg)
        tok_per_step = df["tokens_seen"].values[0] / df["step"].values[0]
        eff = cfg.averaging_k * cfg.context_len
        bs = tok_per_step / eff
        print(f"  {name}: L={cfg.context_len}, d={cfg.d_model}, "
              f"k={cfg.averaging_k}, batch≈{bs:.0f}, "
              f"FLOPs/seq={fps:.3e}")

        mask = ~np.isnan(eval_v)
        if mask.any():
            ax2.plot(cum_flops[mask], eval_v[mask], color=color, linewidth=2.2,
                     label=cfg.label)

    # Chinchilla-optimal markers
    for name in logs:
        cfg = MODEL_CONFIGS[name]
        color = COLOR_OVERRIDE.get(name, cfg.color)
        _, flops_opt, loss_opt = chinchilla_optimal(cfg)
        ax2.scatter([flops_opt], [loss_opt], color=color, s=120, zorder=6,
                    edgecolors="white", linewidths=1.2, marker="*")

    _add_legend(ax2)
    fig2.tight_layout()
    out2 = PLOTS_DIR / "50m_loss_vs_flops.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig2)
    print(f"  → {out2}")


def _add_legend(ax) -> None:
    import matplotlib.lines as mlines
    star = mlines.Line2D([], [], color="white", marker="*", linestyle="None",
                         markersize=9, markeredgecolor="white",
                         label="Chinchilla optimal (D*=20N)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + [star],
        labels=labels + ["Chinchilla optimal (D*=20N)"],
        loc="upper right",
        fontsize=9,
        framealpha=0.4,
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="#c9d1d9",
    )


if __name__ == "__main__":
    print("Generating 50M model comparison plots…")
    make_plots()
    print("Done.")
