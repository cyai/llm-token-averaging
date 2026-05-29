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
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chinchilla.model_configs import MODEL_CONFIGS

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"

TARGET_MODELS = [
    # standard baselines
    "model1_50m",
    "model2_50m_ctx2n",
    # averaging models
    "avg_50m_k2_ctx512",
    "avg_50m_k2",
    "avg_50m_mixed_k2k4",
    "avg_50m_k4",
    "avg_50m_k8",
    "avg_50m_k2_wide",
    "avg_50m_k16",
    "avg_50m_k32",
    "avg_50m_k64",
    # phased (token superposition) models
    "avg_50m_k2_phased",
    "avg_50m_k4_phased",
    "avg_50m_k8_phased",
]

COLOR_OVERRIDE = {
    # standard models – warm/neutral
    "model1_50m": "#4e9de0",  # steel blue   – standard (n=1024)
    "model2_50m_ctx2n": "#f8a500",  # amber        – standard 2n
    # averaging models – cool greens → cyan, ordered by effective context
    "avg_50m_k2_ctx512": "#ff7675",  # salmon       – k=2, ctx=512 (eff=1024)
    "avg_50m_k2": "#3fb950",  # green        – k=2, ctx=1024 (eff=2048)
    "avg_50m_mixed_k2k4": "#9b59b6",  # purple       – mixed k=2/4 (eff=3072)
    "avg_50m_k4": "#f1c40f",  # yellow       – k=4 (eff=4096)
    "avg_50m_k8": "#00c8c8",  # cyan         – k=8 (eff=8192)
    "avg_50m_k2_wide": "#e74c3c",  # red          – k=2, wide (eff=2048)
    "avg_50m_k16": "#808080",  # gray         – k=16 (eff=16384)
    "avg_50m_k32": "#58a6ff",  # blue         – k=32 (eff=32768)
    "avg_50m_k64": "#765341",  # brown       – k=64 (eff=65536)
    # phased (token superposition) models
    "avg_50m_k2_phased": "#2ecc71",  # emerald     – k=2 phased
    "avg_50m_k4_phased": "#e74c3c",  # crimson     – k=4 phased
    "avg_50m_k8_phased": "#1abc9c",  # teal        – k=8 phased
}

# Chinchilla loss constants (Hoffmann et al. 2022)
_A, _ALPHA = 406.4, 0.3392
_B, _BETA = 410.7, 0.2849
_E = 1.6934


def chinchilla_loss(N: float, D: float) -> float:
    return _A / N**_ALPHA + _B / D**_BETA + _E


def chinchilla_optimal(cfg) -> tuple[float, float, float]:
    """Return (tokens_seen_opt, flops_opt, loss_opt) at the Chinchilla D*=20N point."""
    N = cfg.n_params_approx
    D_opt = 20.0 * N  # transformer tokens
    tokens_seen_opt = cfg.averaging_k * D_opt  # original tokens consumed
    flops_divisor = 6.0 / cfg.averaging_k
    flops_opt = flops_divisor * N * D_opt
    loss_opt = chinchilla_loss(N, D_opt)
    return tokens_seen_opt, flops_opt, loss_opt


def flops_per_sequence(cfg) -> float:
    """FLOPs for one forward pass through one sequence of length context_len."""
    L = cfg.context_len
    d = cfg.d_model
    return cfg.n_layers * L * (24 * d * d + 4 * L * d)


def compute_cumulative_flops(df: pd.DataFrame, cfg) -> np.ndarray:
    """
    Derive batch_size from the CSV, then compute cumulative FLOPs analytically.

    For standard models:   effective_tokens_per_seq = context_len
    For averaging models:  effective_tokens_per_seq = k * context_len
    """
    tokens_per_step = df["tokens_seen"].values / df["step"].values
    effective_tokens_per_seq = cfg.averaging_k * cfg.context_len
    batch_size = tokens_per_step / effective_tokens_per_seq
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
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1e}"))
    else:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1e}"))


def _add_legend(ax) -> None:
    star = mlines.Line2D(
        [],
        [],
        color="white",
        marker="*",
        linestyle="None",
        markersize=9,
        markeredgecolor="white",
        label="Chinchilla optimal (D*=20N)",
    )
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


def _plot_tokens(logs: dict, xlog: bool, out_path: Path) -> None:
    scale = "log" if xlog else "linear"
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#0d1117")
    _style_ax(
        ax, "Tokens Seen", f"Loss vs Tokens Seen  (50M)  [{scale} scale]", xlog=xlog
    )

    for name, df in logs.items():
        cfg = MODEL_CONFIGS[name]
        color = COLOR_OVERRIDE.get(name, cfg.color)
        tokens = df["tokens_seen"].values
        eval_v = df["eval_loss"].values
        mask = ~np.isnan(eval_v)
        if mask.any():
            ax.plot(
                tokens[mask], eval_v[mask], color=color, linewidth=2.2, label=cfg.label
            )

    for name in logs:
        cfg = MODEL_CONFIGS[name]
        color = COLOR_OVERRIDE.get(name, cfg.color)
        tok_opt, _, loss_opt = chinchilla_optimal(cfg)
        ax.scatter(
            [tok_opt],
            [loss_opt],
            color=color,
            s=120,
            zorder=6,
            edgecolors="white",
            linewidths=1.2,
            marker="*",
        )

    _add_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  → {out_path}")


def _plot_flops(logs: dict, xlog: bool, out_path: Path) -> None:
    scale = "log" if xlog else "linear"
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#0d1117")
    _style_ax(
        ax,
        "Cumulative FLOPs  (N_layers·L·(24d²+4Ld) formula)",
        f"Loss vs FLOPs  (50M)  [{scale} scale]",
        xlog=xlog,
    )

    for name, df in logs.items():
        cfg = MODEL_CONFIGS[name]
        color = COLOR_OVERRIDE.get(name, cfg.color)
        cum_flops = compute_cumulative_flops(df, cfg)
        eval_v = df["eval_loss"].values
        mask = ~np.isnan(eval_v)
        if mask.any():
            ax.plot(
                cum_flops[mask],
                eval_v[mask],
                color=color,
                linewidth=2.2,
                label=cfg.label,
            )

    for name in logs:
        cfg = MODEL_CONFIGS[name]
        color = COLOR_OVERRIDE.get(name, cfg.color)
        _, flops_opt, loss_opt = chinchilla_optimal(cfg)
        ax.scatter(
            [flops_opt],
            [loss_opt],
            color=color,
            s=120,
            zorder=6,
            edgecolors="white",
            linewidths=1.2,
            marker="*",
        )

    _add_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  → {out_path}")


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

    _plot_tokens(logs, xlog=False, out_path=PLOTS_DIR / "50m_loss_vs_tokens_linear.png")
    _plot_tokens(logs, xlog=True, out_path=PLOTS_DIR / "50m_loss_vs_tokens_log.png")
    _plot_flops(logs, xlog=False, out_path=PLOTS_DIR / "50m_loss_vs_flops_linear.png")
    _plot_flops(logs, xlog=True, out_path=PLOTS_DIR / "50m_loss_vs_flops_log.png")


if __name__ == "__main__":
    print("Generating 50M model comparison plots…")
    make_plots()
    print("Done.")
