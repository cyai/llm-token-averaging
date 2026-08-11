"""
Plot loss vs tokens-seen and loss vs FLOPs for token-averaging experiments.

FLOPs are recomputed from tokens_seen (CSV's cumulative_flops column is ignored):
    Training FLOPs/sequence = N_layers * L * 3 * (23*d^2 + 4*L*d)
where L = context_len / averaging_k (transformer sequence length).
The 3x accounts for forward + backward pass.
23d^2 = attention projections (8d^2) + SwiGLU FFN (15d^2).

Usage examples:

  # Plot specific models by name:
  python plot_50m_models.py model1_50m avg_50m_k2 avg_50m_k4

  # Use a preset group:
  python plot_50m_models.py --preset 125m
  python plot_50m_models.py --preset 250m
  python plot_50m_models.py --preset ablations-k2
  python plot_50m_models.py --preset ablations-k4
  python plot_50m_models.py --preset ctx

  # Mix presets and individual models:
  python plot_50m_models.py --preset 125m model1_250m

  # Control output:
  python plot_50m_models.py --preset 250m --title "250M scaling" --prefix 250m_scaling
  python plot_50m_models.py --preset 250m --no-chinchilla
  python plot_50m_models.py --preset 250m --scale log         # log only
  python plot_50m_models.py --preset 250m --scale linear      # linear only
  python plot_50m_models.py --preset 250m --scale both        # both (default)

  # Override which CSV to load (default: loss_log.csv):
  python plot_50m_models.py model1_50m --csv loss_log_1x_ctx.csv

  # Config-B context-scaling runs (special: uses per-run CSV/seq_len overrides):
  python plot_50m_models.py --preset ctx

  # List all available model configs and all CSVs on disk:
  python plot_50m_models.py --list
"""

from __future__ import annotations

import argparse
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


def _drop_isolated_spikes(
    df: pd.DataFrame, col: str = "eval_loss", factor: float = 1.4
) -> pd.DataFrame:
    """Drop single-row loss spikes that recover by the very next logged step.

    A row is dropped only if its `col` value exceeds BOTH neighbors by
    `factor`, so it never touches monotonic trends (e.g. the initial
    warmup ramp, where neighbors are also elevated). This is a display-only
    filter — it does not modify the underlying CSVs — for the occasional
    single-step training transient (e.g. a bad batch) that fully recovers
    within one log interval and would otherwise dominate a log-scale axis.
    """
    if col not in df.columns or len(df) < 3:
        return df
    v = df[col].to_numpy()
    left = v[:-2]
    mid = v[1:-1]
    right = v[2:]
    is_spike = (mid > factor * left) & (mid > factor * right)
    keep = np.ones(len(df), dtype=bool)
    keep[1:-1] = ~is_spike
    return df[keep].reset_index(drop=True)


RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"

# ── Preset groups ──────────────────────────────────────────────────────────
PRESETS: dict[str, list[str]] = {
    "125m": ["model1_125m", "avg_125m_k2"],
    "250m": ["model1_250m", "avg_250m_k2"],
    "500m": ["model1_500m", "avg_500m_k2"],
    "1b": ["model1_1b", "avg_1b_k2"],
    "50m": ["model1_50m", "avg_50m_k2", "avg_50m_k4"],
    "ablations-k2": [
        "model1_50m",
        "avg_50m_k2",
        "avg_50m_k2_mean_matched",
        "avg_50m_k2_learnable",
        "avg_50m_k2_learnable_pos",
        "avg_50m_k2_ov4s2",
        "avg_50m_k2_wexp",
        "avg_50m_k2_word",
    ],
    "ablations-k4": [
        "model1_50m",
        "avg_50m_k4",
        "avg_50m_k4_mean_matched",
        "avg_50m_k4_learnable",
        "avg_50m_k4_learnable_pos",
        "avg_50m_k4_wexp",
    ],
    "ablations-k8": [
        "model1_50m",
        "avg_50m_k8",
        "avg_50m_k8_mean_matched",
        "avg_50m_k8_learnable",
        "avg_50m_k8_learnable_pos",
        "avg_50m_k8_wexp",
    ],
    "update-control": [
        "model1_50m",
        "model1_50m_update_ctrl",
        "avg_50m_k2_mean_matched",
        "avg_50m_k2_learnable_pos",
    ],
    # Config-B is handled specially (per-run CSV and raw_seq_len overrides)
    "ctx": [],
}

# ── Alternate-directory runs ──────────────────────────────────────────────
# Alias name -> (base config name, results root, label suffix).
# Used for reruns of an existing config whose logs live outside results/
# (e.g. the matched-protocol mean-pooling controls in results_matched_mean/).
ALT_RESULTS: dict[str, tuple[str, Path, str]] = {
    "avg_50m_k2_mean_matched": (
        "avg_50m_k2",
        Path(__file__).parent / "results_matched_mean",
        " [mean, matched protocol]",
    ),
    "avg_50m_k4_mean_matched": (
        "avg_50m_k4",
        Path(__file__).parent / "results_matched_mean",
        " [mean, matched protocol]",
    ),
    "avg_50m_k8_mean_matched": (
        "avg_50m_k8",
        Path(__file__).parent / "results_matched_mean",
        " [mean, matched protocol]",
    ),
    # Update-count control: k=1 with batch 8 -> 122k steps for 1B tokens,
    # matching the matched k=2 run's optimizer-update count at iso-FLOPs
    # with the standard baseline.
    "model1_50m_update_ctrl": (
        "model1_50m",
        Path(__file__).parent / "results_update_control",
        " [batch 8, 2x steps: update-count control]",
    ),
}


def _resolve_run(name: str):
    """Return (cfg, csv_dir, label) for a config name or ALT_RESULTS alias."""
    if name in ALT_RESULTS:
        base, root, suffix = ALT_RESULTS[name]
        cfg = MODEL_CONFIGS[base]
        return cfg, root / base, cfg.label + suffix
    cfg = MODEL_CONFIGS[name]
    return cfg, RESULTS_DIR / name, cfg.label


# Config-B context-scaling runs.
# Each entry: (config_name, csv_filename, raw_seq_len, label, color)
CTX_RUNS = [
    (
        "model1_50m",
        "loss_log_1x_ctx.csv",
        1024,
        "~50M k=1  (seq 1024, L=1024)",
        "#4e9de0",
    ),
    (
        "avg_50m_k2",
        "loss_log_2x_ctx.csv",
        2048,
        "~50M k=2  (seq 2048, L=1024)",
        "#3fb950",
    ),
    (
        "avg_50m_k4",
        "loss_log_4x_ctx.csv",
        4096,
        "~50M k=4  (seq 4096, L=1024)",
        "#f1c40f",
    ),
    (
        "model2_50m_ctx2n",
        "loss_log.csv",
        2048,
        "~50M k=1  (seq 2048, L=2048, full attn)",
        "#e74c3c",
    ),
]

# ── Per-model colors ──────────────────────────────────────────────────────
COLORS: dict[str, str] = {
    "model1_50m": "#7c3aed",
    "model2_50m_ctx2n": "#4e9de0",
    "avg_50m_k2_ctx512": "#ff7675",
    "avg_50m_k2": "#3fb950",
    "avg_50m_mixed_k2k4": "#9b59b6",
    "avg_50m_k4": "#f1c40f",
    "avg_50m_k8": "#00c8c8",
    "avg_50m_k2_wide": "#e74c3c",
    "avg_50m_k16": "#808080",
    "avg_50m_k32": "#58a6ff",
    "avg_50m_k64": "#765341",
    "avg_50m_k2_phased": "#2ecc71",
    "avg_50m_k4_phased": "#e74c3c",
    "avg_50m_k8_phased": "#1abc9c",
    "avg_50m_k4_phased_30": "#1abc9c",
    "avg_50m_k4_tied": "#f1c40f",
    "model1_50m_tied": "#4e9de0",
    "avg_50m_k2_isoflop": "#3fb950",
    "avg_50m_k4_isoflop": "#f1c40f",
    "model1_125m": "#4e9de0",
    "avg_125m_k2": "#3fb950",
    "model1_250m": "#4e9de0",
    "avg_250m_k2": "#3fb950",
    "avg_50m_k2_learnable": "#e67e22",
    "avg_50m_k2_learnable_pos": "#2980b9",
    "avg_50m_k2_ov4s2": "#16a085",
    "avg_50m_k2_wexp": "#8e44ad",
    "avg_50m_k2_word": "#c0392b",
    "avg_50m_k4_learnable": "#d35400",
    "avg_50m_k4_learnable_pos": "#2471a3",
    "avg_50m_k4_wexp": "#8e44ad",
    "avg_50m_k2_mean_matched": "#e84393",
    "avg_50m_k4_mean_matched": "#fd79a8",
    "avg_50m_k8": "#00c8c8",
    "avg_50m_k8_mean_matched": "#17a589",
    "avg_50m_k8_learnable": "#d4ac0d",
    "avg_50m_k8_learnable_pos": "#1a5276",
    "avg_50m_k8_wexp": "#6c3483",
    "model1_50m_update_ctrl": "#e17055",
    "model1_500m": "#4e9de0",
    "avg_500m_k2": "#3fb950",
}


# ── Chinchilla scaling law ────────────────────────────────────────────────
_A, _ALPHA = 406.4, 0.3392
_B, _BETA = 410.7, 0.2849
_E = 1.6934


def chinchilla_loss(N: float, D: float) -> float:
    return _A / N**_ALPHA + _B / D**_BETA + _E


def chinchilla_optimal(cfg) -> tuple[float, float, float]:
    """(tokens_seen_opt, flops_opt, loss_opt) at the Chinchilla D*=20N point."""
    N = cfg.n_params_approx
    D_opt = 20.0 * N
    tokens_seen_opt = cfg.averaging_k * D_opt
    flops_opt = (6.0 / cfg.averaging_k) * N * D_opt
    loss_opt = chinchilla_loss(N, D_opt)
    return tokens_seen_opt, flops_opt, loss_opt


# ── FLOPs helpers ─────────────────────────────────────────────────────────

# Padded vocabulary size (tokenizer vocab 50257 rounded up for the head).
VOCAB_SIZE = 50304


def _per_seq_flops(cfg, raw_seq_len: int) -> float:
    """Training FLOPs for one raw sequence of `raw_seq_len` tokens.

    Transformer body plus the dense output projection, which emits one
    prediction per transformer position. The head is a large share at these
    widths (44% of the total at d=512), so excluding it would inflate every
    efficiency ratio in favour of the averaged arms.
    """
    L = raw_seq_len // cfg.averaging_k
    d = cfg.d_model
    body = cfg.n_layers * (23 * d * d + 4 * L * d)
    head = 2 * d * VOCAB_SIZE
    return 3 * L * (body + head)


def compute_cumulative_flops(df: pd.DataFrame, cfg) -> np.ndarray:
    """Recompute cumulative training FLOPs from tokens_seen.

    seq_len = context_len (raw), transformer_L = context_len / k.
    """
    seq_len = cfg.context_len
    return (df["tokens_seen"].values / seq_len) * _per_seq_flops(cfg, seq_len)


def _ctx_flops(df: pd.DataFrame, cfg, raw_seq_len: int) -> np.ndarray:
    """Cumulative FLOPs for a Config-B run with an explicit raw_seq_len."""
    return (df["tokens_seen"].values / raw_seq_len) * _per_seq_flops(cfg, raw_seq_len)


# ── Matplotlib style ──────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


def _style_ax(ax, xlabel: str, title: str, xlog: bool = False) -> None:
    ax.set_facecolor("white")
    ax.tick_params(which="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", color="#cccccc", linewidth=0.6, alpha=0.8)
    ax.grid(True, which="minor", color="#e6e6e6", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title(title, fontsize=13, pad=10, fontweight="bold")
    if xlog:
        ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1e}"))


def _add_legend(ax, show_star: bool = True) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if show_star:
        star = mlines.Line2D(
            [],
            [],
            color="#333333",
            marker="*",
            linestyle="None",
            markersize=10,
            markeredgecolor="#333333",
            label="Chinchilla optimal (D*=20N)",
        )
        handles.append(star)
        labels.append("Chinchilla optimal (D*=20N)")
    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#cccccc",
        labelcolor="#222222",
    )


# ── Core plotting ─────────────────────────────────────────────────────────


def _plot_tokens(
    logs: dict, xlog: bool, out_path: Path, title_tag: str, show_chinchilla: bool
) -> None:
    scale = "log" if xlog else "linear"
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    _style_ax(
        ax,
        "Tokens Seen",
        f"Loss vs Tokens Seen  ({title_tag})  [{scale} scale]",
        xlog=xlog,
    )

    for name, (cfg, df, label) in logs.items():
        color = COLORS.get(name, cfg.color)
        tokens = df["tokens_seen"].values
        ev = df["eval_loss"].values
        mask = ~np.isnan(ev) & (ev <= 8.0)
        if mask.any():
            ax.plot(tokens[mask], ev[mask], color=color, linewidth=2.2, label=label)

    if show_chinchilla:
        for name, (cfg, _, _) in logs.items():
            color = COLORS.get(name, cfg.color)
            tok_opt, _, loss_opt = chinchilla_optimal(cfg)
            ax.scatter(
                [tok_opt],
                [loss_opt],
                color=color,
                s=140,
                zorder=6,
                edgecolors="#333333",
                linewidths=1.0,
                marker="*",
            )

    _add_legend(ax, show_star=show_chinchilla)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out_path}")


def _plot_flops(
    logs: dict, xlog: bool, out_path: Path, title_tag: str, show_chinchilla: bool
) -> None:
    scale = "log" if xlog else "linear"
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    _style_ax(
        ax,
        "Cumulative Training FLOPs  (3·L·(N_layers·(23d²+4Ld) + 2dV))",
        f"Loss vs FLOPs  ({title_tag})  [{scale} scale]",
        xlog=xlog,
    )

    for name, (cfg, df, label) in logs.items():
        color = COLORS.get(name, cfg.color)
        flops = compute_cumulative_flops(df, cfg)
        ev = df["eval_loss"].values
        mask = ~np.isnan(ev) & (ev <= 8.0)
        if mask.any():
            ax.plot(flops[mask], ev[mask], color=color, linewidth=2.2, label=label)

    if show_chinchilla:
        for name, (cfg, _, _) in logs.items():
            color = COLORS.get(name, cfg.color)
            _, flops_opt, loss_opt = chinchilla_optimal(cfg)
            ax.scatter(
                [flops_opt],
                [loss_opt],
                color=color,
                s=140,
                zorder=6,
                edgecolors="#333333",
                linewidths=1.0,
                marker="*",
            )

    _add_legend(ax, show_star=show_chinchilla)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out_path}")


def plot_models(
    models: list[str],
    *,
    prefix: str,
    title: str,
    csv_name: str = "loss_log.csv",
    show_chinchilla: bool = True,
    scales: list[str] | None = None,
) -> None:
    """Load one CSV per model and emit loss-vs-tokens and loss-vs-FLOPs figures."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    scales = scales or ["linear", "log"]

    logs: dict[str, tuple] = {}
    for name in models:
        cfg, csv_dir, label = _resolve_run(name)
        csv_path = csv_dir / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = _drop_isolated_spikes(df)
            logs[name] = (cfg, df, label)
            print(
                f"  loaded {name}: {len(df)} steps, "
                f"{df['tokens_seen'].iloc[-1]/1e9:.2f}B tokens, "
                f"final eval {df['eval_loss'].iloc[-1]:.3f}"
            )
        else:
            print(f"  [skip] {name} — no {csv_path}")

    if not logs:
        raise RuntimeError("No CSVs found for: " + ", ".join(models))

    for scale in scales:
        xlog = scale == "log"
        _plot_tokens(
            logs,
            xlog,
            PLOTS_DIR / f"{prefix}_loss_vs_tokens_{scale}.png",
            title,
            show_chinchilla,
        )
        _plot_flops(
            logs,
            xlog,
            PLOTS_DIR / f"{prefix}_loss_vs_flops_{scale}.png",
            title,
            show_chinchilla,
        )


def plot_ctx(*, scales: list[str] | None = None) -> None:
    """Config-B context-scaling runs (per-run CSV / raw_seq_len overrides)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    scales = scales or ["linear", "log"]
    title_tag = "50M, same transformer length, k\u00d7 raw context"

    runs = []
    for name, csv_name, raw_seq, label, color in CTX_RUNS:
        csv_path = RESULTS_DIR / name / csv_name
        if not csv_path.exists():
            print(f"  [skip] {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df = _drop_isolated_spikes(df)
        cfg = MODEL_CONFIGS[name]
        runs.append((name, df, cfg, raw_seq, label, color))
        print(
            f"  loaded {name} ({csv_name}): {len(df)} rows, "
            f"{df['tokens_seen'].iloc[-1]/1e9:.2f}B tokens"
        )

    if not runs:
        raise RuntimeError("No Config-B CSVs found")

    for scale in scales:
        xlog = scale == "log"
        scale_label = "log" if xlog else "linear"

        # loss vs tokens
        fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
        _style_ax(
            ax,
            "Raw Tokens Seen",
            f"Loss vs Tokens  ({title_tag})  [{scale_label} scale]",
            xlog=xlog,
        )
        for _, df, _, _, label, color in runs:
            ev = df["eval_loss"].values
            mask = ~np.isnan(ev) & (ev <= 8.0)
            ax.plot(
                df["tokens_seen"].values[mask],
                ev[mask],
                color=color,
                linewidth=2.2,
                label=label,
            )
        _add_legend(ax, show_star=False)
        out = PLOTS_DIR / f"50m_kx_ctx_loss_vs_tokens_{scale_label}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  → {out}")

        # loss vs FLOPs
        fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
        _style_ax(
            ax,
            "Cumulative Training FLOPs  (3·L·(N_layers·(23d²+4Ld) + 2dV), L=seq/k)",
            f"Loss vs FLOPs  ({title_tag})  [{scale_label} scale]",
            xlog=xlog,
        )
        for name, df, cfg, raw_seq, label, color in runs:
            flops = _ctx_flops(df, cfg, raw_seq)
            ev = df["eval_loss"].values
            mask = ~np.isnan(ev) & (ev <= 8.0)
            ax.plot(flops[mask], ev[mask], color=color, linewidth=2.2, label=label)
        _add_legend(ax, show_star=False)
        out = PLOTS_DIR / f"50m_kx_ctx_loss_vs_flops_{scale_label}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────


def _list_available() -> None:
    """Print all registered model configs and all result CSVs on disk."""
    print("Registered model configs:")
    for name in sorted(MODEL_CONFIGS):
        cfg = MODEL_CONFIGS[name]
        print(
            f"  {name:28s}  k={cfg.averaging_k}  d={cfg.d_model}  L={cfg.n_layers}  "
            f"ctx={cfg.context_len}  tokens={cfg.target_tokens/1e9:.1f}B"
        )

    print(f"\nPreset groups (--preset):")
    for pname, members in sorted(PRESETS.items()):
        if members:
            print(f"  {pname:16s}  {', '.join(members)}")
        else:
            print(f"  {pname:16s}  (special handler)")

    print(f"\nCSVs on disk ({RESULTS_DIR}):")
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        csvs = sorted(d.glob("*.csv"))
        if csvs:
            names = ", ".join(c.name for c in csvs)
            print(f"  {d.name:28s}  {names}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot loss curves for token-averaging experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --list to see all available models, presets, and CSVs.",
    )
    ap.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        help="Model config names to plot (from model_configs.py). "
        "Can be combined with --preset.",
    )
    ap.add_argument(
        "--preset",
        action="append",
        default=[],
        metavar="NAME",
        help="Preset model group (125m, 250m, 50m, ablations-k2, ablations-k4, ctx). "
        "Can be specified multiple times.",
    )
    ap.add_argument("--title", type=str, default=None, help="Plot title tag.")
    ap.add_argument("--prefix", type=str, default=None, help="Output filename prefix.")
    ap.add_argument(
        "--csv",
        type=str,
        default="loss_log.csv",
        dest="csv_name",
        help="CSV filename to load from each model's results dir (default: loss_log.csv).",
    )
    ap.add_argument(
        "--no-chinchilla",
        action="store_true",
        help="Hide Chinchilla optimal stars.",
    )
    ap.add_argument(
        "--scale",
        choices=["log", "linear", "both"],
        default="both",
        help="Which x-axis scales to generate (default: both).",
    )
    ap.add_argument(
        "--list", action="store_true", help="List available models and exit."
    )
    args = ap.parse_args()

    if args.list:
        _list_available()
        return

    scales = ["linear", "log"] if args.scale == "both" else [args.scale]

    # Collect ctx presets separately (they need special handling)
    has_ctx = "ctx" in args.preset
    regular_presets = [p for p in args.preset if p != "ctx"]

    # Expand presets into model list
    models: list[str] = []
    for p in regular_presets:
        if p not in PRESETS:
            ap.error(f"Unknown preset: {p!r}. Valid: {', '.join(PRESETS)}")
        models.extend(PRESETS[p])
    models.extend(args.models)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for m in models:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    models = unique

    # Validate model names (alias names in ALT_RESULTS are also allowed)
    for m in models:
        if m not in MODEL_CONFIGS and m not in ALT_RESULTS:
            ap.error(
                f"Unknown model config: {m!r}. Run --list to see available configs."
            )

    if not models and not has_ctx:
        ap.error(
            "Nothing to plot. Provide model names, --preset, or both. "
            "Use --list to see options."
        )

    # Plot standard models
    if models:
        title = args.title or ", ".join(
            sorted({m.split("_")[0] + "_" + m.split("_")[1] for m in models})
        )
        prefix = args.prefix or "_".join(sorted({m.split("_")[1] for m in models}))
        print(f"Plotting {len(models)} models: {', '.join(models)}")
        plot_models(
            models,
            prefix=prefix,
            title=title,
            csv_name=args.csv_name,
            show_chinchilla=not args.no_chinchilla,
            scales=scales,
        )

    # Plot Config-B context runs
    if has_ctx:
        print("Plotting Config-B (k× context) runs...")
        plot_ctx(scales=scales)

    print("Done.")


if __name__ == "__main__":
    main()
