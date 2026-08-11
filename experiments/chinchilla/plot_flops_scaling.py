"""
Visualize how training FLOPs scale across model sizes for k=1 vs k=2 averaging.

Uses the formula:
    Training FLOPs = (tokens_seen / seq_len) × n_layers × transformer_L × 3 × (23d² + 4 × transformer_L × d)

Token counting:
    - tokens_seen = raw tokens fed to the model
    - transformer_L = seq_len / k  (positions the transformer actually processes per sequence)
    - For k=2: each sequence of 1024 raw tokens → 512 transformer positions
    - To give the transformer the same total positions as k=1 (D* = 20N),
      k=2 needs 2× raw tokens (2×D* raw → D* transformer positions)

Model architectures (GPT-style, tied embeddings):
    ~50M:  d=512,  h=8,  l=8,  ctx=1024
    ~125M: d=768,  h=12, l=12, ctx=1024
    ~300M: d=1024, h=16, l=24, ctx=1024
    ~1B:   d=2048, h=16, l=24, ctx=1024

Chinchilla-optimal: D* = 20N (transformer positions, not raw tokens)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

SEQ_LEN = 1024

MODELS = {
    "~50M": {"d_model": 512, "n_layers": 8, "n_params": 51e6},
    "~75M": {"d_model": 640, "n_layers": 12, "n_params": 75e6},
    "~125M": {"d_model": 768, "n_layers": 12, "n_params": 123e6},
    "~150M": {"d_model": 896, "n_layers": 12, "n_params": 150e6},
    "~300M": {"d_model": 1024, "n_layers": 24, "n_params": 302e6},
    "~500M": {"d_model": 1280, "n_layers": 24, "n_params": 500e6},
    "~1B": {"d_model": 2048, "n_layers": 24, "n_params": 1.1e9},
}


def calculate_flops(tokens_seen: float, seq_len: int, n_layers: int, d_model: int, transformer_L: int) -> float:
    return (tokens_seen / seq_len) * n_layers * transformer_L * 3 * (23 * d_model**2 + 4 * transformer_L * d_model)


def chinchilla_optimal_tokens(n_params: float) -> float:
    """D* = 20N transformer positions (Chinchilla-optimal)."""
    return 20 * n_params


def main():
    model_names = list(MODELS.keys())
    n_params_list = [MODELS[m]["n_params"] for m in model_names]

    # D* = 20N = Chinchilla-optimal transformer positions
    chinchilla_positions = [chinchilla_optimal_tokens(MODELS[m]["n_params"]) for m in model_names]

    flops_k1 = []
    flops_k2 = []

    for name in model_names:
        cfg = MODELS[name]
        d = cfg["d_model"]
        l = cfg["n_layers"]
        D_star = chinchilla_optimal_tokens(cfg["n_params"])

        L_k1 = SEQ_LEN          # transformer_L for k=1
        L_k2 = SEQ_LEN // 2     # transformer_L for k=2

        # k=1: raw tokens = D*, transformer sees D* positions
        raw_tokens_k1 = D_star
        # k=2: need 2× raw tokens so transformer sees D* positions
        #       (each seq of 1024 raw → 512 positions, so 2×D* raw → D* positions)
        raw_tokens_k2 = D_star * 2

        f_k1 = calculate_flops(raw_tokens_k1, SEQ_LEN, l, d, L_k1)
        f_k2 = calculate_flops(raw_tokens_k2, SEQ_LEN, l, d, L_k2)

        flops_k1.append(f_k1)
        flops_k2.append(f_k2)

    flops_k1 = np.array(flops_k1)
    flops_k2 = np.array(flops_k2)

    # --- Plot 1: Absolute FLOPs by model size ---
    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(model_names))
    width = 0.3

    bars1 = ax.bar(x - width/2, flops_k1, width,
                   label="k=1  (D* raw tokens, transformer sees D* positions)", color="#4e9de0")
    bars2 = ax.bar(x + width/2, flops_k2, width,
                   label="k=2  (2×D* raw tokens, transformer sees D* positions)", color="#3fb950")

    ax.set_xlabel("Model Size", fontsize=12)
    ax.set_ylabel("Training FLOPs", fontsize=12)
    ax.set_title("Training FLOPs: k=1 vs k=2 Averaging\n(both see D*=20N transformer positions, seq_len=1024)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_yscale("log")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0e}"))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "flops_scaling_absolute.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 2: FLOPs ratio (k=2 / k=1) ---
    ratio = flops_k2 / flops_k1

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(model_names, ratio, "o-", color="#3fb950", linewidth=2.5, markersize=12)
    ax.axhline(1.0, color="#4e9de0", linestyle="--", linewidth=1.5, alpha=0.7, label="k=1 baseline (1.0×)")

    for i, r in enumerate(ratio):
        savings_pct = (1 - r) * 100
        ax.annotate(f"{r:.3f}×\n({savings_pct:.1f}% saved)",
                    (i, r), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=10, fontweight="bold", color="#2d8a3e")

    ax.set_xlabel("Model Size", fontsize=12)
    ax.set_ylabel("FLOPs Ratio  (k=2 / k=1)", fontsize=12)
    ax.set_title("FLOPs Cost of k=2 Averaging vs Standard (k=1)\n"
                 "(both transformers see D*=20N positions; k=2 uses 2× raw tokens)",
                 fontsize=12)
    ax.set_ylim(0.75, 1.05)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "flops_scaling_ratio.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 3: FLOPs savings breakdown (attention vs FFN) ---
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ffn_term = np.array([23 * MODELS[m]["d_model"]**2 for m in model_names])
    attn_k1 = np.array([4 * SEQ_LEN * MODELS[m]["d_model"] for m in model_names])
    attn_k2 = np.array([4 * (SEQ_LEN // 2) * MODELS[m]["d_model"] for m in model_names])

    total_k1 = ffn_term + attn_k1
    total_k2 = ffn_term + attn_k2

    attn_fraction_k1 = attn_k1 / total_k1
    attn_fraction_k2 = attn_k2 / total_k2

    ax.bar(x - 0.15, attn_fraction_k1, 0.3, label="Attention fraction (k=1)", color="#4e9de0", alpha=0.8)
    ax.bar(x + 0.15, attn_fraction_k2, 0.3, label="Attention fraction (k=2)", color="#3fb950", alpha=0.8)

    for i, (a1, a2) in enumerate(zip(attn_fraction_k1, attn_fraction_k2)):
        ax.annotate(f"{a1:.1%}", (i - 0.15, a1), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=9, color="#4e9de0")
        ax.annotate(f"{a2:.1%}", (i + 0.15, a2), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=9, color="#3fb950")

    ax.set_xlabel("Model Size", fontsize=12)
    ax.set_ylabel("Attention Fraction of Per-Token FLOPs", fontsize=12)
    ax.set_title("Attention vs FFN FLOPs Fraction\n(explains why savings shrink for larger d_model)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "flops_scaling_attention_fraction.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Print summary table ---
    print(f"\n{'Model':<8} {'N_params':>10} {'D* (20N)':>12} {'Raw tok k=1':>12} {'Raw tok k=2':>12} "
          f"{'FLOPs k=1':>14} {'FLOPs k=2':>14} {'k=2/k=1':>8} {'Saved':>7}")
    print("-" * 105)
    for i, name in enumerate(model_names):
        D_star = chinchilla_positions[i]
        print(f"{name:<8} {n_params_list[i]/1e6:>8.0f}M {D_star/1e9:>10.1f}B "
              f"{D_star/1e9:>10.1f}B {2*D_star/1e9:>10.1f}B "
              f"{flops_k1[i]:>14.3e} {flops_k2[i]:>14.3e} "
              f"{ratio[i]:>8.3f} {(1-ratio[i])*100:>5.1f}%")

    print(f"\nNote: k=2 needs 2× raw tokens so the transformer sees the same D* positions as k=1.")
    print(f"      transformer_L = seq_len/k → k=2 has L=512, k=1 has L=1024.")
    print(f"      Savings come from reduced attention cost (4·L·d term shrinks with L).")
    print(f"\nPlots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
