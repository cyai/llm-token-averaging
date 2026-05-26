"""
plot_info_loss.py — Information-loss & variance-shrinkage sweep across all
averaging methods.

Produces two figures:
  1. info_loss_plot.png   — cosine retention + entropy retention
  2. variance_plot.png    — variance shrinkage (actual vs. theoretical 1/k),
                            per-family relative shrinkage, overlapping heatmap

Usage
-----
# Fast run — synthetic embeddings, no model download
python plot_info_loss.py --synthetic

# Real model, embedding layer
python plot_info_loss.py --num_sequences 50

# Custom output names
python plot_info_loss.py --synthetic --output loss.png --output_variance var.png
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def cosine_retention(original: np.ndarray, groups: List[Tuple[int, int]]) -> float:
    """
    Mean cosine similarity between each group's average embedding and every
    individual token in that group.  Works for any grouping scheme.

    Args:
        original: [B, seq_len, D]
        groups:   list of (start, end) index pairs (end exclusive)

    Returns:
        Scalar in [-1, 1]; higher = more information preserved after averaging.
    """
    B, L, D = original.shape
    sims: List[float] = []
    for start, end in groups:
        end = min(end, L)
        if end <= start:
            continue
        chunk = original[:, start:end, :]            # [B, k, D]
        avg   = chunk.mean(axis=1, keepdims=True)    # [B, 1, D]

        eps = 1e-8
        chunk_n = chunk / (np.linalg.norm(chunk, axis=-1, keepdims=True) + eps)
        avg_n   = avg   / (np.linalg.norm(avg,   axis=-1, keepdims=True) + eps)

        sim = (chunk_n * avg_n).sum(axis=-1)         # [B, k]
        sims.append(float(sim.mean()))

    return float(np.mean(sims)) if sims else 1.0


def variance_shrinkage(
    original: np.ndarray,
    groups: List[Tuple[int, int]],
) -> Tuple[float, float]:
    """
    Compute the variance shrinkage factor after averaging.

    For independent tokens, averaging k tokens shrinks variance by exactly 1/k
    (the theoretical prediction).  Positive inter-token correlation means less
    shrinkage (tokens are redundant); negative correlation means more shrinkage.

    Args:
        original: [B, seq_len, D]
        groups:   list of (start, end) index pairs

    Returns:
        (shrinkage_factor, mean_group_size)
        shrinkage_factor = Var(averaged_tokens) / Var(original_tokens)
        A value of 1/k means tokens were independent.
        A value > 1/k means adjacent tokens are positively correlated (redundant).
        A value < 1/k means adjacent tokens are negatively correlated.
    """
    B, L, D = original.shape

    orig_flat = original.reshape(-1, D)
    var_orig = float(orig_flat.var(axis=0).mean())

    avg_tokens: List[np.ndarray] = []
    group_sizes: List[int] = []
    for start, end in groups:
        end = min(end, L)
        if end <= start:
            continue
        chunk = original[:, start:end, :]
        avg_tokens.append(chunk.mean(axis=1))   # [B, D]
        group_sizes.append(end - start)

    if not avg_tokens:
        return 1.0, 1.0

    avg_arr  = np.stack(avg_tokens, axis=1)      # [B, n_groups, D]
    avg_flat = avg_arr.reshape(-1, D)
    var_avg  = float(avg_flat.var(axis=0).mean())

    mean_k = float(np.mean(group_sizes))
    shrinkage = var_avg / var_orig if var_orig > 0 else 1.0
    return shrinkage, mean_k


def entropy_retention(original: np.ndarray, averaged: np.ndarray, n_bins: int = 40) -> float:
    """
    Ratio of per-dimension entropy: H(averaged) / H(original).
    A value of 1.0 means the averaged embeddings carry as much entropy as the
    originals; lower values indicate information loss.

    Args:
        original:  [B, L,   D]
        averaged:  [B, L/k, D]

    Returns:
        Scalar in [0, ∞]; values above 1 mean the average is *more* varied than
        the original (possible due to pooling effects on high-k runs).
    """
    def _mean_entropy(arr: np.ndarray) -> float:
        flat = arr.reshape(-1, arr.shape[-1])
        entropies = []
        for d in range(min(flat.shape[1], 64)):
            col = flat[:, d]
            hist, _ = np.histogram(col, bins=n_bins, density=False)
            hist = hist.astype(float)
            total = hist.sum()
            if total == 0:
                continue
            p = hist[hist > 0] / total
            entropies.append(float(-np.sum(p * np.log(p + 1e-12))))
        return float(np.mean(entropies)) if entropies else 0.0

    h_orig = _mean_entropy(original)
    h_avg  = _mean_entropy(averaged)
    if h_orig == 0:
        return 1.0
    return h_avg / h_orig


# ---------------------------------------------------------------------------
# Group builders (return List[Tuple[int, int]])
# ---------------------------------------------------------------------------

def uniform_groups(seq_len: int, k: int) -> List[Tuple[int, int]]:
    return [(i * k, (i + 1) * k) for i in range(seq_len // k)]


def overlapping_groups(seq_len: int, window: int, stride: int) -> List[Tuple[int, int]]:
    groups = []
    pos = 0
    while pos + window <= seq_len:
        groups.append((pos, pos + window))
        pos += stride
    return groups


def alternating_groups(seq_len: int, pattern: List[int]) -> List[Tuple[int, int]]:
    groups, pos, idx = [], 0, 0
    while pos < seq_len:
        k = pattern[idx % len(pattern)]
        if pos + k > seq_len:
            break
        groups.append((pos, pos + k))
        pos += k
        idx += 1
    return groups


def random_groups(seq_len: int, k_min: int, k_max: int, seed: int = 42) -> List[Tuple[int, int]]:
    rng = np.random.RandomState(seed)
    groups, pos = [], 0
    while pos < seq_len:
        k = int(rng.randint(k_min, k_max + 1))
        if pos + k > seq_len:
            break
        groups.append((pos, pos + k))
        pos += k
    return groups


def adaptive_groups(
    embeddings: np.ndarray,
    k_min: int,
    k_max: int,
    threshold: float = 0.85,
) -> List[Tuple[int, int]]:
    """Group boundaries driven by cosine similarity of adjacent tokens."""
    seq = embeddings[0]                        # [L, D]
    L   = seq.shape[0]
    norm = seq / (np.linalg.norm(seq, axis=-1, keepdims=True) + 1e-8)
    sims = (norm[:-1] * norm[1:]).sum(axis=-1) # [L-1]

    groups, pos = [], 0
    while pos < L:
        k = k_min
        for ahead in range(pos, min(pos + k_max - 1, L - 1)):
            if sims[ahead] >= threshold:
                k = ahead - pos + 2
            else:
                break
        k = max(k_min, min(k, k_max))
        if pos + k > L:
            break
        groups.append((pos, pos + k))
        pos += k
    return groups


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_layer_embeddings(
    model,
    tokenizer,
    num_sequences: int,
    max_length: int,
    batch_size: int,
    layer_key: str,
    device: str,
) -> np.ndarray:
    """
    Run the model on WikiText-103 sequences and return embeddings for one layer.

    Returns:
        [N_sequences, seq_len, D]  (not batched further)
    """
    from utils.data_loader import get_data_iterator

    all_embs: List[np.ndarray] = []
    collected = 0

    for batch in get_data_iterator(
        tokenizer,
        num_sequences=num_sequences,
        max_length=max_length,
        batch_size=batch_size,
        split="train",
    ):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        layer_outputs = model.extract(input_ids, attention_mask)

        if layer_key == "last":
            # Find the highest-numbered transformer layer
            keys   = [k for k in layer_outputs if k.startswith("layer_")]
            chosen = sorted(keys, key=lambda x: int(x.split("_")[1]))[-1]
        elif layer_key == "embedding":
            chosen = "embedding"
        else:
            chosen = layer_key

        emb = layer_outputs[chosen].cpu().float().numpy()  # [B, L, D]
        all_embs.append(emb)

        collected += emb.shape[0]
        log.info(f"  collected {collected}/{num_sequences} sequences (layer={chosen})")
        if collected >= num_sequences:
            break

    return np.concatenate(all_embs, axis=0)[:num_sequences]


def make_synthetic_embeddings(num_sequences: int, seq_len: int, d_model: int) -> np.ndarray:
    """Random realistic-ish embeddings (no model needed)."""
    rng = np.random.RandomState(42)
    # Simulate redundancy: adjacent tokens are correlated
    base = rng.randn(num_sequences, seq_len, d_model).astype(np.float32)
    smoothed = np.zeros_like(base)
    for t in range(seq_len):
        w = base[:, max(0, t-2):t+3, :]
        smoothed[:, t, :] = w.mean(axis=1)
    return smoothed


# ---------------------------------------------------------------------------
# Build all (method, label, compression_ratio, groups) configurations
# ---------------------------------------------------------------------------

def build_configs(
    seq_len: int,
    embeddings: np.ndarray,
    uniform_ks: List[int],
    overlap_windows: List[int],
    overlap_strides_per_window: Dict[int, List[int]],
    dynamic_configs: List[dict],
    weighted_ks: List[int],
    weighted_schemes: List[str],
) -> List[dict]:
    """
    Returns a list of dicts with keys:
      family, label, compression_ratio, groups, k_nominal
    """
    configs = []

    # ---- Uniform k ----
    for k in uniform_ks:
        groups = uniform_groups(seq_len, k)
        if len(groups) == 0:
            continue
        configs.append(dict(
            family="uniform",
            label=f"uniform k={k}",
            compression_ratio=float(k),
            groups=groups,
            k_nominal=k,
        ))

    # ---- Overlapping windows ----
    for w in overlap_windows:
        for s in overlap_strides_per_window.get(w, []):
            if s > w or s < 1:
                continue
            groups = overlapping_groups(seq_len, w, s)
            if len(groups) == 0:
                continue
            cr = w / s  # compression ratio: original tokens per output token
            configs.append(dict(
                family="overlapping",
                label=f"overlap w={w} s={s}",
                compression_ratio=cr,
                groups=groups,
                k_nominal=w,
            ))

    # ---- Dynamic k ----
    for dcfg in dynamic_configs:
        strategy = dcfg["strategy"]
        if strategy == "alternating":
            pattern = dcfg["pattern"]
            groups  = alternating_groups(seq_len, pattern)
            mean_k  = float(np.mean([e - s for s, e in groups])) if groups else 1
            label   = f"dynamic alt {pattern}"
        elif strategy == "random":
            k_min, k_max = dcfg["k_min"], dcfg["k_max"]
            groups = random_groups(seq_len, k_min, k_max)
            mean_k = float(np.mean([e - s for s, e in groups])) if groups else 1
            label  = f"dynamic rnd [{k_min},{k_max}]"
        elif strategy == "adaptive":
            k_min, k_max = dcfg["k_min"], dcfg["k_max"]
            groups = adaptive_groups(embeddings, k_min, k_max)
            mean_k = float(np.mean([e - s for s, e in groups])) if groups else 1
            label  = f"dynamic adaptive [{k_min},{k_max}]"
        else:
            continue

        if len(groups) == 0:
            continue
        configs.append(dict(
            family="dynamic",
            label=label,
            compression_ratio=mean_k,
            groups=groups,
            k_nominal=mean_k,
        ))

    # ---- Weighted (non-overlapping, same groups as uniform but different weights) ----
    # Cosine retention is purely a function of the grouping, not the weights,
    # so weighted methods share the same cosine retention as uniform k.
    # We mark them separately so they appear in the plot legend.
    for k in weighted_ks:
        groups = uniform_groups(seq_len, k)
        if len(groups) == 0:
            continue
        for scheme in weighted_schemes:
            if scheme == "uniform":
                continue  # identical to uniform k — already covered
            configs.append(dict(
                family="weighted",
                label=f"weighted {scheme} k={k}",
                compression_ratio=float(k),
                groups=groups,
                k_nominal=k,
            ))

    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot information loss for all averaging methods."
    )
    p.add_argument("--model",          default="EleutherAI/pythia-410m",
                   help="HuggingFace model name (default: pythia-410m)")
    p.add_argument("--num_sequences",  type=int,   default=30,
                   help="Sequences to process (default: 30 — fast)")
    p.add_argument("--max_length",     type=int,   default=256,
                   help="Sequence length (default: 256)")
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--layer",          default="embedding",
                   choices=["embedding", "last"],
                   help="Which layer to analyse (default: embedding)")
    p.add_argument("--device",         default=None,
                   help="cuda or cpu (default: auto-detect)")
    p.add_argument("--output",         default="info_loss_plot.png",
                   help="Output PNG path (default: info_loss_plot.png)")
    p.add_argument("--synthetic",        action="store_true",
                   help="Use synthetic random embeddings (no model download)")
    p.add_argument("--uniform_k_max",   type=int,   default=32,
                   help="Maximum k for uniform sweep (default: 32)")
    p.add_argument("--output_variance", default="variance_plot.png",
                   help="Output PNG for variance shrinkage plot (default: variance_plot.png)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        from utils.model_loader import auto_device
        args.device = auto_device()

    log.info(f"Device: {args.device}")
    log.info(f"Sequences: {args.num_sequences}  |  max_length: {args.max_length}")

    # ------------------------------------------------------------------
    # 1. Get embeddings
    # ------------------------------------------------------------------
    if args.synthetic:
        log.info("Using synthetic embeddings (--synthetic flag set)")
        embeddings = make_synthetic_embeddings(
            args.num_sequences, args.max_length, d_model=512
        )
    else:
        log.info(f"Loading model: {args.model}")
        from utils.model_loader import load_pythia_model
        model, tokenizer = load_pythia_model(args.model, device=args.device)

        log.info("Extracting embeddings …")
        embeddings = extract_layer_embeddings(
            model, tokenizer,
            num_sequences=args.num_sequences,
            max_length=args.max_length,
            batch_size=args.batch_size,
            layer_key=args.layer,
            device=args.device,
        )
        model.remove_hooks()

    B, L, D = embeddings.shape
    log.info(f"Embeddings shape: {embeddings.shape}")

    # ------------------------------------------------------------------
    # 2. Build configs
    # ------------------------------------------------------------------
    uniform_ks = [k for k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
                  if k <= args.uniform_k_max and k <= L // 2]

    overlap_windows = [w for w in [2, 4, 8, 16] if w <= args.uniform_k_max // 2]
    overlap_strides: Dict[int, List[int]] = {}
    for w in overlap_windows:
        overlap_strides[w] = list(range(1, w + 1))  # stride 1 … w (w=no-overlap)

    dynamic_cfgs = [
        dict(strategy="alternating", pattern=[2, 3]),
        dict(strategy="alternating", pattern=[2, 4]),
        dict(strategy="alternating", pattern=[3, 5]),
        dict(strategy="random",      k_min=2, k_max=4),
        dict(strategy="random",      k_min=2, k_max=8),
        dict(strategy="random",      k_min=4, k_max=16),
        dict(strategy="adaptive",    k_min=2, k_max=4),
        dict(strategy="adaptive",    k_min=2, k_max=8),
    ]

    weighted_ks      = [k for k in [2, 4, 8, 16] if k <= args.uniform_k_max]
    weighted_schemes = ["uniform", "linear", "exponential", "gaussian", "triangular"]

    configs = build_configs(
        seq_len=L,
        embeddings=embeddings,
        uniform_ks=uniform_ks,
        overlap_windows=overlap_windows,
        overlap_strides_per_window=overlap_strides,
        dynamic_configs=dynamic_cfgs,
        weighted_ks=weighted_ks,
        weighted_schemes=weighted_schemes,
    )

    log.info(f"Total configurations to evaluate: {len(configs)}")

    # ------------------------------------------------------------------
    # 3. Compute metrics for each config
    # ------------------------------------------------------------------
    for cfg in configs:
        groups = cfg["groups"]
        cos_r  = cosine_retention(embeddings, groups)
        cfg["cosine_retention"] = cos_r
        cfg["info_loss"]        = 1.0 - cos_r

    # Also compute entropy retention for uniform k configs (illustrative)
    log.info("Computing entropy retention for uniform-k configs …")
    for cfg in configs:
        if cfg["family"] != "uniform":
            cfg["entropy_retention"] = None
            continue
        k      = cfg["k_nominal"]
        groups = cfg["groups"]
        # Build averaged array: repeat each group's average k times for shape [B, L, D]
        n_groups = len(groups)
        avg_arr  = np.zeros((B, n_groups, D), dtype=embeddings.dtype)
        for gi, (start, end) in enumerate(groups):
            avg_arr[:, gi, :] = embeddings[:, start:end, :].mean(axis=1)
        er = entropy_retention(embeddings[:, :n_groups * int(k), :], avg_arr)
        cfg["entropy_retention"] = er

    # Variance shrinkage for every config
    log.info("Computing variance shrinkage …")
    for cfg in configs:
        vs, mean_k = variance_shrinkage(embeddings, cfg["groups"])
        cfg["var_shrinkage"]   = vs
        cfg["mean_group_size"] = mean_k
        cfg["theoretical_var_shrinkage"] = 1.0 / mean_k if mean_k > 0 else 1.0
        # relative = actual / theoretical; >1 means tokens are positively correlated
        cfg["relative_var_shrinkage"] = (
            vs / cfg["theoretical_var_shrinkage"]
            if cfg["theoretical_var_shrinkage"] > 0 else 1.0
        )

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    family_style = {
        "uniform":     dict(marker="o", linestyle="-",  linewidth=2.2),
        "overlapping": dict(marker="^", linestyle="--", linewidth=1.6),
        "dynamic":     dict(marker="s", linestyle="-.", linewidth=1.6),
        "weighted":    dict(marker="D", linestyle=":",  linewidth=1.4),
    }

    family_colors = {
        "uniform":     cm.Blues,
        "overlapping": cm.Oranges,
        "dynamic":     cm.Greens,
        "weighted":    cm.Purples,
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_cos, ax_ent = axes

    # --- Panel 1: cosine retention (all methods) -------------------------
    families_seen: Dict[str, List[dict]] = {}
    for cfg in configs:
        families_seen.setdefault(cfg["family"], []).append(cfg)

    legend_handles = []
    for family, cfgs in families_seen.items():
        style = family_style[family]
        cmap  = family_colors[family]
        n     = len(cfgs)

        # Sort by compression ratio for a clean curve
        cfgs_sorted = sorted(cfgs, key=lambda x: x["compression_ratio"])

        # Give each config a shade of the family's colormap
        colors = [cmap(0.4 + 0.5 * i / max(n - 1, 1)) for i in range(n)]

        for i, cfg in enumerate(cfgs_sorted):
            h = ax_cos.scatter(
                cfg["compression_ratio"],
                cfg["cosine_retention"],
                color=colors[i],
                s=60,
                zorder=5,
            )

        # Draw a trend line through the family
        xs = [c["compression_ratio"] for c in cfgs_sorted]
        ys = [c["cosine_retention"]  for c in cfgs_sorted]

        # Sort unique x values for the trend line
        combined = sorted(zip(xs, ys))
        xs_sorted = [p[0] for p in combined]
        ys_sorted = [p[1] for p in combined]

        line, = ax_cos.plot(
            xs_sorted, ys_sorted,
            color=cmap(0.65),
            label=family,
            alpha=0.55,
            **style,
        )
        legend_handles.append(line)

    ax_cos.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.4,
                   label="Perfect retention (1.0)")
    ax_cos.axhline(y=0.9, color="red",   linestyle=":",  linewidth=1, alpha=0.5,
                   label=">0.9 viability threshold")

    ax_cos.set_xlabel("Compression Ratio (original tokens per output token)", fontsize=12)
    ax_cos.set_ylabel("Cosine Retention  (higher = less information loss)", fontsize=12)
    ax_cos.set_title("Information Retention: All Methods vs. Compression Ratio", fontsize=13, fontweight="bold")
    ax_cos.set_xscale("log", base=2)
    ax_cos.set_ylim(bottom=max(0, min(c["cosine_retention"] for c in configs) - 0.05),
                    top=1.02)
    ax_cos.legend(fontsize=10)
    ax_cos.grid(True, alpha=0.3)
    ax_cos.tick_params(axis="both", labelsize=10)

    # Annotate a few standout configs
    for cfg in configs:
        if cfg["family"] == "uniform" and cfg["k_nominal"] in [1, 2, 4, 8, 16, 32]:
            ax_cos.annotate(
                f"k={int(cfg['k_nominal'])}",
                xy=(cfg["compression_ratio"], cfg["cosine_retention"]),
                xytext=(4, -12), textcoords="offset points",
                fontsize=7.5, color=family_colors["uniform"](0.85),
            )

    # --- Panel 2: entropy retention (uniform-k only) + cosine comparison ---
    uniform_cfgs = sorted(
        [c for c in configs if c["family"] == "uniform"],
        key=lambda x: x["k_nominal"],
    )
    ks_u   = [c["k_nominal"] for c in uniform_cfgs]
    cos_u  = [c["cosine_retention"]  for c in uniform_cfgs]
    ent_u  = [c["entropy_retention"] for c in uniform_cfgs]

    ax_ent.plot(ks_u, cos_u, marker="o", linewidth=2,  color="#2563EB",
                label="Cosine retention (uniform k)")
    if any(e is not None for e in ent_u):
        ax_ent.plot(ks_u, ent_u, marker="s", linewidth=2, linestyle="--",
                    color="#7C3AED",
                    label="Entropy retention (uniform k)")

    ax_ent.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.4)
    ax_ent.axhline(y=0.9, color="red",   linestyle=":",  linewidth=1, alpha=0.5,
                   label=">0.9 viability threshold")

    ax_ent.fill_between(ks_u, 0.9, 1.0, alpha=0.07, color="green",
                         label="High-retention zone (>0.9)")

    ax_ent.set_xlabel("Averaging Window Size k (uniform)", fontsize=12)
    ax_ent.set_ylabel("Retention Ratio", fontsize=12)
    ax_ent.set_title("Cosine vs. Entropy Retention — Uniform k Baseline", fontsize=13,
                     fontweight="bold")
    ax_ent.set_xscale("log", base=2)
    ax_ent.set_ylim(0, 1.1)
    ax_ent.legend(fontsize=10)
    ax_ent.grid(True, alpha=0.3)
    ax_ent.tick_params(axis="both", labelsize=10)

    # --- Figure-level text -----------------------------------------------
    model_label = "synthetic" if args.synthetic else args.model.split("/")[-1]
    layer_label = args.layer
    fig.suptitle(
        f"Token Averaging — Information Loss Survey\n"
        f"Model: {model_label}  |  Layer: {layer_label}  |  "
        f"N={B} seqs × {L} tokens  |  {len(configs)} configurations",
        fontsize=12, y=1.01,
    )

    plt.tight_layout()
    output_path = args.output
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved plot → {output_path}")

    # ------------------------------------------------------------------
    # 4b. Variance shrinkage figure  (2 × 2 grid)
    # ------------------------------------------------------------------
    fig_v, axes_v = plt.subplots(2, 2, figsize=(16, 12))
    ax_all, ax_uniform, ax_bar, ax_heat = (
        axes_v[0, 0], axes_v[0, 1], axes_v[1, 0], axes_v[1, 1]
    )

    # ---- Panel A: all methods — actual shrinkage vs compression ratio ----
    for family, cfgs in families_seen.items():
        cmap  = family_colors[family]
        style = family_style[family]
        cfgs_s = sorted(cfgs, key=lambda x: x["compression_ratio"])
        n = len(cfgs_s)
        colors_f = [cmap(0.4 + 0.5 * i / max(n - 1, 1)) for i in range(n)]

        for i, cfg in enumerate(cfgs_s):
            ax_all.scatter(
                cfg["compression_ratio"], cfg["var_shrinkage"],
                color=colors_f[i], s=55, zorder=5,
            )

        xs = [c["compression_ratio"] for c in cfgs_s]
        ys = [c["var_shrinkage"]     for c in cfgs_s]
        combined = sorted(zip(xs, ys))
        ax_all.plot(
            [p[0] for p in combined], [p[1] for p in combined],
            color=cmap(0.65), label=family, alpha=0.55, **style,
        )

    # Theoretical 1/k reference curve
    cr_ref = np.array(sorted({c["compression_ratio"] for c in configs}))
    ax_all.plot(cr_ref, 1.0 / cr_ref, color="black", linestyle="--",
                linewidth=1.5, alpha=0.6, label="Theoretical 1/k (independent tokens)")

    ax_all.set_xscale("log", base=2)
    ax_all.set_yscale("log", base=2)
    ax_all.set_xlabel("Compression Ratio", fontsize=11)
    ax_all.set_ylabel("Variance Shrinkage Factor  (log scale)", fontsize=11)
    ax_all.set_title("A — All Methods: Actual Variance Shrinkage", fontsize=12, fontweight="bold")
    ax_all.legend(fontsize=9)
    ax_all.grid(True, alpha=0.3)

    # ---- Panel B: uniform k — actual vs theoretical, + relative shrinkage ----
    uniform_cfgs_v = sorted(
        [c for c in configs if c["family"] == "uniform" and c["k_nominal"] > 1],
        key=lambda x: x["k_nominal"],
    )
    ks_v   = [c["k_nominal"]               for c in uniform_cfgs_v]
    act_v  = [c["var_shrinkage"]           for c in uniform_cfgs_v]
    theo_v = [c["theoretical_var_shrinkage"] for c in uniform_cfgs_v]
    rel_v  = [c["relative_var_shrinkage"]  for c in uniform_cfgs_v]

    ax_uniform.plot(ks_v, act_v,  marker="o", linewidth=2, color="#2563EB",
                    label="Actual shrinkage")
    ax_uniform.plot(ks_v, theo_v, marker="x", linewidth=2, linestyle="--",
                    color="black", alpha=0.6, label="Theoretical 1/k (independent)")

    ax_uniform.set_xlabel("Averaging Window Size k", fontsize=11)
    ax_uniform.set_ylabel("Variance Shrinkage Factor", fontsize=11)
    ax_uniform.set_xscale("log", base=2)
    ax_uniform.set_yscale("log", base=2)
    ax_uniform.set_title("B — Uniform k: Actual vs. Theoretical Shrinkage", fontsize=12, fontweight="bold")
    ax_uniform.legend(fontsize=9, loc="upper right")
    ax_uniform.grid(True, alpha=0.3)

    # Second y-axis: relative shrinkage (actual / theoretical = 1 + (k-1)·ρ_mean)
    ax2 = ax_uniform.twinx()
    ax2.plot(ks_v, rel_v, marker="s", linewidth=2, linestyle="-.",
             color="#DC2626", alpha=0.8, label="Relative shrinkage (actual/theoretical)")
    ax2.axhline(y=1.0, color="#DC2626", linestyle=":", linewidth=1, alpha=0.4)
    ax2.set_ylabel("Relative Shrinkage  (>1 = positively correlated tokens)", fontsize=10,
                   color="#DC2626")
    ax2.tick_params(axis="y", labelcolor="#DC2626")
    ax2.legend(fontsize=9, loc="lower right")

    # ---- Panel C: relative shrinkage by family at matched k values ----
    pivot_k = [k for k in [2, 4, 8, 16] if k <= args.uniform_k_max]
    family_order = ["uniform", "dynamic", "overlapping", "weighted"]
    family_rel: Dict[str, List[float]] = {f: [] for f in family_order}

    for k_target in pivot_k:
        for fam in family_order:
            candidates = [
                c for c in configs
                if c["family"] == fam
                and abs(c["mean_group_size"] - k_target) < 0.6
            ]
            if candidates:
                family_rel[fam].append(
                    float(np.mean([c["relative_var_shrinkage"] for c in candidates]))
                )
            else:
                family_rel[fam].append(None)

    x_pos   = np.arange(len(pivot_k))
    bar_w   = 0.18
    fam_bar_colors = {
        "uniform":     family_colors["uniform"](0.65),
        "dynamic":     family_colors["dynamic"](0.65),
        "overlapping": family_colors["overlapping"](0.65),
        "weighted":    family_colors["weighted"](0.65),
    }
    for fi, fam in enumerate(family_order):
        vals = family_rel[fam]
        ys_bar = [v if v is not None else 0 for v in vals]
        ax_bar.bar(
            x_pos + fi * bar_w, ys_bar, bar_w,
            label=fam, color=fam_bar_colors[fam], alpha=0.85,
        )

    ax_bar.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5,
                   label="1.0 = independent tokens")
    ax_bar.set_xticks(x_pos + bar_w * 1.5)
    ax_bar.set_xticklabels([f"k≈{k}" for k in pivot_k])
    ax_bar.set_xlabel("Approximate Window Size k", fontsize=11)
    ax_bar.set_ylabel("Relative Shrinkage  (actual / theoretical 1/k)", fontsize=11)
    ax_bar.set_title("C — Relative Shrinkage by Method Family\n"
                     "(>1 = tokens are correlated; averaging wastes less information than expected)",
                     fontsize=11, fontweight="bold")
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, axis="y", alpha=0.3)

    # ---- Panel D: overlapping windows heatmap (window × stride) ----
    overlap_cfgs = [c for c in configs if c["family"] == "overlapping"]
    if overlap_cfgs:
        # Extract unique windows and strides
        windows = sorted({int(round(c["k_nominal"])) for c in overlap_cfgs})
        # Stride is encoded in compression_ratio: stride = window / cr
        def _stride(c):
            return int(round(c["k_nominal"] / c["compression_ratio"]))

        strides = sorted({_stride(c) for c in overlap_cfgs})

        heat_data = np.full((len(windows), len(strides)), np.nan)
        for c in overlap_cfgs:
            wi = windows.index(int(round(c["k_nominal"])))
            si = strides.index(_stride(c))
            heat_data[wi, si] = c["var_shrinkage"]

        im = ax_heat.imshow(
            heat_data, aspect="auto", origin="lower",
            cmap="RdYlGn_r", interpolation="nearest",
        )
        fig_v.colorbar(im, ax=ax_heat, label="Variance Shrinkage Factor")
        ax_heat.set_xticks(range(len(strides)))
        ax_heat.set_xticklabels([str(s) for s in strides])
        ax_heat.set_yticks(range(len(windows)))
        ax_heat.set_yticklabels([str(w) for w in windows])
        ax_heat.set_xlabel("Stride", fontsize=11)
        ax_heat.set_ylabel("Window Size", fontsize=11)
        ax_heat.set_title("D — Overlapping Windows: Variance Shrinkage Heatmap\n"
                          "(diagonal = non-overlapping; off-diagonal = overlapping)",
                          fontsize=11, fontweight="bold")

        # Annotate cells
        for wi in range(len(windows)):
            for si in range(len(strides)):
                v = heat_data[wi, si]
                if not np.isnan(v):
                    ax_heat.text(si, wi, f"{v:.3f}", ha="center", va="center",
                                 fontsize=7.5,
                                 color="white" if v > heat_data[~np.isnan(heat_data)].mean() else "black")
    else:
        ax_heat.text(0.5, 0.5, "No overlapping configs generated",
                     ha="center", va="center", transform=ax_heat.transAxes)
        ax_heat.set_title("D — Overlapping Windows Heatmap", fontsize=11, fontweight="bold")

    # Figure title
    fig_v.suptitle(
        f"Token Averaging — Variance Shrinkage Analysis\n"
        f"Model: {model_label}  |  Layer: {layer_label}  |  "
        f"N={B} seqs × {L} tokens",
        fontsize=12, y=1.01,
    )

    plt.tight_layout()
    var_path = args.output_variance
    fig_v.savefig(var_path, dpi=150, bbox_inches="tight")
    plt.close(fig_v)
    log.info(f"Saved variance plot → {var_path}")

    # ------------------------------------------------------------------
    # 5. Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"{'Label':<38}  {'CR':>6}  {'Cos Ret':>8}  {'Var Shrink':>10}  {'Rel Shrink':>10}")
    print("-" * 88)
    for cfg in sorted(configs, key=lambda x: (x["family"], x["compression_ratio"])):
        print(
            f"{cfg['label']:<38}  "
            f"{cfg['compression_ratio']:>6.2f}  "
            f"{cfg['cosine_retention']:>8.4f}  "
            f"{cfg['var_shrinkage']:>10.4f}  "
            f"{cfg['relative_var_shrinkage']:>10.4f}"
        )
    print("=" * 88)
    print(f"\nInfo-loss plot  → {output_path}")
    print(f"Variance plot   → {var_path}")


if __name__ == "__main__":
    main()
