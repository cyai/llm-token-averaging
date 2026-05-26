"""
plot_probe_accuracy.py — Probing accuracy analysis across all averaging methods.

Three probes measure how much recoverable information survives averaging:

  1. Source Retrieval Accuracy (SRA) — no training, no sklearn
     For each averaged token, rank every original token in the sequence by
     cosine similarity.  Report what fraction of the k source tokens appear
     in the top-k positions.  Perfect = 1.0, random ≈ k / seq_len.

  2. Linear Reconstruction R² — sklearn LinearRegression
     Train a linear map from averaged embedding → each original token embedding.
     R² measures how much of the per-dimension variance is linearly decodable.
     Break it down by within-window position (first / middle / last).

  3. Cluster Probe Accuracy — KMeans + LogisticRegression
     K-means cluster original token embeddings.  A logistic probe trained on
     averaged embeddings tries to predict which cluster the original token
     belonged to.  Reported vs. majority-class baseline.

Output
------
  probe_accuracy_plot.png  — 2 × 3 figure

Usage
-----
python plot_probe_accuracy.py --synthetic                # fast, no model
python plot_probe_accuracy.py --num_sequences 50         # real pythia-410m
python plot_probe_accuracy.py --synthetic --n_clusters 20 --uniform_k_max 16
"""

import argparse
import logging
import warnings
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Group builders (copied from plot_info_loss.py for self-containment)
# ---------------------------------------------------------------------------

def uniform_groups(seq_len: int, k: int) -> List[Tuple[int, int]]:
    return [(i * k, (i + 1) * k) for i in range(seq_len // k)]


def overlapping_groups(seq_len: int, window: int, stride: int) -> List[Tuple[int, int]]:
    groups, pos = [], 0
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
    embeddings: np.ndarray, k_min: int, k_max: int, threshold: float = 0.85,
) -> List[Tuple[int, int]]:
    seq = embeddings[0]
    L   = seq.shape[0]
    norm = seq / (np.linalg.norm(seq, axis=-1, keepdims=True) + 1e-8)
    sims = (norm[:-1] * norm[1:]).sum(axis=-1)
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
# Probe 1 — Source Retrieval Accuracy
# ---------------------------------------------------------------------------

def source_retrieval_accuracy(
    original: np.ndarray,
    groups: List[Tuple[int, int]],
    max_seqs: int = 10,
) -> float:
    """
    For each averaged token (group mean), retrieve the top-k most similar
    tokens from the full sequence.  Report the fraction of true source tokens
    that are successfully retrieved.

    Args:
        original:  [B, seq_len, D]
        groups:    list of (start, end) pairs
        max_seqs:  how many sequences to sample (for speed)

    Returns:
        Scalar in [0, 1].  Perfect = 1.0.
        Random baseline ≈ k / seq_len (very small).
    """
    B, L, D = original.shape
    eps = 1e-8
    hits, total = 0, 0

    for b in range(min(B, max_seqs)):
        seq      = original[b]                                      # [L, D]
        seq_norm = seq / (np.linalg.norm(seq, axis=-1, keepdims=True) + eps)

        for start, end in groups:
            end = min(end, L)
            if end <= start:
                continue
            k    = end - start
            avg  = seq[start:end].mean(axis=0)                      # [D]
            avg_n = avg / (np.linalg.norm(avg) + eps)

            sims    = seq_norm @ avg_n                               # [L]
            top_idx = set(np.argpartition(sims, -k)[-k:])
            source  = set(range(start, end))
            hits   += len(top_idx & source)
            total  += k

    return hits / total if total > 0 else 0.0


def sra_random_baseline(seq_len: int, mean_k: float) -> float:
    """Expected SRA for a random (uniform) retriever."""
    return mean_k / seq_len


# ---------------------------------------------------------------------------
# Probe 2 — Linear Reconstruction R²
# ---------------------------------------------------------------------------

def linear_reconstruction_r2(
    original: np.ndarray,
    groups: List[Tuple[int, int]],
    positions: Optional[List[str]] = None,
    test_frac: float = 0.2,
    max_pairs: int = 4000,
) -> Dict[str, float]:
    """
    Train a linear regression from the group average to each within-window
    position.  Report R² per position label.

    positions: list of labels to report, chosen from
               ["first", "second", "middle", "penultimate", "last"]
               Default: ["first", "middle", "last"].

    Returns:
        dict mapping position label → R² (on held-out test set)
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    if positions is None:
        positions = ["first", "middle", "last"]

    B, L, D = original.shape

    # Build (avg_emb, [pos0_emb, pos_mid_emb, pos_last_emb]) pairs
    avg_list:  List[np.ndarray] = []
    orig_by_pos: Dict[str, List[np.ndarray]] = {p: [] for p in positions}

    for b in range(B):
        for start, end in groups:
            end = min(end, L)
            k = end - start
            if k < 2:
                continue
            chunk = original[b, start:end, :]    # [k, D]
            avg   = chunk.mean(axis=0)            # [D]
            avg_list.append(avg)

            pos_map = {
                "first":       chunk[0],
                "second":      chunk[min(1, k-1)],
                "middle":      chunk[k // 2],
                "penultimate": chunk[max(0, k-2)],
                "last":        chunk[-1],
            }
            for p in positions:
                orig_by_pos[p].append(pos_map[p])

            if len(avg_list) >= max_pairs:
                break
        if len(avg_list) >= max_pairs:
            break

    if len(avg_list) < 10:
        return {p: 0.0 for p in positions}

    X = np.stack(avg_list)  # [N, D]
    n_test = max(1, int(len(X) * test_frac))

    results = {}
    for p in positions:
        Y = np.stack(orig_by_pos[p])  # [N, D]

        X_train, X_test = X[n_test:], X[:n_test]
        Y_train, Y_test = Y[n_test:], Y[:n_test]

        # Ridge regression: avg_emb → original token emb
        reg = Ridge(alpha=1.0, fit_intercept=True)
        reg.fit(X_train, Y_train)
        Y_pred = reg.predict(X_test)

        r2 = float(r2_score(Y_test.ravel(), Y_pred.ravel()))
        results[p] = max(0.0, r2)

    return results


# ---------------------------------------------------------------------------
# Probe 3 — Cluster Probe Accuracy
# ---------------------------------------------------------------------------

def fit_cluster_labels(
    original: np.ndarray,
    n_clusters: int = 10,
    max_fit_tokens: int = 5000,
) -> object:
    """Fit a KMeans model on (a sample of) original token embeddings."""
    from sklearn.cluster import MiniBatchKMeans

    flat = original.reshape(-1, original.shape[-1])
    if flat.shape[0] > max_fit_tokens:
        idx = np.random.RandomState(42).choice(flat.shape[0], max_fit_tokens, replace=False)
        flat = flat[idx]

    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    km.fit(flat)
    return km


def cluster_probe_accuracy(
    original: np.ndarray,
    groups: List[Tuple[int, int]],
    kmeans,
    max_pairs: int = 4000,
    test_frac: float = 0.2,
) -> Tuple[float, float]:
    """
    Train a logistic regression probe on averaged embeddings to predict the
    K-means cluster of the first token in each group.

    Returns:
        (probe_accuracy, majority_baseline_accuracy)
    """
    from sklearn.linear_model import LogisticRegression

    B, L, D = original.shape
    X_list, y_list = [], []

    for b in range(B):
        for start, end in groups:
            end = min(end, L)
            if end <= start:
                continue
            avg   = original[b, start:end, :].mean(axis=0)
            label = int(kmeans.predict(original[b, start:start+1, :])[0])  # first token's cluster
            X_list.append(avg)
            y_list.append(label)
            if len(X_list) >= max_pairs:
                break
        if len(X_list) >= max_pairs:
            break

    if len(X_list) < 20:
        return 0.0, 0.0

    X = np.stack(X_list)
    y = np.array(y_list)

    n_test    = max(1, int(len(X) * test_frac))
    X_tr, X_te = X[n_test:], X[:n_test]
    y_tr, y_te = y[n_test:], y[:n_test]

    if len(np.unique(y_tr)) < 2:
        return 0.0, 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=42, solver="lbfgs")
        clf.fit(X_tr, y_tr)
    probe_acc = float((clf.predict(X_te) == y_te).mean())

    # Majority baseline
    counts   = np.bincount(y_tr)
    majority = float(counts.max() / counts.sum())
    return probe_acc, majority


# ---------------------------------------------------------------------------
# Synthetic / real embeddings
# ---------------------------------------------------------------------------

def make_synthetic_embeddings(num_sequences: int, seq_len: int, d_model: int = 512) -> np.ndarray:
    rng = np.random.RandomState(42)
    base = rng.randn(num_sequences, seq_len, d_model).astype(np.float32)
    smoothed = np.zeros_like(base)
    for t in range(seq_len):
        w = base[:, max(0, t-2):t+3, :]
        smoothed[:, t, :] = w.mean(axis=1)
    return smoothed


def load_real_embeddings(args) -> np.ndarray:
    from utils.model_loader import load_pythia_model
    from utils.data_loader  import get_data_iterator

    log.info(f"Loading model: {args.model}")
    model, tokenizer = load_pythia_model(args.model, device=args.device)

    all_embs = []
    collected = 0
    for batch in get_data_iterator(
        tokenizer,
        num_sequences=args.num_sequences,
        max_length=args.max_length,
        batch_size=args.batch_size,
        split="train",
    ):
        input_ids      = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        layer_outputs  = model.extract(input_ids, attention_mask)

        if args.layer == "last":
            keys   = [k for k in layer_outputs if k.startswith("layer_")]
            chosen = sorted(keys, key=lambda x: int(x.split("_")[1]))[-1]
        else:
            chosen = "embedding"

        emb = layer_outputs[chosen].cpu().float().numpy()
        all_embs.append(emb)
        collected += emb.shape[0]
        log.info(f"  {collected}/{args.num_sequences} sequences (layer={chosen})")
        if collected >= args.num_sequences:
            break

    model.remove_hooks()
    return np.concatenate(all_embs, axis=0)[:args.num_sequences]


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_configs(seq_len: int, embeddings: np.ndarray, args) -> List[dict]:
    uniform_ks = [k for k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
                  if k <= args.uniform_k_max and k <= seq_len // 2]

    configs = []

    for k in uniform_ks:
        g = uniform_groups(seq_len, k)
        if not g:
            continue
        mean_k = float(k)
        configs.append(dict(
            family="uniform", label=f"uniform k={k}",
            k_nominal=k, mean_k=mean_k,
            compression_ratio=float(k), groups=g,
        ))

    overlap_windows = [w for w in [2, 4, 8] if w <= args.uniform_k_max // 2]
    for w in overlap_windows:
        for s in [1, w // 2, w]:
            if s < 1 or s > w:
                continue
            g = overlapping_groups(seq_len, w, s)
            if not g:
                continue
            configs.append(dict(
                family="overlapping", label=f"overlap w={w} s={s}",
                k_nominal=w, mean_k=float(w),
                compression_ratio=w / s, groups=g,
            ))

    dynamic_raw = [
        dict(strategy="alternating", pattern=[2, 3]),
        dict(strategy="alternating", pattern=[2, 4]),
        dict(strategy="random",      k_min=2, k_max=4),
        dict(strategy="random",      k_min=2, k_max=8),
        dict(strategy="adaptive",    k_min=2, k_max=4),
        dict(strategy="adaptive",    k_min=2, k_max=8),
    ]
    for dcfg in dynamic_raw:
        st = dcfg["strategy"]
        if st == "alternating":
            p = dcfg["pattern"]
            g = alternating_groups(seq_len, p)
            lbl = f"dynamic alt {p}"
        elif st == "random":
            g = random_groups(seq_len, dcfg["k_min"], dcfg["k_max"])
            lbl = f"dynamic rnd [{dcfg['k_min']},{dcfg['k_max']}]"
        else:
            g = adaptive_groups(embeddings, dcfg["k_min"], dcfg["k_max"])
            lbl = f"dynamic adp [{dcfg['k_min']},{dcfg['k_max']}]"
        if not g:
            continue
        mean_k = float(np.mean([e - s for s, e in g]))
        configs.append(dict(
            family="dynamic", label=lbl,
            k_nominal=mean_k, mean_k=mean_k,
            compression_ratio=mean_k, groups=g,
        ))

    return configs


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

FAMILY_STYLE = {
    "uniform":     dict(marker="o", linestyle="-",  linewidth=2.2, zorder=5),
    "overlapping": dict(marker="^", linestyle="--", linewidth=1.6, zorder=4),
    "dynamic":     dict(marker="s", linestyle="-.", linewidth=1.6, zorder=4),
}
FAMILY_COLOR = {
    "uniform":     "#2563EB",
    "overlapping": "#EA580C",
    "dynamic":     "#16A34A",
}
FAMILY_CMAP = {
    "uniform":     cm.Blues,
    "overlapping": cm.Oranges,
    "dynamic":     cm.Greens,
}


def scatter_all_methods(ax, configs: List[dict], metric_key: str):
    """Scatter plot of a metric vs compression ratio, coloured by family."""
    families: Dict[str, List[dict]] = {}
    for c in configs:
        families.setdefault(c["family"], []).append(c)

    for fam, cfgs in families.items():
        cmap  = FAMILY_CMAP[fam]
        style = FAMILY_STYLE[fam]
        cfgs_s = sorted(cfgs, key=lambda x: x["compression_ratio"])
        n = len(cfgs_s)
        colors = [cmap(0.4 + 0.5 * i / max(n - 1, 1)) for i in range(n)]

        for i, cfg in enumerate(cfgs_s):
            ax.scatter(cfg["compression_ratio"], cfg[metric_key],
                       color=colors[i], s=55, zorder=5)

        xs = [c["compression_ratio"] for c in cfgs_s]
        ys = [c[metric_key]          for c in cfgs_s]
        combined = sorted(zip(xs, ys))
        ax.plot([p[0] for p in combined], [p[1] for p in combined],
                color=cmap(0.65), label=fam, alpha=0.55, **style)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="EleutherAI/pythia-410m")
    p.add_argument("--num_sequences",  type=int, default=30)
    p.add_argument("--max_length",     type=int, default=256)
    p.add_argument("--batch_size",     type=int, default=8)
    p.add_argument("--layer",          default="embedding",
                   choices=["embedding", "last"])
    p.add_argument("--device",         default=None)
    p.add_argument("--synthetic",      action="store_true")
    p.add_argument("--uniform_k_max",  type=int, default=16)
    p.add_argument("--n_clusters",     type=int, default=10,
                   help="K-means clusters for probe 3 (default: 10)")
    p.add_argument("--output",         default="probe_accuracy_plot.png")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        from utils.model_loader import auto_device
        args.device = auto_device()

    log.info(f"Device: {args.device}  |  Sequences: {args.num_sequences}")

    # ------------------------------------------------------------------ #
    # 1. Embeddings                                                        #
    # ------------------------------------------------------------------ #
    if args.synthetic:
        log.info("Using synthetic embeddings")
        embeddings = make_synthetic_embeddings(args.num_sequences, args.max_length)
    else:
        embeddings = load_real_embeddings(args)

    B, L, D = embeddings.shape
    log.info(f"Embeddings: {embeddings.shape}")

    # ------------------------------------------------------------------ #
    # 2. Build configs                                                     #
    # ------------------------------------------------------------------ #
    configs = build_configs(L, embeddings, args)
    log.info(f"Configs: {len(configs)}")

    # ------------------------------------------------------------------ #
    # 3. Fit shared KMeans on original tokens (probe 3)                   #
    # ------------------------------------------------------------------ #
    log.info(f"Fitting KMeans (k={args.n_clusters}) on original tokens …")
    kmeans = fit_cluster_labels(embeddings, n_clusters=args.n_clusters)

    # ------------------------------------------------------------------ #
    # 4. Compute all three probes for every config                        #
    # ------------------------------------------------------------------ #
    log.info("Running probes on all configs …")
    for i, cfg in enumerate(configs):
        g = cfg["groups"]
        mean_k = cfg["mean_k"]

        # Probe 1 — SRA
        sra = source_retrieval_accuracy(embeddings, g)
        sra_base = sra_random_baseline(L, mean_k)
        cfg["sra"]           = sra
        cfg["sra_baseline"]  = sra_base
        cfg["sra_lift"]      = sra - sra_base   # how much above random

        # Probe 2 — Linear R² (only for groups with k >= 2)
        if mean_k >= 2:
            r2 = linear_reconstruction_r2(embeddings, g)
        else:
            r2 = {"first": 1.0, "middle": 1.0, "last": 1.0}
        cfg["r2_first"]  = r2.get("first",  0.0)
        cfg["r2_middle"] = r2.get("middle", 0.0)
        cfg["r2_last"]   = r2.get("last",   0.0)
        cfg["r2_mean"]   = float(np.mean(list(r2.values())))

        # Probe 3 — Cluster probe
        probe_acc, majority = cluster_probe_accuracy(embeddings, g, kmeans)
        cfg["cluster_probe_acc"] = probe_acc
        cfg["cluster_majority"]  = majority

        log.info(
            f"  [{i+1}/{len(configs)}] {cfg['label']:<32} "
            f"SRA={sra:.3f}  R²={cfg['r2_mean']:.3f}  "
            f"ClusterProbe={probe_acc:.3f}"
        )

    # ------------------------------------------------------------------ #
    # 5. Plot                                                              #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    ax_sra, ax_r2, ax_cluster  = axes[0]
    ax_uniform, ax_pos, ax_dyn = axes[1]

    model_label = "synthetic" if args.synthetic else args.model.split("/")[-1]

    # ---- Panel A: SRA — all methods ----------------------------------- #
    scatter_all_methods(ax_sra, configs, "sra")
    # Random SRA baseline curve
    cr_range = np.array(sorted({c["compression_ratio"] for c in configs if c["compression_ratio"] > 0}))
    ax_sra.plot(cr_range, [sra_random_baseline(L, k) for k in cr_range],
                color="grey", linestyle=":", linewidth=1.5, alpha=0.7,
                label="Random baseline (k/L)")
    ax_sra.set_xscale("log", base=2)
    ax_sra.set_xlabel("Compression Ratio", fontsize=11)
    ax_sra.set_ylabel("Source Retrieval Accuracy", fontsize=11)
    ax_sra.set_title("A — Source Retrieval Accuracy\n"
                     "(fraction of source tokens in top-k NN of averaged embedding)",
                     fontsize=11, fontweight="bold")
    ax_sra.legend(fontsize=9)
    ax_sra.grid(True, alpha=0.3)
    # Annotate uniform k points
    for cfg in configs:
        if cfg["family"] == "uniform" and int(cfg["k_nominal"]) in [2, 4, 8, 16]:
            ax_sra.annotate(
                f"k={int(cfg['k_nominal'])}",
                xy=(cfg["compression_ratio"], cfg["sra"]),
                xytext=(4, 6), textcoords="offset points",
                fontsize=8, color="#1D4ED8",
            )

    # ---- Panel B: Linear R² — all methods ----------------------------- #
    scatter_all_methods(ax_r2, configs, "r2_mean")
    ax_r2.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.4)
    ax_r2.set_xscale("log", base=2)
    ax_r2.set_xlabel("Compression Ratio", fontsize=11)
    ax_r2.set_ylabel("Linear R²  (mean over window positions)", fontsize=11)
    ax_r2.set_title("B — Linear Reconstruction R²\n"
                     "(how much variance in original tokens is linearly decodable)",
                     fontsize=11, fontweight="bold")
    ax_r2.legend(fontsize=9)
    ax_r2.grid(True, alpha=0.3)

    # ---- Panel C: Cluster probe — all methods ------------------------- #
    scatter_all_methods(ax_cluster, configs, "cluster_probe_acc")
    # Majority baseline (roughly constant)
    majority_base = np.mean([c["cluster_majority"] for c in configs])
    ax_cluster.axhline(y=majority_base, color="grey", linestyle=":",
                       linewidth=1.5, alpha=0.7, label=f"Majority baseline ({majority_base:.2f})")
    ax_cluster.axhline(y=1.0 / args.n_clusters, color="red", linestyle="--",
                       linewidth=1, alpha=0.5,
                       label=f"Uniform baseline (1/{args.n_clusters}={1/args.n_clusters:.2f})")
    ax_cluster.set_xscale("log", base=2)
    ax_cluster.set_xlabel("Compression Ratio", fontsize=11)
    ax_cluster.set_ylabel(f"Cluster Probe Accuracy  ({args.n_clusters} clusters)", fontsize=11)
    ax_cluster.set_title("C — Semantic Cluster Probe Accuracy\n"
                         "(logistic regression on averaged emb predicting K-means cluster)",
                         fontsize=11, fontweight="bold")
    ax_cluster.legend(fontsize=9)
    ax_cluster.grid(True, alpha=0.3)

    # ---- Panel D: Uniform k — all three probes on one axis ------------ #
    uniform_cfgs = sorted(
        [c for c in configs if c["family"] == "uniform"],
        key=lambda x: x["k_nominal"],
    )
    ks_u = [c["k_nominal"] for c in uniform_cfgs]

    # Normalize each metric to [0,1] for overlay
    def _norm(vals):
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return [0.5] * len(vals)
        return [(v - lo) / (hi - lo) for v in vals]

    sra_u     = [c["sra"]               for c in uniform_cfgs]
    r2_u      = [c["r2_mean"]           for c in uniform_cfgs]
    probe_u   = [c["cluster_probe_acc"] for c in uniform_cfgs]

    ax_uniform.plot(ks_u, _norm(sra_u),   marker="o", linewidth=2, color="#2563EB",
                    label="SRA (normalized)")
    ax_uniform.plot(ks_u, _norm(r2_u),    marker="s", linewidth=2, color="#7C3AED",
                    linestyle="--", label="Linear R² (normalized)")
    ax_uniform.plot(ks_u, _norm(probe_u), marker="^", linewidth=2, color="#DC2626",
                    linestyle="-.", label="Cluster probe (normalized)")
    ax_uniform.set_xscale("log", base=2)
    ax_uniform.set_xlabel("Averaging Window k (uniform)", fontsize=11)
    ax_uniform.set_ylabel("Normalized Probe Score  [0 = worst, 1 = best]", fontsize=11)
    ax_uniform.set_title("D — All Three Probes: Uniform k Baseline\n"
                         "(all normalized to the same scale for comparison)",
                         fontsize=11, fontweight="bold")
    ax_uniform.legend(fontsize=9)
    ax_uniform.grid(True, alpha=0.3)

    # ---- Panel E: Position bias (R² by position in window) ------------ #
    r2_first  = [c["r2_first"]  for c in uniform_cfgs]
    r2_middle = [c["r2_middle"] for c in uniform_cfgs]
    r2_last   = [c["r2_last"]   for c in uniform_cfgs]

    ax_pos.plot(ks_u, r2_first,  marker="o",  linewidth=2, color="#1D4ED8",
                label="First token in window")
    ax_pos.plot(ks_u, r2_middle, marker="s",  linewidth=2, color="#7C3AED",
                linestyle="--", label="Middle token in window")
    ax_pos.plot(ks_u, r2_last,   marker="^",  linewidth=2, color="#DC2626",
                linestyle="-.", label="Last token in window")
    ax_pos.set_xscale("log", base=2)
    ax_pos.set_xlabel("Averaging Window k (uniform)", fontsize=11)
    ax_pos.set_ylabel("Linear R²", fontsize=11)
    ax_pos.set_title("E — Reconstruction R² by Within-Window Position\n"
                     "(does the average favour first / middle / last token?)",
                     fontsize=11, fontweight="bold")
    ax_pos.legend(fontsize=9)
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_ylim(bottom=0)

    # ---- Panel F: Dynamic strategies — SRA bar chart ------------------ #
    dynamic_cfgs = sorted(
        [c for c in configs if c["family"] == "dynamic"],
        key=lambda x: x["compression_ratio"],
    )
    if dynamic_cfgs:
        labels_d = [c["label"].replace("dynamic ", "") for c in dynamic_cfgs]
        sra_d    = [c["sra"]  for c in dynamic_cfgs]
        r2_d     = [c["r2_mean"] for c in dynamic_cfgs]
        x_d      = np.arange(len(dynamic_cfgs))
        w        = 0.35
        ax_dyn.bar(x_d - w/2, sra_d, w, label="SRA",      color="#2563EB", alpha=0.8)
        ax_dyn.bar(x_d + w/2, r2_d,  w, label="Linear R²", color="#7C3AED", alpha=0.8)
        ax_dyn.set_xticks(x_d)
        ax_dyn.set_xticklabels(labels_d, rotation=30, ha="right", fontsize=8)
        ax_dyn.set_ylabel("Probe Score", fontsize=11)
        ax_dyn.set_title("F — Dynamic Strategies: SRA vs. R²\n"
                         "(all dynamic configs, sorted by compression ratio)",
                         fontsize=11, fontweight="bold")
        ax_dyn.legend(fontsize=9)
        ax_dyn.grid(True, axis="y", alpha=0.3)
        ax_dyn.set_ylim(0, 1)
    else:
        ax_dyn.text(0.5, 0.5, "No dynamic configs", ha="center", va="center",
                    transform=ax_dyn.transAxes)

    # ---- Suptitle ----------------------------------------------------- #
    fig.suptitle(
        f"Token Averaging — Probing Accuracy Analysis\n"
        f"Model: {model_label}  |  Layer: {args.layer}  |  "
        f"N={B} seqs × {L} tokens  |  {args.n_clusters} clusters  |  {len(configs)} configs",
        fontsize=12, y=1.01,
    )

    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {args.output}")

    # ------------------------------------------------------------------ #
    # 6. Summary table                                                     #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 85)
    print(f"{'Label':<34}  {'CR':>5}  {'SRA':>6}  {'R²':>6}  {'ClsProbe':>8}  {'ClsMaj':>6}")
    print("-" * 85)
    for cfg in sorted(configs, key=lambda x: (x["family"], x["compression_ratio"])):
        print(
            f"{cfg['label']:<34}  "
            f"{cfg['compression_ratio']:>5.2f}  "
            f"{cfg['sra']:>6.4f}  "
            f"{cfg['r2_mean']:>6.4f}  "
            f"{cfg['cluster_probe_acc']:>8.4f}  "
            f"{cfg['cluster_majority']:>6.4f}"
        )
    print("=" * 85)
    print(f"\nPlot saved → {args.output}")


if __name__ == "__main__":
    main()
