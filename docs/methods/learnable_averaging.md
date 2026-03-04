# Learnable Weighted Average

## Motivation

All previous methods use pre-defined, content-agnostic compression schemes. The actual optimal weighting for a given window likely depends on the **content** of the tokens in that window: if one token is semantically pivotal, the network should learn to assign it most of the weight.

Learnable weighted averaging trains a small neural module to infer per-token importance weights from the embeddings themselves. This is a lightweight, data-driven approach to finding the best possible averaging strategy.

---

## Architecture: `LearnableAverager`

```
Input:   [batch, seq_len, dim]
         ↓ reshape into non-overlapping k-token windows
         [batch, n_windows, k, dim]
         ↓ shared linear scoring head: Linear(dim → 1)
         [batch, n_windows, k, 1]
         ↓ softmax over the k dimension
         weights [batch, n_windows, k, 1]
         ↓ weighted sum over k
Output:  [batch, n_windows, dim]
```

The scoring head is a **single linear layer with no hidden units** — just `dim` learnable parameters plus a bias. It is shared across all window positions and all sequences. This keeps the parameter count tiny (`dim + 1` parameters, e.g. 1025 for Pythia-410M) while still being content-dependent.

---

## Training Objective

The `LearnableAverager` is trained to **minimise MSE reconstruction loss**: given the averaged embedding, can a lightweight decoder recover all `k` original token embeddings?

```
Encoder: [batch, n_windows, k, dim] → [batch, n_windows, dim]
Decoder: [batch, n_windows, dim]    → [batch, n_windows, k, dim]  (Linear: dim → k*dim)
Loss:    MSE(reconstructed, original_windows)
```

This objective forces the averaged embedding to be a **maximally informative bottleneck**: the weighting must preserve as much of the original k-token information as possible in a single vector.

### Why reconstruction loss and not language modelling loss?

- Training on reconstruction loss requires no modification of the base LM — it runs entirely in embedding space on already-collected activations.
- It is fast (no forward pass through the LM required), making it practical to train per-layer or per-k.
- It directly optimises the compression quality criterion studied by the other analysis metrics (information retention, variance preservation).
- Language modelling loss would require plugging the averaged sequence back into the LM, which involves handling positional encodings and attention masks at non-uniform sequence lengths — a separate research problem.

---

## Training Setup

| Hyperparameter | Default | Config key |
|----------------|---------|------------|
| Learning rate | 1e-3 | `LEARNABLE_LR` |
| Epochs | 3 | `LEARNABLE_EPOCHS` |
| Batch size | 16 | `LEARNABLE_BATCH_SIZE` |
| Training sequences | 500 | `LEARNABLE_TRAIN_SEQUENCES` |
| Optimiser | AdamW, weight_decay=1e-4 | — |

### One averager per k, shared across all layers

For computational efficiency, a single `LearnableAverager` is trained on the concatenated embeddings from **all** layers. Since all transformer layers in Pythia-410M have the same hidden dimension (1024), this is straightforward.

The intuition: the optimal intra-window weighting depends on relative token similarity patterns, which are broadly consistent across layers. If you want layer-specific averagers, reduce `LEARNABLE_TRAIN_SEQUENCES` and run multiple training jobs.

---

## What Gets Measured

After training, the runner:

1. **Applies the trained averager** to every layer's embeddings and runs all 5 standard analysis modules (variance, norm, information theory, spectral, rank).

2. **Runs uniform averaging** at the same k as a baseline on the same embeddings.

3. **Produces a comparison table** (`metrics/comparison_k{k}.csv`) with columns:
   - `{metric}_learned` — metric value under learned averaging
   - `{metric}_uniform` — metric value under uniform averaging
   - `{metric}_delta` — difference (positive = learned is better)

4. **Plots the training loss curve** (`plots/loss_curve_k{k}.png`) to verify training convergence.

5. **Plots the learned weight profile** (`plots/weight_profile_k{k}.png`):
   - Bar chart showing mean attention weight per window position, averaged over all sequences and windows.
   - Dashed red line shows the uniform baseline `1/k`.
   - Comparison with the static weight schemes from the weighted averaging method reveals what shape the network discovered.

---

## Running

```bash
# All default k values
python run_learnable_analysis.py

# Specific k, fewer epochs (fast)
python run_learnable_analysis.py --k_values 2 4 --n_epochs 2

# Quick smoke-test (1 epoch, 50 sequences)
python run_learnable_analysis.py --k_values 2 --num_sequences 50 --n_epochs 1

# With GPU
python run_learnable_analysis.py --device cuda

# Via master script
python run_all_methods.py --methods learnable
python run_all_methods.py --skip_learnable   # exclude when time is limited
```

**CLI options:**

```
--k_values        Window sizes to train (default: 2 4 8 16)
--n_epochs        Training epochs per k (default: from config)
--lr              Learning rate (default: from config)
--num_sequences   Sequences for embedding collection (default: LEARNABLE_TRAIN_SEQUENCES)
--output_dir      Output directory (default: outputs/learnable/)
--device          cuda or cpu
```

---

## Output Structure

```
outputs/learnable/
├── plots/
│   ├── loss_curve_k2.png          # Training loss vs epoch
│   ├── loss_curve_k4.png
│   ├── weight_profile_k2.png      # Mean learned weights per position
│   ├── weight_profile_k4.png
│   └── ...
├── metrics/
│   ├── learnable_metrics.csv      # All rows: learned + uniform per (k, layer)
│   ├── learnable_metrics.json
│   ├── comparison_k2.csv          # Side-by-side learned vs uniform
│   ├── comparison_k4.csv
│   ├── comparison_k8.csv
│   └── comparison_all_k.csv       # Concatenation of all comparison tables
└── learnable_summary.md
```

---

## Research Interpretation

### If learned beats uniform on information retention

The content-dependent scoring head has found a non-trivial weighting that preserves more information than equal weights. Inspect the weight profile to understand what it learned (recency bias? centre focus?).

### If learned ≈ uniform

Uniform is already close to optimal for this embedding space. The representation is genuinely diffuse — no single token in a window is much more informative than any other.

### If learned is worse than uniform

A possible sign of underfitting (increase epochs) or that the reconstruction objective is misaligned with the downstream analysis metrics. Consider increasing training sequences or checking the loss curve for convergence.

### Comparing the learned profile to static schemes

The weight profile plot overlays the learned weights on a uniform baseline. If the learned profile closely resembles one of the static schemes (e.g. linear or exponential), that static scheme is a computationally free substitute. If the learned profile is unique, it suggests the data contains structure not captured by any simple parametric weight function.

### Across k values

Plot `delta_info_retention` vs `k` to see whether the benefit of learned averaging grows or shrinks with window size. For large k, the gap between learned and uniform may be larger because the uniform scheme's flat assumption becomes increasingly wrong.
