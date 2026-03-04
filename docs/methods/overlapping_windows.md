# Overlapping Window Averaging

## Motivation

Standard uniform averaging uses non-overlapping windows: each token belongs to exactly one group, and the output sequence is a factor of `k` shorter. This is a hard downsampling operation that throws away the boundary context between adjacent windows.

Overlapping windows relax this by sliding the window with a stride smaller than the window size. Each output token now aggregates information from a context that partially overlaps with its neighbours. This is analogous to **1-D average pooling** and is conceptually related to convolutional smoothing.

The key trade-off is between **compression** (controlled by `stride`) and **context blending** (controlled by `window_size`).

---

## Formal Definition

Given a sequence `x_1, ..., x_T`, a window of size `k`, and a stride `s` (where `1 ≤ s ≤ k`):

```
x̃_j = (1/k) * Σ_{i=0}^{k-1}  x_{j·s + i}      for j = 0, 1, ..., L-1
```

where the output length is:

```
L = floor((T - k) / s) + 1
```

**Implementation** uses `torch.nn.functional.avg_pool1d` with `kernel_size=k` and `stride=s`, applied along the sequence dimension.

---

## Parameters

| Parameter | Symbol | Range | Interpretation |
|-----------|--------|-------|----------------|
| `window_size` | k | ≥ 1 | Tokens averaged per output token |
| `stride` | s | 1 ≤ s ≤ k | Step between consecutive windows |

**Derived quantities:**

| Quantity | Formula | Interpretation |
|----------|---------|----------------|
| Compression ratio | s / k | Fraction of original length retained |
| Output length | (T − k) / s + 1 | Number of output tokens |
| Overlap fraction | 1 − s/k | Fraction of each window shared with the next |

---

## Special Cases

| s | k | Behaviour |
|---|---|-----------|
| s = k | any | Non-overlapping (same as uniform k-averaging) |
| s = 1 | any | Maximum overlap; output ≈ original length; each token is smoothed |
| s = k/2 | even k | 50% overlap; output ≈ 2× length of non-overlapping |

---

## Sweep Grid

The runner sweeps a grid of `(window_size, stride)` pairs:

| window_size | strides tested |
|------------|----------------|
| 2 | 1, 2 |
| 4 | 1, 2, 4 |
| 8 | 1, 2, 4, 8 |

(when using defaults; fully customisable via CLI)

---

## Running

```bash
# Full sweep with defaults
python run_overlapping_analysis.py

# Custom windows and strides
python run_overlapping_analysis.py --window_sizes 4 8 --strides 1 2 4

# Quick smoke-test
python run_overlapping_analysis.py --num_sequences 50 --window_sizes 4

# Via master script
python run_all_methods.py --methods overlapping
```

**CLI options:**

```
--window_sizes    Window sizes to sweep (default: 2 4 8)
--strides         Stride values (default: auto-generated per window)
--num_sequences   Number of sequences to process
--output_dir      Output directory (default: outputs/overlapping/)
--device          cuda or cpu
```

---

## Output Structure

```
outputs/overlapping/
├── heatmaps/
│   ├── heatmap_embedding_variance_shrinkage_factor.png
│   ├── heatmap_embedding_info_retention_ratio.png
│   ├── heatmap_embedding_spectral_total_energy_loss_percentage.png
│   └── ...  (one per metric)
├── metrics/
│   ├── overlapping_metrics.csv   # One row per (window, stride, layer)
│   └── overlapping_metrics.json
└── overlapping_summary.md
```

The **heatmaps** are the signature output: a 2-D grid of (window_size × stride) coloured by metric value, making it immediately visible how compression level and overlap jointly affect information retention.

---

## Research Interpretation

### Effect of stride (compression level)

Lower stride → less compression → higher information retention, but shorter context savings. The heatmap reveals where the "cliff" is: the stride at which a given window size starts causing significant information loss.

### Effect of window size (smoothing radius)

Larger window → more averaging per output token → lower norm and variance, but potentially more information loss. However, large windows with large strides compress aggressively; large windows with small strides produce heavily smoothed but long sequences.

### Key question

Does a `window_size=4, stride=2` configuration (50% overlap) retain substantially more information than `window_size=4, stride=4` (no overlap, same compression ratio as uniform `k=4`)? If yes, that suggests overlapping averaging is worth the implementation cost for context extension.

### Comparison baseline

The `s = k` cells in the heatmap are identical to the results from `run_all_analyses.py` (uniform averaging), so the heatmap directly shows the incremental value of adding overlap.
