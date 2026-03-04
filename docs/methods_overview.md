# Token Averaging Methods — Overview

This document describes all five averaging strategies implemented in this research framework, their relationships to each other, and guidance on which to run for different research goals.

---

## The Core Research Question

> Are language model embeddings redundant across small token windows?  
> If so, can we average k adjacent tokens into one without significant information loss — effectively multiplying the model's context length by k?

The five methods below probe this question from different angles:

| Method | Compression fixed? | Weights fixed? | Content-dependent? |
|--------|-------------------|----------------|-------------------|
| Uniform k | Yes (1/k ratio) | Yes (equal) | No |
| Dynamic K | No (variable) | Yes (equal) | Partially (adaptive) |
| Overlapping windows | Partial (stride/window) | Yes (equal) | No |
| Weighted average | Yes (1/k ratio) | No (shaped) | No |
| Learnable average | Yes (1/k ratio) | No (trained) | Yes |

---

## Method 1: Uniform K

**Script:** `run_all_analyses.py`  
**Output:** `outputs/`  
**Docs:** `docs/analysis.md`

The baseline. Sweeps `k ∈ {1, 2, 4, 8, ..., 128}`. Every window of k tokens is averaged with equal weights. Establishes the information-loss curve as a function of compression ratio.

**Run first.** All other methods are interpreted relative to this baseline.

---

## Method 2: Dynamic K

**Script:** `run_dynamic_analysis.py`  
**Output:** `outputs/dynamic/`  
**Docs:** `docs/methods/dynamic_k.md`

Replaces the fixed window size with a variable-size schedule along the sequence dimension. Three strategies:

- **Alternating** — deterministic cycling through a pattern (e.g. `[2, 3]`)
- **Random** — sizes sampled uniformly from `[k_min, k_max]`
- **Adaptive** — sizes driven by cosine similarity between adjacent embeddings

The adaptive strategy is the most scientifically interesting: it is the first step toward a system that decides _when_ to merge tokens based on content.

**Key question answered:** Does matching the compression to local redundancy patterns preserve more information than a fixed-k scheme with the same average compression?

---

## Method 3: Overlapping Windows

**Script:** `run_overlapping_analysis.py`  
**Output:** `outputs/overlapping/`  
**Docs:** `docs/methods/overlapping_windows.md`

Introduces a `stride < window_size` so that consecutive output tokens share source tokens. Implemented as `torch.nn.functional.avg_pool1d`.

Produces a 2-D heatmap of information retention as a function of `(window_size, stride)`. The `stride = window_size` column is identical to uniform k-averaging.

**Key question answered:** Does smoothly blending adjacent windows (overlap) reduce the information loss at token boundaries compared to hard non-overlapping cuts?

---

## Method 4: Weighted Average

**Script:** `run_weighted_analysis.py`  
**Output:** `outputs/weighted/`  
**Docs:** `docs/methods/weighted_averaging.md`

Keeps non-overlapping windows but replaces equal weights with five static schemes: uniform, linear, exponential, gaussian, triangular. All weights are normalised to sum to 1.

Also reports **weight entropy** — a measure of how selective the scheme is.

**Key question answered:** Is the information in a k-token window concentrated in a particular position (first, last, or centre), or is it diffuse?

---

## Method 5: Learnable Weighted Average

**Script:** `run_learnable_analysis.py`  
**Output:** `outputs/learnable/`  
**Docs:** `docs/methods/learnable_averaging.md`

Trains a minimal neural module (`LearnableAverager`: one `Linear(dim→1)` layer, ~1K parameters) to assign data-dependent weights within each window. Training objective: minimise MSE reconstruction loss in embedding space.

Produces per-k comparison tables (learned vs. uniform) and a plot of the mean learned weight profile.

**Key question answered:** What is the theoretically optimal static-weight approximation for this embedding space, and does a learned scheme outperform all hand-designed alternatives?

---

## Master Script

**Script:** `run_all_methods.py`

Runs all methods in sequence from a single command, loading the model and collecting embeddings once. Writes per-method subdirectories and a unified cross-method comparison report.

```bash
# All methods
python run_all_methods.py

# All except learnable (faster)
python run_all_methods.py --skip_learnable

# Select specific methods
python run_all_methods.py --methods uniform dynamic weighted
```

**Output:** `outputs/comparison_report.md` — side-by-side table comparing mean metrics across all methods.

---

## Recommended Experiment Order

1. **Start with uniform k** to establish baselines:
   ```bash
   python run_all_analyses.py --k_max 16 --num_sequences 100
   ```

2. **Run weighted averaging** (fast, no training):
   ```bash
   python run_weighted_analysis.py --num_sequences 100
   ```

3. **Run overlapping windows** (moderate speed):
   ```bash
   python run_overlapping_analysis.py --num_sequences 100
   ```

4. **Run dynamic k** (moderate speed):
   ```bash
   python run_dynamic_analysis.py --num_sequences 100
   ```

5. **Run learnable** with full data when ready:
   ```bash
   python run_learnable_analysis.py --device cuda
   ```

6. **Compare everything**:
   ```bash
   python run_all_methods.py --skip_learnable  # fast comparison
   python run_all_methods.py                    # full comparison with learnable
   ```

---

## Understanding the Output Metrics

All methods produce the same five core metrics for every `(method_config, layer)` combination:

| Metric | Good value | Bad value | What it measures |
|--------|-----------|-----------|-----------------|
| `variance_shrinkage_factor` | close to 1.0 | close to 0 | How much variance survives |
| `norm_shrinkage_factor` | close to 1.0 | close to 0 | How much norm survives |
| `info_retention_ratio` | close to 1.0 | close to 0 | Information preserved (entropy ratio) |
| `spectral_total_energy_loss_pct` | close to 0 | > 50% | Spectral energy destroyed |
| `rank_reduction` | close to 0 | large positive | Dimensionality lost |

The cross-method comparison in `outputs/comparison_report.md` shows these five metrics averaged across all layers and configurations for each method. A method that consistently scores closer to the "good" column is a better compression strategy.

---

## File Map

```
token-averaging/
├── run_all_analyses.py           ← Uniform k baseline
├── run_dynamic_analysis.py       ← Dynamic k
├── run_overlapping_analysis.py   ← Overlapping windows
├── run_weighted_analysis.py      ← Weighted average
├── run_learnable_analysis.py     ← Learnable weighted average
├── run_all_methods.py            ← Master script
├── config.py                     ← All hyperparameters
├── utils/
│   ├── averaging_methods/
│   │   ├── dynamic.py
│   │   ├── overlapping.py
│   │   ├── weighted.py
│   │   └── learnable.py
│   └── runner_utils.py           ← Shared boilerplate
├── analysis/                     ← Five analysis modules (reused by all methods)
└── docs/
    ├── analysis.md               ← Mathematical derivations
    ├── methods_overview.md       ← This file
    └── methods/
        ├── dynamic_k.md
        ├── overlapping_windows.md
        ├── weighted_averaging.md
        └── learnable_averaging.md
```
