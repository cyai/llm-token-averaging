# Weighted Averaging

## Motivation

Uniform averaging assigns equal weight to every token in a window, implicitly assuming each token contributes equally to the compressed representation. This is almost certainly wrong: the last token in a window is likely the most informative (recency bias), or a central token may be a semantic anchor.

Weighted averaging replaces the flat `1/k` weight with a static, hand-designed weight vector `w = [w_1, ..., w_k]` (with `Σ w_i = 1`). Different weight shapes encode different priors about which tokens within a window carry the most information.

---

## Formal Definition

Given a window `x_j = [x_{jk+1}, ..., x_{(j+1)k}]` and a weight vector `w ∈ R^k` with `Σ w_i = 1`:

```
x̃_j = Σ_{i=1}^{k}  w_i * x_{jk+i}
```

The entire compressed sequence uses the **same weight vector** for every window.

---

## Weight Schemes

### Uniform (baseline)

```
w_i = 1/k   for all i
```

Identical to standard token averaging. Included as an internal baseline for direct comparison.

---

### Linear (recency bias)

```
w_i = i / Σ_{j=1}^{k} j
```

Weights increase linearly toward the last token. The final token in the window gets the most weight, reflecting the intuition that the most recent token is the most relevant summary.

**Weight profile (k=4):** `[0.10, 0.20, 0.30, 0.40]`

---

### Exponential

```
w_i = exp(α * i)   then normalised
```

where `α = ln(20) / (k-1)` (so the last token always has ~20× more weight than the first, regardless of k).

Stronger recency bias than linear. Approaches selecting only the last token as k grows.

**Weight profile (k=4):** `[0.04, 0.11, 0.27, 0.58]` (approx.)

---

### Gaussian

```
w_i = exp(-0.5 * ((i - center) / σ)²)   then normalised
```

where `center = (k-1)/2` and `σ = k/4`.

Bell curve centred in the window. The central token receives the most weight; tokens at the edges are down-weighted. Encoding the prior that the window's midpoint is the semantic centre.

**Weight profile (k=4):** roughly symmetric bell shape

---

### Triangular

```
w_i = 1 - |i - center| / (center + ε)   then clipped to [0, ∞) and normalised
```

Linear ramp up to centre, linear ramp down from centre. A simpler approximation to Gaussian.

**Weight profile (k=4):** `[0.17, 0.33, 0.33, 0.17]` (approx.)

---

## Weight Entropy

Each scheme has a **weight entropy** measuring how concentrated the weighting is:

```
H(w) = -Σ w_i * ln(w_i)      (nats)
```

| H = 0 | Degenerate: one token has all the weight (selection) |
| H = ln(k) | Maximum entropy: uniform weights |

The runner reports both raw entropy and entropy normalised by `ln(k)`.

| Scheme | Relative entropy (k=8) |
|--------|------------------------|
| uniform | 1.00 (maximum) |
| triangular | ~0.94 |
| gaussian | ~0.90 |
| linear | ~0.82 |
| exponential | ~0.55 |

Lower normalised entropy → the scheme is more selective → closer to choosing one representative token → more information loss but also a cleaner representation.

---

## Running

```bash
# All schemes, all default k values
python run_weighted_analysis.py

# Specific schemes
python run_weighted_analysis.py --schemes linear exponential

# Custom k values
python run_weighted_analysis.py --k_values 4 8 16 32

# Quick smoke-test
python run_weighted_analysis.py --num_sequences 50 --k_values 4

# Via master script
python run_all_methods.py --methods weighted
```

**CLI options:**

```
--schemes         Weight schemes to run (default: all 5)
--k_values        Window sizes (default: 2 4 8 16)
--num_sequences   Number of sequences to process
--output_dir      Output directory (default: outputs/weighted/)
--device          cuda or cpu
```

---

## Output Structure

```
outputs/weighted/
├── plots/
│   └── weight_profiles.png      # Grid plot: weight vector shape for each (k, scheme)
├── metrics/
│   ├── weighted_metrics.csv     # One row per (scheme, k, layer)
│   └── weighted_metrics.json
└── weighted_summary.md
```

The **weight profiles plot** is a grid of bar charts showing the normalised weight vector for each `(k, scheme)` combination — a quick visual reference for the shape of each scheme at each window size.

---

## Research Interpretation

### If uniform performs best

No static weighting scheme improves on equal weighting. The tokens within a window contribute roughly equally to the compressed representation, which is consistent with the information being diffuse.

### If linear or exponential performs best

The last token in a window is disproportionately informative. This would suggest that in a context-extension scenario, a sliding-window approach with a recency-biased aggregator is preferable to centred schemes.

### If gaussian or triangular performs best

The central token is the semantic anchor of the window. A centred aggregation strategy should be preferred.

### Weight entropy as a predictor

If information retention correlates positively with weight entropy across schemes, that suggests more uniform mixing (i.e. closer to standard averaging) is better. If it correlates negatively, selective single-token representations survive better.

### Baseline for learnable averaging

The weighted averaging schemes serve as interpretability references for the learnable method: after training `LearnableAverager`, the mean learned weight profile should be compared visually against these five shapes to understand what the network has learned.
