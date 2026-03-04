# Dynamic K Averaging

## Motivation

Uniform averaging compresses every k consecutive tokens into one with the same window size throughout the sequence. But natural language is not uniformly redundant: highly similar adjacent tokens (e.g. repeated phrases, determiners followed by nouns) may safely be merged into larger groups, while information-dense transitions (e.g. clause boundaries, topic shifts) should be kept in smaller groups.

Dynamic K averaging replaces the single fixed window size with a per-group schedule that can vary along the sequence.

---

## Formal Definition

Let the sequence be `x_1, x_2, ..., x_T`. A dynamic schedule is a partition of consecutive indices into non-overlapping, contiguous groups `G_1, G_2, ..., G_M` such that:

```
G_j = {i : s_j ≤ i < e_j}    with  s_{j+1} = e_j   (contiguous)
```

The compressed sequence is:

```
x̃_j = (1 / |G_j|) * Σ_{i ∈ G_j} x_i
```

The group sizes `|G_j|` are determined by a **scheduling strategy** rather than being fixed.

---

## Strategies

### 1. Alternating

The group sizes cycle through a fixed user-provided pattern.

**Example** — pattern `[2, 3]`:

```
Tokens: t1 t2 | t3 t4 t5 | t6 t7 | t8 t9 t10 | ...
Groups:  size=2  size=3    size=2   size=3
```

**Parameters:**
- `--pattern 2 3` (or any list of positive integers)

**Use case:** Controllable non-uniform compression with deterministic structure.

---

### 2. Random

Each group size is sampled independently and uniformly from `[k_min, k_max]`.

**Example** — `k_min=2, k_max=4`:

```
Tokens: t1 t2 t3 | t4 t5 | t6 t7 t8 t9 | t10 t11 | ...
Groups:  size=3    size=2   size=4         size=2
```

**Parameters:**
- `--k_min 2 --k_max 4`
- Random seed is fixed to `RANDOM_SEED` from `config.py` for reproducibility.

**Use case:** Studying average-case information loss under variable compression.

---

### 3. Adaptive

Group sizes are determined dynamically by the **cosine similarity** between adjacent token embeddings. Consecutive tokens that are highly similar (semantically redundant) are merged into a larger group; dissimilar tokens (marking semantic transitions) form smaller groups.

**Algorithm:**

```
pos = 0
while pos < T:
    k = k_min
    for ahead in [pos, pos+1, ..., pos + k_max - 2]:
        if cosine_sim(x[ahead], x[ahead+1]) >= threshold:
            k = ahead - pos + 2
        else:
            break
    k = clamp(k, k_min, k_max)
    G_next = (pos, pos + k)
    pos += k
```

**Parameters:**
- `--k_min 2 --k_max 6`
- `--high_sim_threshold 0.85` (cosine similarity above which tokens are considered redundant)

**Use case:** Content-aware compression — the core research question of whether semantically similar tokens can be safely merged.

---

## Key Metrics

In addition to the five standard analysis metrics, the dynamic runner reports:

| Metric | Description |
|--------|-------------|
| `avg_group_size` | Mean group size across the full schedule |
| `std_group_size` | Variability of group sizes |
| `min_group_size` | Smallest group formed |
| `max_group_size` | Largest group formed |

---

## Running

```bash
# All three strategies with defaults
python run_dynamic_analysis.py

# Only alternating, custom pattern
python run_dynamic_analysis.py --strategies alternating --pattern 2 3 4

# Adaptive with custom similarity threshold
python run_dynamic_analysis.py --strategies adaptive

# Quick smoke-test
python run_dynamic_analysis.py --num_sequences 50

# Via master script
python run_all_methods.py --methods dynamic
```

**CLI options:**

```
--strategies      Strategies to run: alternating random adaptive (default: all)
--pattern         Group-size pattern for alternating strategy (default: 2 3)
--num_sequences   Number of sequences to process
--output_dir      Output directory (default: outputs/dynamic/)
--device          cuda or cpu
```

---

## Output Structure

```
outputs/dynamic/
├── metrics/
│   ├── dynamic_metrics.csv      # Flat table: strategy × layer × metric
│   └── dynamic_metrics.json     # Nested results
├── dynamic_summary.md            # Markdown report
└── <strategy_label>/             # Per-strategy analysis plots
    └── <layer_name>/
        ├── covariance_decay.png
        ├── norm_distribution.png
        ├── power_spectrum_*.png
        └── singular_values_*.png
```

---

## Research Interpretation

| Adaptive behaviour | Interpretation |
|---|---|
| `avg_group_size` >> `k_min` | Model finds many redundant token pairs |
| `std_group_size` high | Compression is highly non-uniform |
| Info retention similar to uniform | Dynamic schedule adds no extra cost |
| Info retention better than uniform | Adaptive merging avoids destroying information at transitions |

If the adaptive strategy achieves similar or better information retention than a fixed `k = avg_group_size`, that is evidence that content-aware compression can be superior to blind uniform compression.
