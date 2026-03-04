# Token Averaging Research Framework

A comprehensive research framework for analysing whether averaging adjacent tokens in a language model can extend the effective context length without significant information loss.

## Research Question

**Can we increase LLM context length by averaging adjacent tokens?**

If language embeddings are redundant across small windows, averaging k consecutive tokens into one halves (or further reduces) the effective sequence length while keeping the model architecture unchanged. This framework provides rigorous empirical analysis to test this hypothesis across five averaging strategies.

---

## Averaging Methods

Five strategies are implemented, each probing a different aspect of the compression trade-off:

| # | Method | Script | What it tests |
|---|--------|--------|---------------|
| 1 | **Uniform k** | `run_all_analyses.py` | Baseline: fixed window, equal weights, k = 1–128 |
| 2 | **Dynamic K** | `run_dynamic_analysis.py` | Variable window sizes along the sequence (alternating / random / adaptive) |
| 3 | **Overlapping windows** | `run_overlapping_analysis.py` | Sliding window with stride < window size |
| 4 | **Weighted average** | `run_weighted_analysis.py` | Shaped static weights (linear, exponential, gaussian, triangular) |
| 5 | **Learnable average** | `run_learnable_analysis.py` | Trained content-dependent weights (minimises reconstruction loss) |

All five methods share the same five analysis modules and produce comparable metrics, enabling direct cross-method comparison.

See [`docs/methods_overview.md`](docs/methods_overview.md) for a detailed description of every method and when to use each one.

---

## Quick Start

### Installation

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Smoke-test (all methods, tiny dataset, ~5 min)

```bash
python run_all_methods.py --num_sequences 50 --skip_learnable
```

### Run each method individually

```bash
# 1. Baseline uniform k (k = 1 to 128, 1000 sequences)
python run_all_analyses.py --k_max 128 --num_sequences 1000

# 2. Dynamic k
python run_dynamic_analysis.py --num_sequences 1000

# 3. Overlapping windows
python run_overlapping_analysis.py --num_sequences 1000

# 4. Weighted average
python run_weighted_analysis.py --num_sequences 1000

# 5. Learnable weighted average
python run_learnable_analysis.py --num_sequences 500 --device cuda
```

### Run everything at once

```bash
# All methods (embeddings collected once, shared across all methods)
python run_all_methods.py

# Skip learnable for a faster run
python run_all_methods.py --skip_learnable

# Select specific methods
python run_all_methods.py --methods uniform dynamic weighted
```

---

## Analysis Modules

Every averaging method runs the same five analyses, making results directly comparable across methods.

### 1. Variance Analysis

- Variance before and after compression
- Covariance between adjacent tokens at increasing distances
- Variance shrinkage factor vs. theoretical `1/k` prediction

### 2. Norm Shrinkage Analysis

- L2 norm distribution before and after
- Norm shrinkage factor
- LayerNorm behaviour simulation

### 3. Information Theory Analysis

- Entropy estimation using histogram binning
- Mutual information between original and compressed embeddings
- Information retention ratio: `H(averaged) / H(original)`

### 4. Spectral Analysis

- FFT power spectrum along the sequence dimension
- Low-frequency vs. high-frequency energy split
- Energy loss percentage (averaging acts as a low-pass filter)

### 5. Rank Analysis

- Singular Value Decomposition of the embedding matrix
- Effective rank (cumulative variance ≥ 0.95 threshold)
- Stable rank and rank reduction

---

## Command-Line Reference

### `run_all_analyses.py` — Uniform k baseline

```
--k_min INT            Minimum k value (default: 1)
--k_max INT            Maximum k value (default: 128)
--num_sequences INT    Sequences to process (default: 1000)
--output_dir PATH      Output directory (default: outputs/)
--device STR           cuda or cpu (default: auto)
```

### `run_dynamic_analysis.py`

```
--strategies STR ...   alternating random adaptive (default: all)
--pattern INT ...      Group-size pattern for alternating (default: 2 3)
--num_sequences INT
--output_dir PATH      (default: outputs/dynamic/)
--device STR
```

### `run_overlapping_analysis.py`

```
--window_sizes INT ... Window sizes to sweep (default: 2 4 8)
--strides INT ...      Strides to test (default: auto-generated)
--num_sequences INT
--output_dir PATH      (default: outputs/overlapping/)
--device STR
```

### `run_weighted_analysis.py`

```
--schemes STR ...      uniform linear exponential gaussian triangular (default: all)
--k_values INT ...     Window sizes (default: 2 4 8 16)
--num_sequences INT
--output_dir PATH      (default: outputs/weighted/)
--device STR
```

### `run_learnable_analysis.py`

```
--k_values INT ...     Window sizes to train (default: 2 4 8 16)
--n_epochs INT         Training epochs per k (default: 3)
--lr FLOAT             Learning rate (default: 0.001)
--num_sequences INT    (default: 500)
--output_dir PATH      (default: outputs/learnable/)
--device STR
```

### `run_all_methods.py` — Master script

```
--methods STR ...      Methods to run: uniform dynamic overlapping weighted learnable
--skip_learnable       Shortcut to exclude the learnable method
--num_sequences INT    (default: 1000)
--output_dir PATH      (default: outputs/)
--device STR
--uniform_k_max INT    Max k for uniform sweep (default: 16)
--learnable_epochs INT
--learnable_lr FLOAT
```

---

## Project Structure

```
token-averaging/
├── run_all_analyses.py           # Uniform k baseline
├── run_dynamic_analysis.py       # Dynamic k analysis
├── run_overlapping_analysis.py   # Overlapping windows analysis
├── run_weighted_analysis.py      # Weighted average analysis
├── run_learnable_analysis.py     # Learnable average analysis
├── run_all_methods.py            # Master orchestration script
├── config.py                     # Central configuration
├── requirements.txt
│
├── analysis/                     # Analysis modules (shared by all methods)
│   ├── variance_analysis.py
│   ├── norm_analysis.py
│   ├── information_theory.py
│   ├── spectral_analysis.py
│   └── rank_analysis.py
│
├── utils/
│   ├── model_loader.py           # Pythia-410M + forward hooks
│   ├── data_loader.py            # WikiText-103 streaming
│   ├── embedding_extractor.py    # Embedding extraction + uniform averaging
│   ├── visualization.py          # Plot helpers
│   ├── runner_utils.py           # Shared boilerplate for all runner scripts
│   └── averaging_methods/        # One module per new averaging method
│       ├── dynamic.py
│       ├── overlapping.py
│       ├── weighted.py
│       └── learnable.py
│
├── outputs/                      # Generated outputs (gitignored)
│   ├── uniform/
│   ├── dynamic/
│   ├── overlapping/
│   ├── weighted/
│   ├── learnable/
│   ├── comparison_report.md      # Cross-method comparison (master script)
│   ├── comparison_chart.png
│   └── all_methods_metrics.csv
│
└── docs/
    ├── analysis.md               # Mathematical derivations
    ├── methods_overview.md       # All methods: descriptions + guidance
    └── methods/
        ├── dynamic_k.md
        ├── overlapping_windows.md
        ├── weighted_averaging.md
        └── learnable_averaging.md
```

---

## Configuration

Edit `config.py` to change defaults. CLI flags override these values at runtime.

```python
# Model
MODEL_NAME = "EleutherAI/pythia-410m"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-v1"

# Experiment
K_MIN = 1
K_MAX = 128
NUM_SEQUENCES = 1000
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 8

# Analysis
VARIANCE_COVARIANCE_MAX_DISTANCE = 20
ENTROPY_BINS = 50
SPECTRAL_WINDOW_SIZE = 256
SVD_EXPLAINED_VARIANCE_THRESHOLD = 0.95

# Learnable method
LEARNABLE_LR = 1e-3
LEARNABLE_EPOCHS = 3
LEARNABLE_TRAIN_SEQUENCES = 500
LEARNABLE_BATCH_SIZE = 16
```

---

## Output Files

### Per-method outputs

Each method writes to its own subdirectory under `outputs/`:

```
outputs/<method>/
├── metrics/
│   ├── <method>_metrics.csv     # Flat table: config × layer × all metrics
│   └── <method>_metrics.json    # Nested full results
├── <method>_summary.md
└── ...                          # Method-specific plots
```

### Cross-method outputs (master script only)

```
outputs/
├── comparison_report.md         # Summary table: method × metric
├── comparison_chart.png         # Bar chart of mean metrics per method
└── all_methods_metrics.csv      # All rows from all methods combined
```

### Plots generated per `(config, layer)`

- `covariance_decay.png` — covariance vs. token distance
- `norm_distribution.png` — norm histogram before/after
- `power_spectrum_*.png` — frequency domain analysis
- `singular_values_*.png` — rank and dimensionality analysis
- `weight_profiles.png` — weight schemes grid (weighted method only)
- `loss_curve_k*.png` — training loss (learnable method only)
- `weight_profile_k*.png` — learned weight profile (learnable method only)
- `heatmap_*.png` — 2-D (window, stride) metric grid (overlapping method only)

---

## Understanding Key Metrics

| Metric | Ideal | Meaning |
|--------|-------|---------|
| `variance_shrinkage_factor` | ≈ 1.0 | Variance largely preserved |
| `norm_shrinkage_factor` | ≈ 1.0 | Norm largely preserved |
| `info_retention_ratio` | ≈ 1.0 | Most information survives |
| `spectral_total_energy_loss_pct` | ≈ 0 | Little energy destroyed |
| `rank_reduction` | ≈ 0 | Dimensionality intact |

**Interpreting retention ratio thresholds:**

- `> 0.9` — Averaging is likely viable; embeddings are highly redundant
- `0.7–0.9` — Moderate loss; may be acceptable depending on downstream task
- `< 0.7` — Significant information destroyed; context extension would hurt performance

**Layer-by-layer patterns:**

Early layers (embedding, layer_0–5) tend to encode syntactic, fine-grained information that is harder to compress. Later layers encode higher-level semantics and tend to be more redundant. If early layers show high loss but late layers show low loss, a selective strategy (only average in later layers) may be worth exploring.

---

## Recommended Workflow

```bash
# Step 1: verify setup with a quick smoke-test
python run_all_methods.py --num_sequences 50 --skip_learnable

# Step 2: run uniform baseline for reference
python run_all_analyses.py --k_max 16 --num_sequences 500

# Step 3: run fast methods with more data
python run_weighted_analysis.py --num_sequences 1000
python run_overlapping_analysis.py --num_sequences 1000
python run_dynamic_analysis.py --num_sequences 1000

# Step 4: run learnable (GPU recommended)
python run_learnable_analysis.py --device cuda --num_sequences 500

# Step 5: generate unified comparison
python run_all_methods.py --num_sequences 1000

# Step 6: examine results
cat outputs/comparison_report.md
open outputs/comparison_chart.png
```

---

## Troubleshooting

**Out of memory:**
- Reduce `--num_sequences`
- Reduce `MAX_SEQUENCE_LENGTH` in `config.py`
- Reduce `BATCH_SIZE` in `config.py`

**Slow on CPU:**
- Pass `--device cuda` if a GPU is available
- Reduce `--k_max` for the uniform baseline
- Use `--skip_learnable` when running `run_all_methods.py`

**Missing plots:**
- Check the log file in `outputs/logs/`
- Verify matplotlib is installed: `pip install matplotlib`

**Learnable training not converging:**
- Increase `--n_epochs`
- Decrease `--lr` (try `1e-4`)
- Increase `--num_sequences` to provide more training data

---

## Documentation

| File | Contents |
|------|----------|
| `docs/analysis.md` | Mathematical derivations for all five analysis types |
| `docs/methods_overview.md` | All five averaging methods: descriptions, comparisons, guidance |
| `docs/methods/dynamic_k.md` | Dynamic K strategies in detail |
| `docs/methods/overlapping_windows.md` | Overlapping windows and the (window, stride) sweep |
| `docs/methods/weighted_averaging.md` | Weight schemes, entropy, and research interpretation |
| `docs/methods/learnable_averaging.md` | Architecture, training, and interpreting learned weights |

---

## License

MIT — free to use and modify for research purposes.
