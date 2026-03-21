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
nohup python run_all_methods.py --num_sequences 1000 --uniform_k_max 128 > outputs/all_methods.log 2>&1 &

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

## LLM Performance Experiments (Phase 2)

Beyond embedding-space analysis, the framework provides three end-to-end
experiments that measure how real language models perform when their inputs
are token-averaged. All three share the same method × k grid and produce
perplexity on the WikiText-103 test set for direct comparison.

| Experiment | Script | Model | What it measures |
|---|---|---|---|
| **Zero-shot** | `experiments/zero_shot/run_zero_shot.py` | Pythia-410m (frozen) | Immediate effect of averaging on a pretrained model |
| **From scratch (OLM)** | `experiments/from_scratch/run_from_scratch.py` | OLM GPT (random init) | Whether a model can *learn* to work with averaged tokens |
| **Finetune** | `experiments/finetune/run_finetune.py` | Pythia-410m (finetuned) | How much recovery is possible with adaptation |
| **Compare** | `experiments/compare/run_compare.py` | — | Aggregates all three into one table + plots |

### Architecture: OLM (From-Scratch Experiment)

The from-scratch experiment uses **OpenLanguageModel (OLM)** to build the
model architecture. OLM's `Block`, `Repeat`, and `Residual` combinators
define a transparent GPT-style transformer (~70M parameters, comparable to
Pythia-70m) with explicit embedding and transformer sub-modules, making it
easy to insert the averaging step between embedding lookup and the first
transformer layer.

Default architecture (`config.py`):

```
d_model  = 512    # hidden dimension
n_heads  = 8      # attention heads
n_layers = 6      # transformer layers
context  = 512    # maximum sequence length
```

### Installation (Phase 2 only)

```bash
pip install openlanguagemodel    # OLM (from-scratch experiment)
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

### Running the experiments

**1. Zero-shot** — evaluate a frozen pretrained Pythia-410m across all method × k combos:

```bash
python experiments/zero_shot/run_zero_shot.py --device cuda

# Specific configs only
python experiments/zero_shot/run_zero_shot.py \
    --configs baseline_k1 uniform_k2 uniform_k4 weighted_exponential_k2 \
    --device cuda
```

**2. From scratch (OLM)** — train a fresh OLM model for each config, then evaluate:

```bash
# Representative subset (~11 configs, ~5 000 steps each)
python experiments/from_scratch/run_from_scratch.py --device cuda

# Specific configs
python experiments/from_scratch/run_from_scratch.py \
    --configs baseline_k1 uniform_k2 uniform_k4 learnable_k2 \
    --train_steps 5000 --lr 5e-4 --device cuda

# Custom OLM architecture size
python experiments/from_scratch/run_from_scratch.py \
    --d_model 256 --n_heads 4 --n_layers 4 \
    --train_steps 3000 --device cuda

# All configs (long — GPU strongly recommended)
python experiments/from_scratch/run_from_scratch.py --all_configs --device cuda
```

**3. Finetune** — finetune Pythia-410m with averaged inputs and record pre/post PPL:

```bash
python experiments/finetune/run_finetune.py --device cuda

# Specific configs
python experiments/finetune/run_finetune.py \
    --configs baseline_k1 uniform_k2 uniform_k4 \
    --finetune_steps 2000 --lr 5e-5 --device cuda
```

**4. Compare** — aggregate all result CSVs into a unified table and plots:

```bash
python experiments/compare/run_compare.py
```

Outputs are written to `outputs/experiments/compare/`:
- `comparison_table.md` — wide table: config × (zero-shot PPL, from-scratch PPL, finetune PPL)
- `ppl_by_k_{method}.png` — PPL vs k for each method, faceted by experiment
- `ppl_by_method_k{k}.png` — PPL vs method for each k value

### Experiment CLI Reference

#### `run_zero_shot.py`

```
--configs STR ...      Config names to evaluate (default: representative subset)
--all_configs          Evaluate every config in the registry
--model_name STR       HuggingFace model (default: EleutherAI/pythia-410m)
--eval_sequences INT   Test sequences for PPL (default: 500)
--batch_size INT       (default: 4)
--max_length INT       (default: 512)
--device STR           cuda or cpu
--output_dir PATH      (default: outputs/experiments/zero_shot/)
```

#### `run_from_scratch.py`

```
--configs STR ...      Config names to train (default: representative subset)
--all_configs          Train every config in the registry
--tokenizer_name STR   HuggingFace tokeniser — weights NOT loaded
                       (default: EleutherAI/pythia-70m for GPT-NeoX vocab)
--d_model INT          OLM hidden dimension (default: 512)
--n_heads INT          OLM attention heads  (default: 8)
--n_layers INT         OLM transformer layers (default: 6)
--train_steps INT      Steps per config (default: 5000)
--lr FLOAT             Learning rate (default: 5e-4)
--warmup_steps INT     Cosine-warmup steps (default: 200)
--grad_clip FLOAT      Gradient clip norm (default: 1.0)
--batch_size INT       (default: 4)
--max_length INT       (default: 512)
--train_sequences INT  Sequences per data-iterator pass (default: 10000)
--eval_sequences INT   Test sequences for PPL (default: 500)
--checkpoint_every INT Save checkpoint every N steps (default: 1000)
--device STR           cuda or cpu
--output_dir PATH      (default: outputs/experiments/from_scratch/)
--seed INT             (default: 42)
```

#### `run_finetune.py`

```
--configs STR ...      Config names to finetune (default: representative subset)
--all_configs
--model_name STR       HuggingFace model (default: EleutherAI/pythia-410m)
--finetune_steps INT   (default: 2000)
--lr FLOAT             (default: 5e-5)
--warmup_steps INT     (default: 200)
--batch_size INT       (default: 4)
--device STR
--output_dir PATH      (default: outputs/experiments/finetune/)
```

#### `run_compare.py`

```
--zero_shot_csv PATH   (default: outputs/experiments/zero_shot/results.csv)
--from_scratch_csv PATH
--finetune_csv PATH
--output_dir PATH      (default: outputs/experiments/compare/)
```

### Recommended experiment workflow

```bash
# Step 1 — quick zero-shot sanity check (fastest, no training)
python experiments/zero_shot/run_zero_shot.py \
    --configs baseline_k1 uniform_k2 uniform_k4 \
    --eval_sequences 100 --device cuda

# Step 2 — from-scratch with OLM on a small subset
python experiments/from_scratch/run_from_scratch.py \
    --configs baseline_k1 uniform_k2 uniform_k4 \
    --train_steps 2000 --device cuda

# Step 3 — finetune
python experiments/finetune/run_finetune.py \
    --configs baseline_k1 uniform_k2 uniform_k4 \
    --device cuda

# Step 4 — compare
python experiments/compare/run_compare.py

# Step 5 — full run (GPU, ~several hours)
python experiments/zero_shot/run_zero_shot.py --all_configs --device cuda
python experiments/from_scratch/run_from_scratch.py --all_configs --device cuda
python experiments/finetune/run_finetune.py --all_configs --device cuda
python experiments/compare/run_compare.py

# nohup background run
nohup python experiments/zero_shot/run_zero_shot.py --all_configs --device cuda > outputs/zero_shot.log 2>&1 &
nohup python experiments/from_scratch/run_from_scratch.py --all_configs --device cuda > outputs/from_scratch.log 2>&1 &
nohup python experiments/finetune/run_finetune.py --all_configs --device cuda > outputs/finetune.log 2>&1 &
nohup python experiments/compare/run_compare.py > outputs/compare.log 2>&1 &
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
├── experiments/                  # Phase 2: LLM performance evaluation
│   ├── shared/
│   │   ├── averaged_lm.py        # AveragedLanguageModel (Pythia wrapper)
│   │   ├── olm_model.py          # OLMTransformerBody + OLMAveragedLanguageModel
│   │   └── eval_utils.py         # Perplexity, result saving, logging helpers
│   ├── zero_shot/
│   │   └── run_zero_shot.py      # Frozen Pythia-410m evaluation
│   ├── from_scratch/
│   │   └── run_from_scratch.py   # OLM architecture trained from scratch
│   ├── finetune/
│   │   └── run_finetune.py       # Pythia-410m finetuned on averaged inputs
│   └── compare/
│       └── run_compare.py        # Aggregate + plot all three experiments
│
├── outputs/                      # Generated outputs (gitignored)
│   ├── uniform/
│   ├── dynamic/
│   ├── overlapping/
│   ├── weighted/
│   ├── learnable/
│   ├── comparison_report.md      # Cross-method comparison (master script)
│   ├── comparison_chart.png
│   ├── all_methods_metrics.csv
│   └── experiments/              # Phase 2 outputs
│       ├── zero_shot/results.csv
│       ├── from_scratch/results.csv
│       ├── finetune/results.csv
│       └── compare/
│           ├── comparison_table.md
│           └── ppl_by_*.png
│
└── docs/
    ├── analysis.md               # Mathematical derivations
    ├── results_analysis.md       # Blog-style results write-up
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
# ---------------------------------------------------------------------------
# Phase 1 — Embedding analysis
# ---------------------------------------------------------------------------
MODEL_NAME = "EleutherAI/pythia-410m"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-v1"

K_MIN = 1
K_MAX = 128
NUM_SEQUENCES = 1000
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 8

VARIANCE_COVARIANCE_MAX_DISTANCE = 20
ENTROPY_BINS = 50
SPECTRAL_WINDOW_SIZE = 256
SVD_EXPLAINED_VARIANCE_THRESHOLD = 0.95

# Learnable averaging method
LEARNABLE_LR = 1e-3
LEARNABLE_EPOCHS = 3
LEARNABLE_TRAIN_SEQUENCES = 500
LEARNABLE_BATCH_SIZE = 16

# ---------------------------------------------------------------------------
# Phase 2 — LLM performance experiments
# ---------------------------------------------------------------------------

# HuggingFace models used for zero-shot / finetune experiments
EXPERIMENT_MODEL_ZEROSHOT  = "EleutherAI/pythia-410m"
EXPERIMENT_MODEL_FINETUNE  = "EleutherAI/pythia-410m"
EXPERIMENT_MODEL_SCRATCH   = "EleutherAI/pythia-70m"   # tokeniser only

# OLM architecture (from-scratch experiment)
EXPERIMENT_OLM_D_MODEL  = 512   # hidden dimension  (~70M params total)
EXPERIMENT_OLM_N_HEADS  = 8     # attention heads
EXPERIMENT_OLM_N_LAYERS = 6     # transformer layers

# Training / evaluation
EXPERIMENT_TRAIN_STEPS     = 5_000
EXPERIMENT_FINETUNE_STEPS  = 2_000
EXPERIMENT_LR_SCRATCH      = 5e-4
EXPERIMENT_LR_FINETUNE     = 5e-5
EXPERIMENT_WARMUP_STEPS    = 200
EXPERIMENT_GRAD_CLIP       = 1.0
EXPERIMENT_K_VALUES        = [1, 2, 4, 8]
EXPERIMENT_EVAL_SEQUENCES  = 500
EXPERIMENT_TRAIN_SEQUENCES = 10_000
EXPERIMENT_BATCH_SIZE      = 4
EXPERIMENT_OUTPUT_DIR      = "outputs/experiments"
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

### Phase 1 — Embedding analysis

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

### Phase 2 — LLM experiments

**`ModuleNotFoundError: No module named 'olm'`:**
- Run `pip install openlanguagemodel` before the from-scratch experiment

**OLM LM split error** (`Expected OLM LM to have >=3 child blocks`):
- OLM's internal block structure may have changed in a newer release
- Inspect with `python3 -c "from olm.nn.blocks import LM; print(list(LM(1000,64,4,2,32).children()))"`
- Adjust the `OLMTransformerBody` split logic in `experiments/shared/olm_model.py` if needed

**From-scratch experiment runs out of memory:**
- Reduce `--d_model`, `--n_heads`, or `--n_layers` for a smaller model
- Reduce `--batch_size` (default 4)
- Reduce `--max_length` (default 512)

**PPL diverges / is `inf` during training:**
- Lower `--lr` (try `1e-4`)
- Increase `--warmup_steps` (try 500)
- Ensure `--grad_clip 1.0` is set (default)

**Finetune / zero-shot experiments slow to download model:**
- The Pythia-410m weights (~800 MB) are cached after the first download
- Set `HF_HOME` to a fast disk if the default cache location is slow

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
