"""
Central configuration for token averaging research.
"""

import os
import torch

# Model configuration
MODEL_NAME = "EleutherAI/pythia-410m"
MODEL_REVISION = "main"

# Dataset configuration
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-v1"
DATASET_SPLIT = "train"

# Experiment configuration
K_MIN = 1
K_MAX = 128
NUM_SEQUENCES = 1000
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 8

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output paths
OUTPUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Analysis configuration
VARIANCE_COVARIANCE_MAX_DISTANCE = 20  # Maximum token distance for covariance analysis
ENTROPY_BINS = 50  # Number of bins for entropy estimation
SPECTRAL_WINDOW_SIZE = 256  # Window size for FFT analysis
SVD_EXPLAINED_VARIANCE_THRESHOLD = 0.95  # Threshold for effective rank

# Visualization configuration
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# Random seed for reproducibility
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Learnable averaging hyperparameters
# ---------------------------------------------------------------------------
LEARNABLE_LR = 1e-3
LEARNABLE_EPOCHS = 3
LEARNABLE_TRAIN_SEQUENCES = 500   # sequences used to train the scoring network
LEARNABLE_BATCH_SIZE = 16         # mini-batch size during LearnableAverager training

# ---------------------------------------------------------------------------
# LLM experiment hyperparameters (zero-shot / finetune / from-scratch)
# ---------------------------------------------------------------------------
EXPERIMENT_MODEL_ZEROSHOT  = "EleutherAI/pythia-410m"
EXPERIMENT_MODEL_FINETUNE  = "EleutherAI/pythia-410m"
EXPERIMENT_MODEL_SCRATCH   = "EleutherAI/pythia-70m"   # tokeniser only (OLM builds the weights)

# OLM architecture hyperparameters for the from-scratch experiment
# Sized to be roughly comparable to Pythia-70m (~70M parameters)
EXPERIMENT_OLM_D_MODEL      = 512   # hidden dimension
EXPERIMENT_OLM_N_HEADS      = 8    # attention heads
EXPERIMENT_OLM_N_LAYERS     = 6    # transformer layers
EXPERIMENT_TRAIN_STEPS     = 5_000
EXPERIMENT_FINETUNE_STEPS  = 2_000
EXPERIMENT_LR_SCRATCH      = 5e-4
EXPERIMENT_LR_FINETUNE     = 5e-5
EXPERIMENT_WARMUP_STEPS    = 200
EXPERIMENT_GRAD_CLIP       = 1.0
EXPERIMENT_K_VALUES        = [1, 2, 4, 8]   # k=1 is the no-averaging baseline
EXPERIMENT_EVAL_SEQUENCES  = 500
EXPERIMENT_TRAIN_SEQUENCES = 10_000
EXPERIMENT_BATCH_SIZE      = 4    # reduced vs analysis batch size to fit training GPU memory
EXPERIMENT_OUTPUT_DIR      = os.path.join(OUTPUT_DIR, "experiments")
