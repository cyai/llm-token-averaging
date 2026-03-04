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
