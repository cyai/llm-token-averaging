"""
Averaging methods for token compression research.
"""

from .dynamic import apply_dynamic_averaging, get_group_stats, build_alternating_schedule, build_random_schedule, build_adaptive_schedule
from .overlapping import apply_overlapping_averaging, get_output_length, get_compression_ratio
from .weighted import apply_weighted_averaging, compute_weights, compute_weight_entropy, WEIGHT_SCHEMES
from .learnable import LearnableAverager, ReconstructionDecoder, train_learnable_averager, apply_trained_averager

__all__ = [
    # Dynamic
    "apply_dynamic_averaging",
    "get_group_stats",
    "build_alternating_schedule",
    "build_random_schedule",
    "build_adaptive_schedule",
    # Overlapping
    "apply_overlapping_averaging",
    "get_output_length",
    "get_compression_ratio",
    # Weighted
    "apply_weighted_averaging",
    "compute_weights",
    "compute_weight_entropy",
    "WEIGHT_SCHEMES",
    # Learnable
    "LearnableAverager",
    "ReconstructionDecoder",
    "train_learnable_averager",
    "apply_trained_averager",
]
