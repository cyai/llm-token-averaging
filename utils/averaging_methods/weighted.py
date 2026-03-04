"""
Weighted average: each position within a k-token window receives a static scalar
weight determined by a named scheme.  All weight vectors are L1-normalised so
they sum to 1, making them directly comparable to uniform averaging.

Available schemes
-----------------
uniform       equal weights [1/k, ..., 1/k]  — baseline
linear        linearly increasing toward the last token (recency bias)
exponential   exponential increase toward the last token
gaussian      bell curve centred in the window
triangular    ramp up then ramp down, peak at the centre
"""

import numpy as np
import torch
from typing import List


WEIGHT_SCHEMES: List[str] = [
    "uniform",
    "linear",
    "exponential",
    "gaussian",
    "triangular",
]


def compute_weights(k: int, scheme: str) -> np.ndarray:
    """
    Compute a normalised weight vector of length k for the given scheme.

    Args:
        k: window size
        scheme: one of WEIGHT_SCHEMES

    Returns:
        weights: float32 array of shape [k] that sums to 1
    """
    positions = np.arange(k, dtype=np.float64)

    if scheme == "uniform":
        weights = np.ones(k)

    elif scheme == "linear":
        # Positions go 1, 2, ..., k — more weight on later (more recent) tokens
        weights = positions + 1.0

    elif scheme == "exponential":
        # e^(α·i) where α is chosen so the ratio last/first ≈ 20 for any k
        alpha = np.log(20.0) / max(k - 1, 1)
        weights = np.exp(alpha * positions)

    elif scheme == "gaussian":
        center = (k - 1) / 2.0
        sigma = k / 4.0
        weights = np.exp(-0.5 * ((positions - center) / sigma) ** 2)

    elif scheme == "triangular":
        half = (k - 1) / 2.0
        weights = 1.0 - np.abs(positions - half) / (half + 1e-8)
        weights = np.maximum(weights, 0.0)

    else:
        raise ValueError(
            f"Unknown scheme '{scheme}'.  Choose from {WEIGHT_SCHEMES}."
        )

    total = weights.sum()
    if total == 0:
        weights = np.ones(k)
        total = k
    return (weights / total).astype(np.float32)


def apply_weighted_averaging(
    embeddings: torch.Tensor,
    k: int,
    weights: np.ndarray,
) -> torch.Tensor:
    """
    Apply weighted averaging with non-overlapping windows of size k.

    Args:
        embeddings: [batch, seq_len, dim]
        k: window size
        weights: [k] float array that sums to 1

    Returns:
        Averaged embeddings [batch, seq_len // k, dim]
    """
    if k == 1:
        return embeddings

    batch_size, seq_len, hidden_dim = embeddings.shape
    new_seq_len = seq_len // k
    truncated_len = new_seq_len * k

    emb = embeddings[:, :truncated_len, :]                          # [B, T', D]
    emb = emb.reshape(batch_size, new_seq_len, k, hidden_dim)       # [B, N, k, D]

    w = torch.tensor(weights, dtype=emb.dtype, device=emb.device)   # [k]
    w = w.view(1, 1, k, 1)                                          # [1, 1, k, 1]

    averaged = (emb * w).sum(dim=2)                                  # [B, N, D]
    return averaged


def compute_weight_entropy(weights: np.ndarray) -> float:
    """
    Shannon entropy of the weight distribution (nats).

    Lower entropy → more concentrated weighting (closer to selecting one token).
    Maximum entropy = ln(k) → uniform weights.
    """
    w = np.clip(weights, 1e-12, None)
    return float(-np.sum(w * np.log(w)))
