"""
Overlapping window averaging: a fixed-size window slides along the sequence with
a configurable stride that can be smaller than the window (causing overlap).

  stride = window_size  →  no overlap, same as uniform k-averaging
  stride = 1            →  maximum overlap, output length ≈ seq_len
  compression_ratio     =  stride / window_size

Implementation delegates to torch.nn.functional.avg_pool1d, which is a
highly-optimised 1-D average pooling kernel with exact padding semantics.
"""

import torch
import torch.nn.functional as F


def apply_overlapping_averaging(
    embeddings: torch.Tensor,
    window_size: int,
    stride: int,
) -> torch.Tensor:
    """
    Apply average pooling with a sliding window of fixed size and given stride.

    Args:
        embeddings: [batch, seq_len, dim]
        window_size: number of tokens averaged in each window (k)
        stride: step between consecutive windows (1 ≤ stride ≤ window_size)

    Returns:
        Averaged embeddings [batch, output_len, dim]
        where output_len = (seq_len - window_size) // stride + 1
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if stride > window_size:
        raise ValueError(
            f"stride ({stride}) must be <= window_size ({window_size})"
        )

    # avg_pool1d expects [batch, channels, length] with channels = hidden_dim
    x = embeddings.permute(0, 2, 1)          # [batch, dim, seq_len]
    pooled = F.avg_pool1d(
        x,
        kernel_size=window_size,
        stride=stride,
    )                                         # [batch, dim, output_len]
    return pooled.permute(0, 2, 1)            # [batch, output_len, dim]


def get_output_length(seq_len: int, window_size: int, stride: int) -> int:
    """Compute the output sequence length after overlapping pooling."""
    return (seq_len - window_size) // stride + 1


def get_compression_ratio(window_size: int, stride: int) -> float:
    """
    Effective compression ratio.
    Returns stride / window_size (fraction of original length that is retained).
    A value of 1/window_size means non-overlapping (same as uniform averaging).
    """
    return stride / window_size
