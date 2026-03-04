"""
Dynamic K averaging: each group of tokens along the sequence dimension gets its own
group size, drawn from one of three schedules:

  alternating  – cycles through a fixed list, e.g. [2, 3, 2, 3, ...]
  random       – each group size is sampled uniformly from [k_min, k_max]
  adaptive     – group size is determined by cosine similarity between adjacent
                 tokens; highly similar (redundant) tokens are merged into larger
                 groups, dissimilar tokens into smaller groups.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


DYNAMIC_STRATEGIES = ["alternating", "random", "adaptive"]


def build_alternating_schedule(
    seq_len: int,
    pattern: List[int],
) -> List[Tuple[int, int]]:
    """
    Build non-overlapping groups by cycling through a fixed size pattern.

    Args:
        seq_len: total sequence length
        pattern: list of group sizes to cycle through, e.g. [2, 3]

    Returns:
        List of (start, end) index pairs (end is exclusive)
    """
    groups: List[Tuple[int, int]] = []
    pos = 0
    idx = 0
    while pos < seq_len:
        k = pattern[idx % len(pattern)]
        if pos + k > seq_len:
            break
        groups.append((pos, pos + k))
        pos += k
        idx += 1
    return groups


def build_random_schedule(
    seq_len: int,
    k_min: int,
    k_max: int,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """
    Build non-overlapping groups with sizes sampled from Uniform[k_min, k_max].

    Args:
        seq_len: total sequence length
        k_min: minimum group size (inclusive)
        k_max: maximum group size (inclusive)
        seed: random seed for reproducibility

    Returns:
        List of (start, end) index pairs
    """
    rng = np.random.RandomState(seed)
    groups: List[Tuple[int, int]] = []
    pos = 0
    while pos < seq_len:
        k = int(rng.randint(k_min, k_max + 1))
        if pos + k > seq_len:
            break
        groups.append((pos, pos + k))
        pos += k
    return groups


def build_adaptive_schedule(
    embeddings: torch.Tensor,
    k_min: int,
    k_max: int,
    high_sim_threshold: float = 0.85,
) -> List[Tuple[int, int]]:
    """
    Build non-overlapping groups whose sizes are driven by cosine similarity
    between adjacent tokens.  High similarity → merge up to k_max tokens;
    low similarity → keep to k_min tokens.

    The schedule is computed from the first sequence in the batch because
    group boundaries must be deterministic across the batch dimension.

    Args:
        embeddings: [batch, seq_len, dim]
        k_min: minimum group size
        k_max: maximum group size
        high_sim_threshold: cosine-similarity threshold above which tokens
                            are considered redundant (and merged further)

    Returns:
        List of (start, end) index pairs
    """
    # Use first sequence to compute the adaptive schedule
    seq = embeddings[0]  # [seq_len, dim]
    seq_len = seq.shape[0]

    # Pairwise cosine similarity between consecutive tokens
    normed = F.normalize(seq.float(), dim=-1)
    sims = (normed[:-1] * normed[1:]).sum(dim=-1).cpu().numpy()  # [seq_len - 1]

    groups: List[Tuple[int, int]] = []
    pos = 0
    while pos < seq_len:
        k = k_min
        # Try to extend the group as long as similarity stays high
        for ahead in range(pos, min(pos + k_max - 1, seq_len - 1)):
            if sims[ahead] >= high_sim_threshold:
                k = ahead - pos + 2  # +2 because we include both endpoints
            else:
                break
        k = max(k_min, min(k, k_max))
        if pos + k > seq_len:
            break
        groups.append((pos, pos + k))
        pos += k
    return groups


def apply_dynamic_averaging(
    embeddings: torch.Tensor,
    strategy: str = "alternating",
    pattern: Optional[List[int]] = None,
    k_min: int = 2,
    k_max: int = 4,
    seed: int = 42,
    high_sim_threshold: float = 0.85,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Apply dynamic k averaging to a batch of embeddings.

    Args:
        embeddings: [batch, seq_len, dim]
        strategy: one of "alternating", "random", "adaptive"
        pattern: group-size pattern for alternating strategy (default [2, 3])
        k_min: minimum group size for random/adaptive strategies
        k_max: maximum group size for random/adaptive strategies
        seed: random seed for reproducibility
        high_sim_threshold: cosine similarity threshold for adaptive strategy

    Returns:
        averaged: [batch, n_groups, dim]
        groups: list of (start, end) pairs that define the grouping
    """
    if pattern is None:
        pattern = [2, 3]

    batch_size, seq_len, hidden_dim = embeddings.shape

    if strategy == "alternating":
        groups = build_alternating_schedule(seq_len, pattern)
    elif strategy == "random":
        groups = build_random_schedule(seq_len, k_min, k_max, seed)
    elif strategy == "adaptive":
        groups = build_adaptive_schedule(embeddings, k_min, k_max, high_sim_threshold)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {DYNAMIC_STRATEGIES}."
        )

    if len(groups) == 0:
        raise ValueError(
            f"No complete groups could be formed for seq_len={seq_len} "
            f"with strategy='{strategy}'.  Try reducing k_min."
        )

    averaged_tokens = [
        embeddings[:, start:end, :].mean(dim=1)  # [batch, dim]
        for start, end in groups
    ]
    averaged = torch.stack(averaged_tokens, dim=1)  # [batch, n_groups, dim]
    return averaged, groups


def get_group_stats(groups: List[Tuple[int, int]]) -> dict:
    """Compute descriptive statistics about a group schedule."""
    sizes = [end - start for start, end in groups]
    return {
        "n_groups": len(groups),
        "avg_group_size": float(np.mean(sizes)),
        "std_group_size": float(np.std(sizes)),
        "min_group_size": int(np.min(sizes)),
        "max_group_size": int(np.max(sizes)),
    }
