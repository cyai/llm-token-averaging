"""
Embedding extraction and averaging utilities.
"""

import torch
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def extract_embeddings(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Extract embeddings from all layers of the model.

    Args:
        model: EmbeddingExtractorModel instance
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        device: Device to use

    Returns:
        Dictionary mapping layer names to embeddings
    """
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    embeddings = model.extract(input_ids, attention_mask)

    return embeddings


def apply_averaging(
    embeddings: torch.Tensor, k: int, attention_mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Apply token averaging with window size k.

    Implements: x̃_j = (1/k) * Σ_{i=(j-1)k+1}^{jk} x_i

    Args:
        embeddings: Token embeddings [batch_size, seq_len, hidden_dim]
        k: Averaging window size
        attention_mask: Optional mask to exclude padding tokens [batch_size, seq_len]

    Returns:
        Averaged embeddings [batch_size, seq_len//k, hidden_dim]
    """
    if k == 1:
        return embeddings

    batch_size, seq_len, hidden_dim = embeddings.shape

    # Calculate new sequence length
    new_seq_len = seq_len // k

    # Truncate to make sequence length divisible by k
    truncated_len = new_seq_len * k
    embeddings_truncated = embeddings[:, :truncated_len, :]

    # Reshape to [batch_size, new_seq_len, k, hidden_dim]
    embeddings_reshaped = embeddings_truncated.reshape(
        batch_size, new_seq_len, k, hidden_dim
    )

    # Handle attention mask if provided
    if attention_mask is not None:
        attention_mask_truncated = attention_mask[:, :truncated_len]
        attention_mask_reshaped = attention_mask_truncated.reshape(
            batch_size, new_seq_len, k
        )

        # Expand mask to match embedding dimensions
        mask_expanded = attention_mask_reshaped.unsqueeze(-1).float()

        # Compute weighted average (only over non-padded tokens)
        masked_embeddings = embeddings_reshaped * mask_expanded
        sum_embeddings = masked_embeddings.sum(dim=2)
        count_tokens = mask_expanded.sum(dim=2).clamp(min=1)  # Avoid division by zero

        averaged = sum_embeddings / count_tokens
    else:
        # Simple average over k tokens
        averaged = embeddings_reshaped.mean(dim=2)

    return averaged


def batch_apply_averaging(
    layer_embeddings: Dict[str, torch.Tensor],
    k: int,
    attention_mask: torch.Tensor = None,
) -> Dict[str, torch.Tensor]:
    """
    Apply averaging to embeddings from all layers.

    Args:
        layer_embeddings: Dictionary mapping layer names to embeddings
        k: Averaging window size
        attention_mask: Optional attention mask

    Returns:
        Dictionary mapping layer names to averaged embeddings
    """
    averaged_embeddings = {}

    for layer_name, embeddings in layer_embeddings.items():
        averaged_embeddings[layer_name] = apply_averaging(embeddings, k, attention_mask)

    return averaged_embeddings


def embeddings_to_numpy(
    layer_embeddings: Dict[str, torch.Tensor],
) -> Dict[str, np.ndarray]:
    """
    Convert embeddings from tensors to numpy arrays.

    Args:
        layer_embeddings: Dictionary mapping layer names to embeddings tensors

    Returns:
        Dictionary mapping layer names to numpy arrays
    """
    return {
        layer_name: embeddings.cpu().numpy()
        for layer_name, embeddings in layer_embeddings.items()
    }
