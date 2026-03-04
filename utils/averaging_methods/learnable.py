"""
Learnable weighted average: a small neural module trained to find optimal
per-token importance weights within each k-token window.

Architecture — LearnableAverager
---------------------------------
input  : [batch, seq_len, dim]
         ↓  reshape into non-overlapping windows of size k
         [batch, n_windows, k, dim]
         ↓  shared linear scoring head:  dim → 1
         [batch, n_windows, k, 1]
         ↓  softmax over the k dimension
         weights [batch, n_windows, k, 1]
         ↓  weighted sum
output : [batch, n_windows, dim]

Training objective
------------------
We train the averager jointly with a lightweight reconstruction decoder that
tries to recover all k original token embeddings from the single averaged
vector.  Minimising MSE reconstruction loss forces the averaged embedding to
retain as much information as possible from the k tokens.

This training runs entirely in embedding space — it never re-runs the base LM
(Pythia-410M) and works on the already-collected numpy embedding arrays.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class LearnableAverager(nn.Module):
    """
    Content-dependent averaging: a shared linear scorer assigns importance
    weights to each token within its k-token window.
    """

    def __init__(self, hidden_dim: int, k: int):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.scorer = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.zeros_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, dim]

        Returns:
            averaged : [batch, n_windows, dim]
            weights  : [batch, n_windows, k]   (attention weights, for inspection)
        """
        batch_size, seq_len, hidden_dim = x.shape
        n_windows = seq_len // self.k
        truncated_len = n_windows * self.k

        x_trunc = x[:, :truncated_len, :]                              # [B, N*k, D]
        x_windows = x_trunc.reshape(batch_size, n_windows, self.k, hidden_dim)

        scores = self.scorer(x_windows)                                # [B, N, k, 1]
        weights = F.softmax(scores, dim=2)                             # [B, N, k, 1]
        averaged = (x_windows * weights).sum(dim=2)                    # [B, N, D]

        return averaged, weights.squeeze(-1)                           # weights: [B, N, k]

    @torch.no_grad()
    def get_effective_weights(
        self, embeddings: np.ndarray, device: str = "cpu", batch_size: int = 64
    ) -> np.ndarray:
        """
        Compute the mean attention weight vector across all windows and sequences.

        Returns:
            mean_weights: [k] array showing average weight per window position
        """
        self.eval()
        all_weights: List[np.ndarray] = []
        t = torch.tensor(embeddings, dtype=torch.float32)

        for i in range(0, len(embeddings), batch_size):
            batch = t[i : i + batch_size].to(device)
            _, w = self(batch)          # w: [B, n_windows, k]
            all_weights.append(w.cpu().numpy())

        stacked = np.concatenate(all_weights, axis=0)  # [N_seq, n_windows, k]
        return stacked.mean(axis=(0, 1))               # [k]


class ReconstructionDecoder(nn.Module):
    """
    Decodes a single averaged embedding back to k original-token embeddings.
    Used only during training; discarded afterwards.
    """

    def __init__(self, hidden_dim: int, k: int):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.decoder = nn.Linear(hidden_dim, k * hidden_dim, bias=True)

    def forward(self, averaged: torch.Tensor) -> torch.Tensor:
        """
        Args:
            averaged: [batch, n_windows, dim]

        Returns:
            reconstructed: [batch, n_windows, k, dim]
        """
        batch_size, n_windows, hidden_dim = averaged.shape
        out = self.decoder(averaged)                       # [B, N, k*D]
        return out.view(batch_size, n_windows, self.k, hidden_dim)


def train_learnable_averager(
    embeddings: np.ndarray,
    k: int,
    hidden_dim: int,
    n_epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str = "cpu",
    logger=None,
) -> Tuple["LearnableAverager", List[float]]:
    """
    Train LearnableAverager + ReconstructionDecoder to minimise MSE reconstruction
    loss on the provided embedding data.

    Args:
        embeddings: [n_seq, seq_len, dim] numpy array (float32 or float64)
        k: window size
        hidden_dim: embedding dimension
        n_epochs: number of full passes over the data
        lr: learning rate for AdamW
        batch_size: mini-batch size
        device: torch device string
        logger: optional Python logger for progress messages

    Returns:
        averager      : trained LearnableAverager (eval mode, on CPU)
        loss_history  : list of per-epoch mean MSE losses
    """
    n_seq, seq_len, _ = embeddings.shape
    n_windows = seq_len // k

    if n_windows == 0:
        raise ValueError(
            f"seq_len={seq_len} is too short to form any window of size k={k}"
        )

    averager = LearnableAverager(hidden_dim, k).to(device)
    decoder = ReconstructionDecoder(hidden_dim, k).to(device)

    params = list(averager.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    loss_history: List[float] = []
    rng = np.random.RandomState(42)

    for epoch in range(n_epochs):
        indices = rng.permutation(n_seq)
        epoch_losses: List[float] = []

        for i in range(0, n_seq, batch_size):
            batch_idx = indices[i : i + batch_size]
            batch = torch.tensor(
                embeddings[batch_idx], dtype=torch.float32, device=device
            )

            averaged, _ = averager(batch)        # [B, N_win, D]
            reconstructed = decoder(averaged)    # [B, N_win, k, D]

            # Build the target: original tokens reshaped into windows
            truncated_len = n_windows * k
            target = (
                batch[:, :truncated_len, :]
                .reshape(batch.shape[0], n_windows, k, hidden_dim)
            )

            loss = F.mse_loss(reconstructed, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)

        if logger:
            logger.info(
                f"  [LearnableAverager k={k}] Epoch {epoch + 1}/{n_epochs},"
                f" MSE loss: {mean_loss:.6f}"
            )

    averager.eval().cpu()
    return averager, loss_history


def apply_trained_averager(
    embeddings: np.ndarray,
    averager: "LearnableAverager",
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Apply a trained LearnableAverager to a numpy embedding array.

    Args:
        embeddings: [n_seq, seq_len, dim]
        averager: trained LearnableAverager instance
        device: torch device string
        batch_size: inference batch size

    Returns:
        averaged_embeddings: [n_seq, n_windows, dim]
    """
    averager.eval()
    averager.to(device)
    all_averaged: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.tensor(
                embeddings[i : i + batch_size], dtype=torch.float32, device=device
            )
            averaged, _ = averager(batch)
            all_averaged.append(averaged.cpu().numpy())

    return np.concatenate(all_averaged, axis=0)
