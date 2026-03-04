"""
Rank Analysis: Analyze rank and dimensionality of token embeddings.

Computes:
1. Singular Value Decomposition (SVD)
2. Effective rank (intrinsic dimensionality)
3. Explained variance by top singular values
4. Rank reduction from averaging
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple
from scipy.linalg import svd

logger = logging.getLogger(__name__)


class RankAnalysis:
    """Perform rank analysis on token embeddings."""

    def __init__(self, explained_variance_threshold: float = 0.95):
        """
        Initialize rank analysis.

        Args:
            explained_variance_threshold: Threshold for effective rank (e.g., 0.95 = 95% variance)
        """
        self.explained_variance_threshold = explained_variance_threshold
        self.results = {}

    def compute_svd(
        self, embeddings: np.ndarray, full_matrices: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Singular Value Decomposition of embeddings.

        Args:
            embeddings: Embeddings array [batch_size, seq_len, hidden_dim]
            full_matrices: Whether to compute full U and Vh matrices

        Returns:
            Tuple of (U, singular_values, Vh)
        """
        # Flatten batch dimension: [batch_size * seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = embeddings.shape
        flattened = embeddings.reshape(-1, hidden_dim)

        # Compute SVD
        try:
            U, s, Vh = svd(flattened, full_matrices=full_matrices)
            return U, s, Vh
        except Exception as e:
            logger.error(f"SVD computation failed: {e}")
            return np.array([]), np.array([]), np.array([])

    def compute_explained_variance(self, singular_values: np.ndarray) -> np.ndarray:
        """
        Compute explained variance ratio for each singular value.

        Args:
            singular_values: Array of singular values

        Returns:
            Cumulative explained variance ratio
        """
        # Compute variance from singular values
        variance = singular_values**2
        total_variance = np.sum(variance)

        # Compute explained variance ratio
        explained_var_ratio = variance / total_variance

        # Compute cumulative explained variance
        cumulative_explained_var = np.cumsum(explained_var_ratio)

        return cumulative_explained_var

    def estimate_effective_rank(
        self, singular_values: np.ndarray, threshold: float = None
    ) -> Dict[str, float]:
        """
        Estimate effective rank (intrinsic dimensionality).

        Args:
            singular_values: Array of singular values
            threshold: Explained variance threshold (default: self.explained_variance_threshold)

        Returns:
            Dictionary with rank metrics
        """
        if threshold is None:
            threshold = self.explained_variance_threshold

        if len(singular_values) == 0:
            return {"effective_rank": 0, "stable_rank": 0.0, "full_rank": 0}

        # Compute cumulative explained variance
        cumulative_var = self.compute_explained_variance(singular_values)

        # Find effective rank (number of components to reach threshold)
        effective_rank = np.searchsorted(cumulative_var, threshold) + 1

        # Compute stable rank (ratio of Frobenius norm to spectral norm)
        # stable_rank = (sum of squared singular values) / (max singular value)^2
        stable_rank = np.sum(singular_values**2) / (singular_values[0] ** 2)

        full_rank = len(singular_values)

        return {
            "effective_rank": int(effective_rank),
            "stable_rank": float(stable_rank),
            "full_rank": int(full_rank),
            "effective_rank_ratio": float(effective_rank / full_rank),
        }

    def analyze_rank_reduction(
        self, original_singular_values: np.ndarray, averaged_singular_values: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze rank reduction from averaging.

        Args:
            original_singular_values: Singular values of original embeddings
            averaged_singular_values: Singular values of averaged embeddings

        Returns:
            Dictionary with rank reduction metrics
        """
        original_rank = self.estimate_effective_rank(original_singular_values)
        averaged_rank = self.estimate_effective_rank(averaged_singular_values)

        rank_reduction = (
            original_rank["effective_rank"] - averaged_rank["effective_rank"]
        )
        rank_reduction_ratio = (
            rank_reduction / original_rank["effective_rank"]
            if original_rank["effective_rank"] > 0
            else 0
        )

        return {
            "original_effective_rank": original_rank["effective_rank"],
            "averaged_effective_rank": averaged_rank["effective_rank"],
            "rank_reduction": int(rank_reduction),
            "rank_reduction_ratio": float(rank_reduction_ratio),
            "original_stable_rank": original_rank["stable_rank"],
            "averaged_stable_rank": averaged_rank["stable_rank"],
        }

    def plot_singular_value_spectrum(
        self,
        singular_values: np.ndarray,
        layer_name: str,
        k: int,
        output_path: str,
        n_values: int = 100,
    ):
        """
        Plot singular value spectrum.

        Args:
            singular_values: Array of singular values
            layer_name: Name of the layer
            k: Averaging window size
            output_path: Path to save plot
            n_values: Number of singular values to plot
        """
        if len(singular_values) == 0:
            logger.warning("No singular values to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot singular values
        n_plot = min(n_values, len(singular_values))
        ax1.plot(
            range(1, n_plot + 1),
            singular_values[:n_plot],
            marker="o",
            markersize=3,
            linewidth=1,
        )
        ax1.set_xlabel("Index")
        ax1.set_ylabel("Singular Value")
        ax1.set_title(f"Singular Value Spectrum - {layer_name} (k={k})")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)

        # Plot cumulative explained variance
        cumulative_var = self.compute_explained_variance(singular_values)
        ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, linewidth=2)
        ax2.axhline(
            y=self.explained_variance_threshold,
            color="red",
            linestyle="--",
            label=f"{self.explained_variance_threshold*100}% threshold",
        )
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Explained Variance")
        ax2.set_title(f"Explained Variance - {layer_name} (k={k})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved singular value spectrum plot to {output_path}")

    def plot_rank_comparison(
        self,
        k_values: List[int],
        effective_ranks: List[int],
        layer_name: str,
        output_path: str,
    ):
        """
        Plot effective rank across k values.

        Args:
            k_values: List of k values
            effective_ranks: List of effective rank values
            layer_name: Name of the layer
            output_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(k_values, effective_ranks, marker="o", linewidth=2, markersize=6)
        ax.set_xlabel("Averaging Window Size (k)")
        ax.set_ylabel("Effective Rank")
        ax.set_title(f"Effective Rank vs k - {layer_name}")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved rank comparison plot to {output_path}")

    def plot_spectrum_comparison(
        self,
        original_sv: np.ndarray,
        averaged_sv: np.ndarray,
        layer_name: str,
        k: int,
        output_path: str,
        n_values: int = 50,
    ):
        """
        Plot comparison of singular value spectra.

        Args:
            original_sv: Original singular values
            averaged_sv: Averaged singular values
            layer_name: Name of the layer
            k: Averaging window size
            output_path: Path to save plot
            n_values: Number of values to plot
        """
        if len(original_sv) == 0 or len(averaged_sv) == 0:
            logger.warning("No singular values to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        n_plot = min(n_values, len(original_sv), len(averaged_sv))
        ax.plot(
            range(1, n_plot + 1),
            original_sv[:n_plot],
            label="Original",
            marker="o",
            markersize=4,
            linewidth=1.5,
            alpha=0.7,
        )
        ax.plot(
            range(1, n_plot + 1),
            averaged_sv[:n_plot],
            label=f"Averaged (k={k})",
            marker="s",
            markersize=4,
            linewidth=1.5,
            alpha=0.7,
        )

        ax.set_xlabel("Index")
        ax.set_ylabel("Singular Value")
        ax.set_title(f"Singular Value Comparison - {layer_name}")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved spectrum comparison plot to {output_path}")

    def analyze(
        self,
        original_embeddings: np.ndarray,
        averaged_embeddings: np.ndarray,
        layer_name: str,
        k: int,
        output_dir: str,
    ) -> Dict:
        """
        Run complete rank analysis.

        Args:
            original_embeddings: Original embeddings
            averaged_embeddings: Averaged embeddings
            layer_name: Name of the layer
            k: Averaging window size
            output_dir: Directory to save outputs

        Returns:
            Dictionary with all analysis results
        """
        logger.info(f"Running rank analysis for {layer_name}, k={k}")

        # Compute SVD
        _, original_sv, _ = self.compute_svd(original_embeddings)
        _, averaged_sv, _ = self.compute_svd(averaged_embeddings)

        # Estimate effective ranks
        original_rank_info = self.estimate_effective_rank(original_sv)
        averaged_rank_info = self.estimate_effective_rank(averaged_sv)

        # Analyze rank reduction
        rank_reduction = self.analyze_rank_reduction(original_sv, averaged_sv)

        # Plot results
        import os

        plot_dir = os.path.join(output_dir, f"k_{k}", layer_name)
        os.makedirs(plot_dir, exist_ok=True)

        self.plot_singular_value_spectrum(
            original_sv,
            layer_name,
            1,
            os.path.join(plot_dir, "singular_values_original.png"),
        )

        self.plot_singular_value_spectrum(
            averaged_sv,
            layer_name,
            k,
            os.path.join(plot_dir, "singular_values_averaged.png"),
        )

        self.plot_spectrum_comparison(
            original_sv,
            averaged_sv,
            layer_name,
            k,
            os.path.join(plot_dir, "singular_values_comparison.png"),
        )

        results = {
            "layer": layer_name,
            "k": k,
            "original_rank_info": original_rank_info,
            "averaged_rank_info": averaged_rank_info,
            "rank_reduction": rank_reduction,
            "original_singular_values": (
                original_sv.tolist() if len(original_sv) > 0 else []
            ),
            "averaged_singular_values": (
                averaged_sv.tolist() if len(averaged_sv) > 0 else []
            ),
        }

        return results
