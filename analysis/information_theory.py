"""
Information Theory Analysis: Measure information loss from token averaging.

Computes:
1. Entropy before and after averaging
2. Mutual information between original and averaged embeddings
3. Information retention ratio
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy as scipy_entropy

logger = logging.getLogger(__name__)


class InformationTheoryAnalysis:
    """Perform information theory analysis on token embeddings."""

    def __init__(self, n_bins: int = 50):
        """
        Initialize information theory analysis.

        Args:
            n_bins: Number of bins for entropy estimation
        """
        self.n_bins = n_bins
        self.results = {}

    def estimate_entropy_1d(self, data: np.ndarray, bins: int = None) -> float:
        """
        Estimate entropy of 1D data using histogram.

        Args:
            data: 1D array of data
            bins: Number of bins for histogram

        Returns:
            Entropy estimate in nats
        """
        if bins is None:
            bins = self.n_bins

        # Create histogram
        hist, _ = np.histogram(data, bins=bins, density=True)

        # Normalize to get probability distribution
        hist = hist / np.sum(hist)

        # Remove zero bins
        hist = hist[hist > 0]

        # Compute entropy
        ent = scipy_entropy(hist)

        return float(ent)

    def estimate_entropy(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Estimate entropy of embeddings.

        For high-dimensional data, we estimate:
        1. Per-dimension entropy (average)
        2. Norm entropy

        Args:
            embeddings: Embeddings array [batch_size, seq_len, hidden_dim]

        Returns:
            Dictionary with entropy estimates
        """
        # Flatten batch and sequence dimensions
        flattened = embeddings.reshape(-1, embeddings.shape[-1])

        # Estimate per-dimension entropy
        per_dim_entropies = []
        for dim in range(min(flattened.shape[1], 100)):  # Sample up to 100 dims
            ent = self.estimate_entropy_1d(flattened[:, dim])
            per_dim_entropies.append(ent)

        mean_per_dim_entropy = np.mean(per_dim_entropies)

        # Estimate entropy of norms
        norms = np.linalg.norm(flattened, axis=1)
        norm_entropy = self.estimate_entropy_1d(norms)

        return {
            "mean_per_dim_entropy": float(mean_per_dim_entropy),
            "norm_entropy": float(norm_entropy),
            "total_estimated_entropy": float(mean_per_dim_entropy * flattened.shape[1]),
        }

    def estimate_mutual_information(
        self,
        original_embeddings: np.ndarray,
        averaged_embeddings: np.ndarray,
        n_samples: int = 10000,
    ) -> Dict[str, float]:
        """
        Estimate mutual information between original and averaged embeddings.

        We sample dimensions and estimate MI for each, then average.

        Args:
            original_embeddings: Original embeddings [batch_size, seq_len, hidden_dim]
            averaged_embeddings: Averaged embeddings [batch_size, new_seq_len, hidden_dim]
            n_samples: Number of samples to use for MI estimation

        Returns:
            Dictionary with MI estimates
        """
        hidden_dim = original_embeddings.shape[-1]

        # Flatten embeddings
        orig_flat = original_embeddings.reshape(-1, hidden_dim)
        avg_flat = averaged_embeddings.reshape(-1, hidden_dim)

        # Sample if too many points
        if orig_flat.shape[0] > n_samples:
            indices = np.random.choice(orig_flat.shape[0], n_samples, replace=False)
            orig_sample = orig_flat[indices]
        else:
            orig_sample = orig_flat

        if avg_flat.shape[0] > n_samples:
            indices = np.random.choice(avg_flat.shape[0], n_samples, replace=False)
            avg_sample = avg_flat[indices]
        else:
            avg_sample = avg_flat

        # Estimate MI for a subset of dimensions
        n_dims_to_check = min(20, hidden_dim)
        mi_values = []

        for dim in range(n_dims_to_check):
            try:
                # Use sklearn's mutual_info_regression
                # We predict one dimension of averaged from one dimension of original
                mi = mutual_info_regression(
                    orig_sample[
                        : min(len(orig_sample), len(avg_sample)), dim : dim + 1
                    ],
                    avg_sample[: min(len(orig_sample), len(avg_sample)), dim],
                    random_state=42,
                )[0]
                mi_values.append(mi)
            except:
                continue

        mean_mi = np.mean(mi_values) if mi_values else 0.0

        return {
            "mean_mutual_information": float(mean_mi),
            "estimated_total_mi": float(mean_mi * hidden_dim),
        }

    def compute_information_retention(
        self,
        original_entropy: Dict[str, float],
        averaged_entropy: Dict[str, float],
        mutual_info: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute information retention metrics.

        Args:
            original_entropy: Entropy of original embeddings
            averaged_entropy: Entropy of averaged embeddings
            mutual_info: Mutual information between original and averaged

        Returns:
            Dictionary with retention metrics
        """
        # Information retention ratio (using per-dim entropy)
        orig_ent = original_entropy["mean_per_dim_entropy"]
        avg_ent = averaged_entropy["mean_per_dim_entropy"]

        retention_ratio = avg_ent / orig_ent if orig_ent > 0 else 0

        # Information loss
        information_loss = orig_ent - avg_ent

        return {
            "retention_ratio": float(retention_ratio),
            "information_loss": float(information_loss),
            "relative_information_loss": float(
                information_loss / orig_ent if orig_ent > 0 else 0
            ),
        }

    def plot_entropy_comparison(
        self,
        k_values: List[int],
        original_entropies: List[float],
        averaged_entropies: List[float],
        layer_name: str,
        output_path: str,
    ):
        """
        Plot entropy comparison across k values.

        Args:
            k_values: List of k values
            original_entropies: List of original entropies
            averaged_entropies: List of averaged entropies
            layer_name: Name of the layer
            output_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            k_values,
            original_entropies,
            marker="o",
            label="Original",
            linewidth=2,
            markersize=6,
        )
        ax.plot(
            k_values,
            averaged_entropies,
            marker="s",
            label="Averaged",
            linewidth=2,
            markersize=6,
        )

        ax.set_xlabel("Averaging Window Size (k)")
        ax.set_ylabel("Entropy (nats)")
        ax.set_title(f"Entropy Comparison - {layer_name}")
        ax.set_xscale("log", base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved entropy comparison plot to {output_path}")

    def plot_information_retention(
        self,
        k_values: List[int],
        retention_ratios: List[float],
        layer_name: str,
        output_path: str,
    ):
        """
        Plot information retention across k values.

        Args:
            k_values: List of k values
            retention_ratios: List of retention ratios
            layer_name: Name of the layer
            output_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            k_values,
            retention_ratios,
            marker="o",
            linewidth=2,
            markersize=6,
            color="green",
        )
        ax.axhline(
            y=1.0, color="red", linestyle="--", alpha=0.5, label="Perfect retention"
        )

        ax.set_xlabel("Averaging Window Size (k)")
        ax.set_ylabel("Information Retention Ratio")
        ax.set_title(f"Information Retention - {layer_name}")
        ax.set_xscale("log", base=2)
        ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved information retention plot to {output_path}")

    def analyze(
        self,
        original_embeddings: np.ndarray,
        averaged_embeddings: np.ndarray,
        layer_name: str,
        k: int,
        output_dir: str,
    ) -> Dict:
        """
        Run complete information theory analysis.

        Args:
            original_embeddings: Original embeddings
            averaged_embeddings: Averaged embeddings
            layer_name: Name of the layer
            k: Averaging window size
            output_dir: Directory to save outputs

        Returns:
            Dictionary with all analysis results
        """
        logger.info(f"Running information theory analysis for {layer_name}, k={k}")

        # Estimate entropies
        original_entropy = self.estimate_entropy(original_embeddings)
        averaged_entropy = self.estimate_entropy(averaged_embeddings)

        # Estimate mutual information
        mutual_info = self.estimate_mutual_information(
            original_embeddings, averaged_embeddings
        )

        # Compute retention metrics
        retention = self.compute_information_retention(
            original_entropy, averaged_entropy, mutual_info
        )

        results = {
            "layer": layer_name,
            "k": k,
            "original_entropy": original_entropy,
            "averaged_entropy": averaged_entropy,
            "mutual_information": mutual_info,
            "retention": retention,
        }

        return results
