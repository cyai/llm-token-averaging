"""
Variance Analysis: Analyze how variance changes with token averaging.

Computes:
1. Variance of embeddings before and after averaging
2. Covariance between adjacent tokens
3. Covariance decay with distance
4. Variance shrinkage factor
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Tuple, List
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class VarianceAnalysis:
    """Perform variance analysis on token embeddings."""

    def __init__(self, max_covariance_distance: int = 20):
        """
        Initialize variance analysis.

        Args:
            max_covariance_distance: Maximum token distance for covariance computation
        """
        self.max_covariance_distance = max_covariance_distance
        self.results = {}

    def compute_variance(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Compute variance statistics for embeddings.

        Args:
            embeddings: Embeddings array [batch_size, seq_len, hidden_dim]

        Returns:
            Dictionary with variance statistics
        """
        # Flatten batch and sequence dimensions
        flattened = embeddings.reshape(-1, embeddings.shape[-1])

        # Compute variance along each dimension
        variances = np.var(flattened, axis=0)

        return {
            "mean_variance": float(np.mean(variances)),
            "std_variance": float(np.std(variances)),
            "total_variance": float(np.sum(variances)),
            "max_variance": float(np.max(variances)),
            "min_variance": float(np.min(variances)),
        }

    def compute_covariance_matrix(
        self, embeddings: np.ndarray, max_distance: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute covariance between tokens at different distances.

        Args:
            embeddings: Embeddings array [batch_size, seq_len, hidden_dim]
            max_distance: Maximum distance to compute (default: self.max_covariance_distance)

        Returns:
            Tuple of (distances, covariances)
        """
        if max_distance is None:
            max_distance = self.max_covariance_distance

        batch_size, seq_len, hidden_dim = embeddings.shape

        # Compute covariance for each distance
        distances = []
        covariances = []

        for distance in range(max_distance + 1):
            if distance >= seq_len:
                break

            # Collect pairs of tokens at this distance
            all_pairs_x = []
            all_pairs_y = []

            for b in range(batch_size):
                for i in range(seq_len - distance):
                    x = embeddings[b, i, :]
                    y = embeddings[b, i + distance, :]
                    all_pairs_x.append(x)
                    all_pairs_y.append(y)

            if len(all_pairs_x) == 0:
                continue

            all_pairs_x = np.array(all_pairs_x)
            all_pairs_y = np.array(all_pairs_y)

            # Compute covariance (averaged across dimensions)
            cov_per_dim = []
            for dim in range(hidden_dim):
                cov = np.cov(all_pairs_x[:, dim], all_pairs_y[:, dim])[0, 1]
                cov_per_dim.append(cov)

            mean_cov = np.mean(cov_per_dim)

            distances.append(distance)
            covariances.append(mean_cov)

        return np.array(distances), np.array(covariances)

    def measure_shrinkage_factor(
        self, original_embeddings: np.ndarray, averaged_embeddings: np.ndarray
    ) -> Dict[str, float]:
        """
        Measure variance shrinkage from averaging.

        Args:
            original_embeddings: Original embeddings [batch_size, seq_len, hidden_dim]
            averaged_embeddings: Averaged embeddings [batch_size, new_seq_len, hidden_dim]

        Returns:
            Dictionary with shrinkage metrics
        """
        original_var = self.compute_variance(original_embeddings)
        averaged_var = self.compute_variance(averaged_embeddings)

        shrinkage_factor = averaged_var["mean_variance"] / original_var["mean_variance"]

        return {
            "original_variance": original_var["mean_variance"],
            "averaged_variance": averaged_var["mean_variance"],
            "shrinkage_factor": float(shrinkage_factor),
            "variance_reduction": float(1 - shrinkage_factor),
        }

    def plot_covariance_decay(
        self,
        distances: np.ndarray,
        covariances: np.ndarray,
        layer_name: str,
        k: int,
        output_path: str,
    ):
        """
        Plot covariance decay with distance.

        Args:
            distances: Array of distances
            covariances: Array of covariances
            layer_name: Name of the layer
            k: Averaging window size
            output_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(distances, covariances, marker="o", linewidth=2, markersize=6)
        ax.set_xlabel("Token Distance")
        ax.set_ylabel("Mean Covariance")
        ax.set_title(f"Covariance Decay - {layer_name} (k={k})")
        ax.grid(True, alpha=0.3)

        # Add exponential fit if data looks exponential
        if len(distances) > 3:
            try:
                # Fit exponential decay
                from scipy.optimize import curve_fit

                def exp_decay(x, a, b):
                    return a * np.exp(-b * x)

                popt, _ = curve_fit(
                    exp_decay, distances, covariances, p0=[covariances[0], 0.1]
                )
                fitted = exp_decay(distances, *popt)
                ax.plot(
                    distances,
                    fitted,
                    "--",
                    alpha=0.7,
                    label=f"Exp fit: {popt[0]:.3f}*exp(-{popt[1]:.3f}*x)",
                )
                ax.legend()
            except:
                pass

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved covariance decay plot to {output_path}")

    def plot_variance_comparison(
        self,
        k_values: List[int],
        variances: List[float],
        layer_name: str,
        output_path: str,
    ):
        """
        Plot variance across different k values.

        Args:
            k_values: List of k values
            variances: List of variance values
            layer_name: Name of the layer
            output_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(k_values, variances, marker="o", linewidth=2, markersize=6)
        ax.set_xlabel("Averaging Window Size (k)")
        ax.set_ylabel("Mean Variance")
        ax.set_title(f"Variance vs k - {layer_name}")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

        # Add 1/k reference line
        if len(k_values) > 0:
            baseline = variances[0] if variances[0] > 0 else 1.0
            theoretical = [baseline / k for k in k_values]
            ax.plot(k_values, theoretical, "--", alpha=0.7, label="Theoretical 1/k")
            ax.legend()

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved variance comparison plot to {output_path}")

    def analyze(
        self,
        original_embeddings: np.ndarray,
        averaged_embeddings: np.ndarray,
        layer_name: str,
        k: int,
        output_dir: str,
    ) -> Dict:
        """
        Run complete variance analysis.

        Args:
            original_embeddings: Original embeddings
            averaged_embeddings: Averaged embeddings
            layer_name: Name of the layer
            k: Averaging window size
            output_dir: Directory to save outputs

        Returns:
            Dictionary with all analysis results
        """
        logger.info(f"Running variance analysis for {layer_name}, k={k}")

        # Compute variance statistics
        original_var = self.compute_variance(original_embeddings)
        averaged_var = self.compute_variance(averaged_embeddings)

        # Compute shrinkage
        shrinkage = self.measure_shrinkage_factor(
            original_embeddings, averaged_embeddings
        )

        # Compute covariance matrix (only for original embeddings)
        distances, covariances = self.compute_covariance_matrix(original_embeddings)

        # Plot covariance decay
        import os

        plot_dir = os.path.join(output_dir, f"k_{k}", layer_name)
        os.makedirs(plot_dir, exist_ok=True)

        self.plot_covariance_decay(
            distances,
            covariances,
            layer_name,
            k,
            os.path.join(plot_dir, "covariance_decay.png"),
        )

        results = {
            "layer": layer_name,
            "k": k,
            "original_variance": original_var,
            "averaged_variance": averaged_var,
            "shrinkage": shrinkage,
            "covariance_distances": distances.tolist(),
            "covariance_values": covariances.tolist(),
        }

        return results
