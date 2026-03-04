"""
Norm Shrinkage Analysis: Analyze how vector norms change with token averaging.

Computes:
1. Norm distribution before and after averaging
2. Mean and std of norms
3. Norm shrinkage factor
4. LayerNorm behavior analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List
from scipy import stats

logger = logging.getLogger(__name__)


class NormAnalysis:
    """Perform norm shrinkage analysis on token embeddings."""

    def __init__(self):
        """Initialize norm analysis."""
        self.results = {}

    def compute_norms(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute L2 norms of embeddings.

        Args:
            embeddings: Embeddings array [batch_size, seq_len, hidden_dim]

        Returns:
            Array of norms [batch_size * seq_len]
        """
        # Compute L2 norm along last dimension
        norms = np.linalg.norm(embeddings, axis=-1)
        # Flatten to 1D array
        norms_flat = norms.reshape(-1)
        return norms_flat

    def compute_norm_statistics(self, norms: np.ndarray) -> Dict[str, float]:
        """
        Compute statistics of norm distribution.

        Args:
            norms: Array of norms

        Returns:
            Dictionary with norm statistics
        """
        return {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "median": float(np.median(norms)),
            "min": float(np.min(norms)),
            "max": float(np.max(norms)),
            "q25": float(np.percentile(norms, 25)),
            "q75": float(np.percentile(norms, 75)),
        }

    def measure_norm_shrinkage(
        self, original_norms: np.ndarray, averaged_norms: np.ndarray
    ) -> Dict[str, float]:
        """
        Measure norm shrinkage from averaging.

        Args:
            original_norms: Original norms
            averaged_norms: Averaged norms

        Returns:
            Dictionary with shrinkage metrics
        """
        original_mean = np.mean(original_norms)
        averaged_mean = np.mean(averaged_norms)

        shrinkage_factor = averaged_mean / original_mean if original_mean > 0 else 0

        return {
            "original_mean_norm": float(original_mean),
            "averaged_mean_norm": float(averaged_mean),
            "shrinkage_factor": float(shrinkage_factor),
            "norm_reduction": float(1 - shrinkage_factor),
        }

    def analyze_layernorm_impact(
        self, embeddings: np.ndarray, epsilon: float = 1e-5
    ) -> Dict[str, float]:
        """
        Analyze how LayerNorm would affect the embeddings.

        LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
        We simulate with gamma=1, beta=0

        Args:
            embeddings: Embeddings array [batch_size, seq_len, hidden_dim]
            epsilon: Small constant for numerical stability

        Returns:
            Dictionary with LayerNorm statistics
        """
        # Compute mean and variance along last dimension
        mean = np.mean(embeddings, axis=-1, keepdims=True)
        variance = np.var(embeddings, axis=-1, keepdims=True)

        # Apply LayerNorm
        normalized = (embeddings - mean) / np.sqrt(variance + epsilon)

        # Compute norms after normalization
        norms_after_ln = self.compute_norms(normalized)

        return {
            "mean_norm_after_layernorm": float(np.mean(norms_after_ln)),
            "std_norm_after_layernorm": float(np.std(norms_after_ln)),
            "mean_rescale_factor": float(np.mean(np.sqrt(variance))),
        }

    def plot_norm_distributions(
        self,
        original_norms: np.ndarray,
        averaged_norms: np.ndarray,
        layer_name: str,
        k: int,
        output_path: str,
        bins: int = 50,
    ):
        """
        Plot histogram comparison of norm distributions.

        Args:
            original_norms: Original norms
            averaged_norms: Averaged norms
            layer_name: Name of the layer
            k: Averaging window size
            output_path: Path to save plot
            bins: Number of histogram bins
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histograms
        ax.hist(original_norms, bins=bins, alpha=0.6, label="Original", density=True)
        ax.hist(
            averaged_norms,
            bins=bins,
            alpha=0.6,
            label=f"Averaged (k={k})",
            density=True,
        )

        # Add vertical lines for means
        ax.axvline(
            np.mean(original_norms),
            color="blue",
            linestyle="--",
            label=f"Original mean: {np.mean(original_norms):.3f}",
        )
        ax.axvline(
            np.mean(averaged_norms),
            color="orange",
            linestyle="--",
            label=f"Averaged mean: {np.mean(averaged_norms):.3f}",
        )

        ax.set_xlabel("Norm (L2)")
        ax.set_ylabel("Density")
        ax.set_title(f"Norm Distribution - {layer_name} (k={k})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved norm distribution plot to {output_path}")

    def plot_norm_vs_k(
        self,
        k_values: List[int],
        mean_norms: List[float],
        layer_name: str,
        output_path: str,
    ):
        """
        Plot mean norm across different k values.

        Args:
            k_values: List of k values
            mean_norms: List of mean norm values
            layer_name: Name of the layer
            output_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(k_values, mean_norms, marker="o", linewidth=2, markersize=6)
        ax.set_xlabel("Averaging Window Size (k)")
        ax.set_ylabel("Mean Norm")
        ax.set_title(f"Norm Shrinkage vs k - {layer_name}")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

        # Add 1/sqrt(k) reference line (expected for uncorrelated embeddings)
        if len(k_values) > 0 and mean_norms[0] > 0:
            baseline = mean_norms[0]
            theoretical = [baseline / np.sqrt(k) for k in k_values]
            ax.plot(k_values, theoretical, "--", alpha=0.7, label="Theoretical 1/√k")
            ax.legend()

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved norm vs k plot to {output_path}")

    def analyze(
        self,
        original_embeddings: np.ndarray,
        averaged_embeddings: np.ndarray,
        layer_name: str,
        k: int,
        output_dir: str,
    ) -> Dict:
        """
        Run complete norm analysis.

        Args:
            original_embeddings: Original embeddings
            averaged_embeddings: Averaged embeddings
            layer_name: Name of the layer
            k: Averaging window size
            output_dir: Directory to save outputs

        Returns:
            Dictionary with all analysis results
        """
        logger.info(f"Running norm analysis for {layer_name}, k={k}")

        # Compute norms
        original_norms = self.compute_norms(original_embeddings)
        averaged_norms = self.compute_norms(averaged_embeddings)

        # Compute statistics
        original_stats = self.compute_norm_statistics(original_norms)
        averaged_stats = self.compute_norm_statistics(averaged_norms)

        # Measure shrinkage
        shrinkage = self.measure_norm_shrinkage(original_norms, averaged_norms)

        # Analyze LayerNorm impact
        ln_impact_original = self.analyze_layernorm_impact(original_embeddings)
        ln_impact_averaged = self.analyze_layernorm_impact(averaged_embeddings)

        # Plot distributions
        import os

        plot_dir = os.path.join(output_dir, f"k_{k}", layer_name)
        os.makedirs(plot_dir, exist_ok=True)

        self.plot_norm_distributions(
            original_norms,
            averaged_norms,
            layer_name,
            k,
            os.path.join(plot_dir, "norm_distribution.png"),
        )

        results = {
            "layer": layer_name,
            "k": k,
            "original_stats": original_stats,
            "averaged_stats": averaged_stats,
            "shrinkage": shrinkage,
            "layernorm_impact_original": ln_impact_original,
            "layernorm_impact_averaged": ln_impact_averaged,
        }

        return results
