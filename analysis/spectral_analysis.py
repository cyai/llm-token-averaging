"""
Spectral Analysis: Analyze frequency domain properties of token embeddings.

Computes:
1. Power spectrum via FFT across sequence dimension
2. High-frequency vs low-frequency energy distribution
3. Energy loss from averaging (acts as low-pass filter)
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)


class SpectralAnalysis:
    """Perform spectral analysis on token embeddings."""

    def __init__(self, window_size: int = 256):
        """
        Initialize spectral analysis.

        Args:
            window_size: Window size for FFT analysis
        """
        self.window_size = window_size
        self.results = {}

    def compute_power_spectrum(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum via FFT across sequence dimension.

        Args:
            embeddings: Embeddings array [batch_size, seq_len, hidden_dim]

        Returns:
            Tuple of (frequencies, power_spectrum)
        """
        batch_size, seq_len, hidden_dim = embeddings.shape

        # Ensure sequence length is sufficient
        if seq_len < 8:
            logger.warning(f"Sequence length {seq_len} too short for spectral analysis")
            return np.array([]), np.array([])

        # Compute FFT for each dimension and average
        all_power_spectra = []

        for b in range(batch_size):
            for d in range(min(hidden_dim, 100)):  # Sample up to 100 dimensions
                # Get sequence for this batch and dimension
                sequence = embeddings[b, :, d]

                # Apply FFT
                fft_values = fft(sequence)

                # Compute power spectrum (magnitude squared)
                power = np.abs(fft_values) ** 2

                all_power_spectra.append(power)

        # Average across all sampled dimensions and batches
        mean_power_spectrum = np.mean(all_power_spectra, axis=0)

        # Get frequencies
        frequencies = fftfreq(seq_len)

        # Take only positive frequencies
        pos_freq_mask = frequencies >= 0
        frequencies = frequencies[pos_freq_mask]
        mean_power_spectrum = mean_power_spectrum[pos_freq_mask]

        return frequencies, mean_power_spectrum

    def measure_frequency_energy(
        self,
        frequencies: np.ndarray,
        power_spectrum: np.ndarray,
        high_freq_threshold: float = 0.3,
    ) -> Dict[str, float]:
        """
        Measure energy distribution across frequency bands.

        Args:
            frequencies: Array of frequencies
            power_spectrum: Power spectrum values
            high_freq_threshold: Threshold for high frequency (as fraction of max freq)

        Returns:
            Dictionary with energy distribution metrics
        """
        if len(frequencies) == 0:
            return {
                "total_energy": 0.0,
                "low_freq_energy": 0.0,
                "high_freq_energy": 0.0,
                "high_freq_percentage": 0.0,
            }

        total_energy = np.sum(power_spectrum)

        # Define high frequency cutoff
        max_freq = np.max(frequencies)
        high_freq_cutoff = high_freq_threshold * max_freq

        # Split into low and high frequency
        low_freq_mask = frequencies < high_freq_cutoff
        high_freq_mask = frequencies >= high_freq_cutoff

        low_freq_energy = np.sum(power_spectrum[low_freq_mask])
        high_freq_energy = np.sum(power_spectrum[high_freq_mask])

        high_freq_percentage = (
            (high_freq_energy / total_energy * 100) if total_energy > 0 else 0
        )

        return {
            "total_energy": float(total_energy),
            "low_freq_energy": float(low_freq_energy),
            "high_freq_energy": float(high_freq_energy),
            "high_freq_percentage": float(high_freq_percentage),
            "low_freq_percentage": float(100 - high_freq_percentage),
        }

    def analyze_energy_loss(
        self, original_energy: Dict[str, float], averaged_energy: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Analyze energy loss from averaging.

        Args:
            original_energy: Energy metrics for original embeddings
            averaged_energy: Energy metrics for averaged embeddings

        Returns:
            Dictionary with energy loss metrics
        """
        total_loss = original_energy["total_energy"] - averaged_energy["total_energy"]
        high_freq_loss = (
            original_energy["high_freq_energy"] - averaged_energy["high_freq_energy"]
        )
        low_freq_loss = (
            original_energy["low_freq_energy"] - averaged_energy["low_freq_energy"]
        )

        orig_total = original_energy["total_energy"]

        return {
            "total_energy_loss": float(total_loss),
            "high_freq_energy_loss": float(high_freq_loss),
            "low_freq_energy_loss": float(low_freq_loss),
            "total_energy_loss_percentage": float(
                total_loss / orig_total * 100 if orig_total > 0 else 0
            ),
            "high_freq_loss_percentage": float(
                high_freq_loss / orig_total * 100 if orig_total > 0 else 0
            ),
        }

    def plot_power_spectrum(
        self,
        frequencies: np.ndarray,
        power_spectrum: np.ndarray,
        layer_name: str,
        k: int,
        output_path: str,
    ):
        """
        Plot power spectrum.

        Args:
            frequencies: Array of frequencies
            power_spectrum: Power spectrum values
            layer_name: Name of the layer
            k: Averaging window size
            output_path: Path to save plot
        """
        if len(frequencies) == 0:
            logger.warning("No frequencies to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(frequencies, power_spectrum, linewidth=2)
        ax.set_xlabel("Frequency (normalized)")
        ax.set_ylabel("Power")
        ax.set_title(f"Power Spectrum - {layer_name} (k={k})")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved power spectrum plot to {output_path}")

    def plot_spectrum_comparison(
        self,
        original_freq: np.ndarray,
        original_power: np.ndarray,
        averaged_freq: np.ndarray,
        averaged_power: np.ndarray,
        layer_name: str,
        k: int,
        output_path: str,
    ):
        """
        Plot comparison of power spectra.

        Args:
            original_freq: Original frequencies
            original_power: Original power spectrum
            averaged_freq: Averaged frequencies
            averaged_power: Averaged power spectrum
            layer_name: Name of the layer
            k: Averaging window size
            output_path: Path to save plot
        """
        if len(original_freq) == 0 or len(averaged_freq) == 0:
            logger.warning("No frequencies to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(original_freq, original_power, label="Original", linewidth=2, alpha=0.7)
        ax.plot(
            averaged_freq,
            averaged_power,
            label=f"Averaged (k={k})",
            linewidth=2,
            alpha=0.7,
        )

        ax.set_xlabel("Frequency (normalized)")
        ax.set_ylabel("Power")
        ax.set_title(f"Power Spectrum Comparison - {layer_name}")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved spectrum comparison plot to {output_path}")

    def plot_energy_distribution(
        self,
        k_values: List[int],
        high_freq_percentages: List[float],
        layer_name: str,
        output_path: str,
    ):
        """
        Plot high-frequency energy percentage across k values.

        Args:
            k_values: List of k values
            high_freq_percentages: List of high-frequency energy percentages
            layer_name: Name of the layer
            output_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(k_values, high_freq_percentages, marker="o", linewidth=2, markersize=6)
        ax.set_xlabel("Averaging Window Size (k)")
        ax.set_ylabel("High-Frequency Energy (%)")
        ax.set_title(f"High-Frequency Energy vs k - {layer_name}")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved energy distribution plot to {output_path}")

    def analyze(
        self,
        original_embeddings: np.ndarray,
        averaged_embeddings: np.ndarray,
        layer_name: str,
        k: int,
        output_dir: str,
    ) -> Dict:
        """
        Run complete spectral analysis.

        Args:
            original_embeddings: Original embeddings
            averaged_embeddings: Averaged embeddings
            layer_name: Name of the layer
            k: Averaging window size
            output_dir: Directory to save outputs

        Returns:
            Dictionary with all analysis results
        """
        logger.info(f"Running spectral analysis for {layer_name}, k={k}")

        # Compute power spectra
        original_freq, original_power = self.compute_power_spectrum(original_embeddings)
        averaged_freq, averaged_power = self.compute_power_spectrum(averaged_embeddings)

        # Measure frequency energy
        original_energy = self.measure_frequency_energy(original_freq, original_power)
        averaged_energy = self.measure_frequency_energy(averaged_freq, averaged_power)

        # Analyze energy loss
        energy_loss = self.analyze_energy_loss(original_energy, averaged_energy)

        # Plot results
        import os

        plot_dir = os.path.join(output_dir, f"k_{k}", layer_name)
        os.makedirs(plot_dir, exist_ok=True)

        self.plot_power_spectrum(
            original_freq,
            original_power,
            layer_name,
            1,
            os.path.join(plot_dir, "power_spectrum_original.png"),
        )

        self.plot_power_spectrum(
            averaged_freq,
            averaged_power,
            layer_name,
            k,
            os.path.join(plot_dir, "power_spectrum_averaged.png"),
        )

        self.plot_spectrum_comparison(
            original_freq,
            original_power,
            averaged_freq,
            averaged_power,
            layer_name,
            k,
            os.path.join(plot_dir, "spectrum_comparison.png"),
        )

        results = {
            "layer": layer_name,
            "k": k,
            "original_energy": original_energy,
            "averaged_energy": averaged_energy,
            "energy_loss": energy_loss,
            "original_frequencies": (
                original_freq.tolist() if len(original_freq) > 0 else []
            ),
            "original_power": (
                original_power.tolist() if len(original_power) > 0 else []
            ),
            "averaged_frequencies": (
                averaged_freq.tolist() if len(averaged_freq) > 0 else []
            ),
            "averaged_power": (
                averaged_power.tolist() if len(averaged_power) > 0 else []
            ),
        }

        return results
