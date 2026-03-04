"""
Visualization utilities for creating consistent plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def setup_plot_style(style: str = "seaborn-v0_8-darkgrid"):
    """
    Setup consistent plot style for all visualizations.

    Args:
        style: Matplotlib style to use
    """
    try:
        plt.style.use(style)
    except:
        # Fallback to default if style not available
        plt.style.use("default")
        logger.warning(f"Style '{style}' not available, using default")

    sns.set_palette("husl")

    # Set default font sizes
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 16


def save_figure(fig, filepath: str, dpi: int = 300, format: str = "png"):
    """
    Save figure to file with consistent settings.

    Args:
        fig: Matplotlib figure object
        filepath: Path to save figure
        dpi: Resolution in dots per inch
        format: File format (png, pdf, svg, etc.)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save figure
    fig.savefig(filepath, dpi=dpi, format=format, bbox_inches="tight")
    logger.info(f"Saved figure to {filepath}")

    plt.close(fig)


def create_comparison_plot(
    data: dict,
    xlabel: str,
    ylabel: str,
    title: str,
    filepath: Optional[str] = None,
    log_scale: bool = False,
    figsize: tuple = (10, 6),
):
    """
    Create a comparison line plot for multiple series.

    Args:
        data: Dictionary mapping series names to (x, y) tuples
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        filepath: Optional path to save figure
        log_scale: Whether to use log scale on axes
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, (x, y) in data.items():
        ax.plot(x, y, marker="o", label=label, linewidth=2, markersize=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    if filepath:
        save_figure(fig, filepath)

    return fig


def create_heatmap(
    data,
    xlabel: str,
    ylabel: str,
    title: str,
    filepath: Optional[str] = None,
    cmap: str = "viridis",
    figsize: tuple = (10, 8),
):
    """
    Create a heatmap visualization.

    Args:
        data: 2D array or DataFrame for heatmap
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        filepath: Optional path to save figure
        cmap: Colormap to use
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.colorbar(im, ax=ax)

    if filepath:
        save_figure(fig, filepath)

    return fig


def create_distribution_plot(
    data_dict: dict,
    xlabel: str,
    title: str,
    filepath: Optional[str] = None,
    bins: int = 50,
    figsize: tuple = (10, 6),
):
    """
    Create overlaid distribution plots (histograms).

    Args:
        data_dict: Dictionary mapping labels to data arrays
        xlabel: X-axis label
        title: Plot title
        filepath: Optional path to save figure
        bins: Number of histogram bins
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, data in data_dict.items():
        ax.hist(data, bins=bins, alpha=0.6, label=label, density=True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if filepath:
        save_figure(fig, filepath)

    return fig
