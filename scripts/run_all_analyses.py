"""
Token Averaging Research - Main Orchestration Script

This script runs all analyses across different k values (averaging window sizes)
on embeddings from Pythia-410M model using WikiText-103 dataset.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import (
    load_pythia_model,
    get_data_iterator,
    extract_embeddings,
    apply_averaging,
    setup_plot_style,
)
from analysis import (
    VarianceAnalysis,
    NormAnalysis,
    InformationTheoryAnalysis,
    SpectralAnalysis,
    RankAnalysis,
)


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return logging.getLogger(__name__)


def collect_embeddings(
    model,
    tokenizer,
    num_sequences: int,
    max_length: int,
    batch_size: int,
    device: str,
    logger: logging.Logger,
) -> Dict[str, List[np.ndarray]]:
    """
    Collect embeddings from all layers for a set of sequences.

    Returns:
        Dictionary mapping layer names to lists of embedding arrays
    """
    logger.info(f"Collecting embeddings from {num_sequences} sequences...")

    layer_embeddings = {}
    sequences_processed = 0

    data_iterator = get_data_iterator(
        tokenizer=tokenizer,
        num_sequences=num_sequences,
        max_length=max_length,
        batch_size=batch_size,
    )

    for batch in tqdm(
        data_iterator, desc="Processing sequences", total=num_sequences // batch_size
    ):
        if sequences_processed >= num_sequences:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Extract embeddings from all layers
        batch_embeddings = extract_embeddings(model, input_ids, attention_mask, device)

        # Store embeddings for each layer
        for layer_name, embeddings in batch_embeddings.items():
            if layer_name not in layer_embeddings:
                layer_embeddings[layer_name] = []

            # Convert to numpy and store
            embeddings_np = embeddings.cpu().numpy()
            layer_embeddings[layer_name].append(embeddings_np)

        sequences_processed += input_ids.shape[0]

    # Concatenate all batches
    logger.info("Concatenating embeddings from all batches...")
    for layer_name in layer_embeddings:
        layer_embeddings[layer_name] = np.concatenate(
            layer_embeddings[layer_name], axis=0
        )
        logger.info(f"{layer_name}: shape {layer_embeddings[layer_name].shape}")

    return layer_embeddings


def run_analyses_for_k(
    original_embeddings: Dict[str, np.ndarray],
    k: int,
    output_dir: str,
    logger: logging.Logger,
) -> Dict[str, Dict]:
    """
    Run all analyses for a specific k value.

    Returns:
        Dictionary mapping layer names to analysis results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running analyses for k={k}")
    logger.info(f"{'='*80}\n")

    all_results = {}

    # Initialize analyzers
    variance_analyzer = VarianceAnalysis(
        max_covariance_distance=config.VARIANCE_COVARIANCE_MAX_DISTANCE
    )
    norm_analyzer = NormAnalysis()
    info_theory_analyzer = InformationTheoryAnalysis(n_bins=config.ENTROPY_BINS)
    spectral_analyzer = SpectralAnalysis(window_size=config.SPECTRAL_WINDOW_SIZE)
    rank_analyzer = RankAnalysis(
        explained_variance_threshold=config.SVD_EXPLAINED_VARIANCE_THRESHOLD
    )

    # Process each layer
    for layer_name, original_emb in tqdm(
        original_embeddings.items(), desc=f"Analyzing layers (k={k})"
    ):
        logger.info(f"Analyzing {layer_name}...")

        # Apply averaging
        # Note: We need to add batch dimension if needed
        if len(original_emb.shape) == 2:
            original_emb = original_emb[np.newaxis, :, :]

        # Apply averaging with torch tensors
        original_tensor = torch.from_numpy(original_emb)
        averaged_tensor = apply_averaging(original_tensor, k)
        averaged_emb = averaged_tensor.numpy()

        # Run each analysis
        try:
            variance_results = variance_analyzer.analyze(
                original_emb, averaged_emb, layer_name, k, output_dir
            )
        except Exception as e:
            logger.error(f"Variance analysis failed for {layer_name}, k={k}: {e}")
            variance_results = {"error": str(e)}

        try:
            norm_results = norm_analyzer.analyze(
                original_emb, averaged_emb, layer_name, k, output_dir
            )
        except Exception as e:
            logger.error(f"Norm analysis failed for {layer_name}, k={k}: {e}")
            norm_results = {"error": str(e)}

        try:
            info_results = info_theory_analyzer.analyze(
                original_emb, averaged_emb, layer_name, k, output_dir
            )
        except Exception as e:
            logger.error(
                f"Information theory analysis failed for {layer_name}, k={k}: {e}"
            )
            info_results = {"error": str(e)}

        try:
            spectral_results = spectral_analyzer.analyze(
                original_emb, averaged_emb, layer_name, k, output_dir
            )
        except Exception as e:
            logger.error(f"Spectral analysis failed for {layer_name}, k={k}: {e}")
            spectral_results = {"error": str(e)}

        try:
            rank_results = rank_analyzer.analyze(
                original_emb, averaged_emb, layer_name, k, output_dir
            )
        except Exception as e:
            logger.error(f"Rank analysis failed for {layer_name}, k={k}: {e}")
            rank_results = {"error": str(e)}

        # Combine results
        all_results[layer_name] = {
            "variance": variance_results,
            "norm": norm_results,
            "information_theory": info_results,
            "spectral": spectral_results,
            "rank": rank_results,
        }

    return all_results


def export_results_to_csv(all_results: Dict, output_path: str, logger: logging.Logger):
    """Export results to CSV format."""
    logger.info("Exporting results to CSV...")

    rows = []

    for k, k_results in all_results.items():
        for layer_name, layer_results in k_results.items():
            row = {"k": k, "layer": layer_name}

            # Extract variance metrics
            if "variance" in layer_results and "shrinkage" in layer_results["variance"]:
                shrinkage = layer_results["variance"]["shrinkage"]
                row["variance_shrinkage_factor"] = shrinkage.get(
                    "shrinkage_factor", np.nan
                )
                row["variance_reduction"] = shrinkage.get("variance_reduction", np.nan)

            # Extract norm metrics
            if "norm" in layer_results and "shrinkage" in layer_results["norm"]:
                norm_shrinkage = layer_results["norm"]["shrinkage"]
                row["norm_shrinkage_factor"] = norm_shrinkage.get(
                    "shrinkage_factor", np.nan
                )
                row["norm_reduction"] = norm_shrinkage.get("norm_reduction", np.nan)

            # Extract information theory metrics
            if (
                "information_theory" in layer_results
                and "retention" in layer_results["information_theory"]
            ):
                retention = layer_results["information_theory"]["retention"]
                row["info_retention_ratio"] = retention.get("retention_ratio", np.nan)
                row["info_loss"] = retention.get("information_loss", np.nan)

            # Extract spectral metrics
            if (
                "spectral" in layer_results
                and "energy_loss" in layer_results["spectral"]
            ):
                energy_loss = layer_results["spectral"]["energy_loss"]
                row["spectral_total_energy_loss_pct"] = energy_loss.get(
                    "total_energy_loss_percentage", np.nan
                )
                row["spectral_high_freq_loss_pct"] = energy_loss.get(
                    "high_freq_loss_percentage", np.nan
                )

            # Extract rank metrics
            if "rank" in layer_results and "rank_reduction" in layer_results["rank"]:
                rank_red = layer_results["rank"]["rank_reduction"]
                row["effective_rank_original"] = rank_red.get(
                    "original_effective_rank", np.nan
                )
                row["effective_rank_averaged"] = rank_red.get(
                    "averaged_effective_rank", np.nan
                )
                row["rank_reduction"] = rank_red.get("rank_reduction", np.nan)

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"CSV exported to {output_path}")


def export_results_to_json(all_results: Dict, output_path: str, logger: logging.Logger):
    """Export results to JSON format."""
    logger.info("Exporting results to JSON...")

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_results = convert_to_serializable(all_results)

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"JSON exported to {output_path}")


def create_summary_report(all_results: Dict, output_path: str, logger: logging.Logger):
    """Create a summary report in Markdown format."""
    logger.info("Creating summary report...")

    with open(output_path, "w") as f:
        f.write("# Token Averaging Analysis - Summary Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Model: {config.MODEL_NAME}\n")
        f.write(f"- Dataset: {config.DATASET_NAME} ({config.DATASET_CONFIG})\n")
        f.write(f"- Number of sequences: {config.NUM_SEQUENCES}\n")
        f.write(f"- K range: {config.K_MIN} to {config.K_MAX}\n\n")

        f.write("## Key Findings\n\n")

        # Analyze variance shrinkage
        f.write("### Variance Shrinkage\n\n")
        f.write("Shows how variance decreases with averaging window size.\n\n")

        # Analyze norm shrinkage
        f.write("### Norm Shrinkage\n\n")
        f.write("Shows how vector norms decrease with averaging.\n\n")

        # Analyze information retention
        f.write("### Information Retention\n\n")
        f.write("Shows how much information is preserved after averaging.\n\n")

        # Analyze spectral energy
        f.write("### Spectral Analysis\n\n")
        f.write("Shows energy distribution across frequency components.\n\n")

        # Analyze rank reduction
        f.write("### Rank Analysis\n\n")
        f.write("Shows how effective dimensionality changes with averaging.\n\n")

        f.write("## Plots\n\n")
        f.write(
            "Detailed plots are available in the `outputs/plots/` directory, organized by k value and layer.\n\n"
        )

        f.write("## Data Files\n\n")
        f.write("- **CSV**: `outputs/metrics/summary_metrics.csv`\n")
        f.write("- **JSON**: `outputs/metrics/summary_metrics.json`\n\n")

    logger.info(f"Summary report created at {output_path}")


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(description="Token Averaging Research Analysis")
    parser.add_argument(
        "--k_min", type=int, default=config.K_MIN, help="Minimum k value"
    )
    parser.add_argument(
        "--k_max", type=int, default=config.K_MAX, help="Maximum k value"
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=config.NUM_SEQUENCES,
        help="Number of sequences to process",
    )
    parser.add_argument(
        "--output_dir", type=str, default=config.OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--device", type=str, default=config.DEVICE, help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(config.LOGS_DIR)
    logger.info("=" * 80)
    logger.info("Token Averaging Research - Starting Analysis")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Model: {config.MODEL_NAME}")
    logger.info(f"  Dataset: {config.DATASET_NAME}")
    logger.info(f"  K range: {args.k_min} to {args.k_max}")
    logger.info(f"  Sequences: {args.num_sequences}")
    logger.info(f"  Device: {args.device}")
    logger.info("=" * 80)

    # Setup plot style
    setup_plot_style(config.PLOT_STYLE)

    # Load model
    logger.info("\nLoading model...")
    model, tokenizer = load_pythia_model(config.MODEL_NAME, args.device)

    # Collect embeddings (do this once for all k values)
    original_embeddings = collect_embeddings(
        model=model,
        tokenizer=tokenizer,
        num_sequences=args.num_sequences,
        max_length=config.MAX_SEQUENCE_LENGTH,
        batch_size=config.BATCH_SIZE,
        device=args.device,
        logger=logger,
    )

    # Generate k values (powers of 2)
    k_values = []
    k = args.k_min
    while k <= args.k_max:
        k_values.append(k)
        k *= 2

    logger.info(f"\nAnalyzing k values: {k_values}")

    # Run analyses for each k value
    all_results = {}

    for k in k_values:
        results = run_analyses_for_k(
            original_embeddings=original_embeddings,
            k=k,
            output_dir=args.output_dir,
            logger=logger,
        )
        all_results[k] = results

    # Export results
    logger.info("\n" + "=" * 80)
    logger.info("Exporting Results")
    logger.info("=" * 80)

    os.makedirs(config.METRICS_DIR, exist_ok=True)

    csv_path = os.path.join(config.METRICS_DIR, "summary_metrics.csv")
    export_results_to_csv(all_results, csv_path, logger)

    json_path = os.path.join(config.METRICS_DIR, "summary_metrics.json")
    export_results_to_json(all_results, json_path, logger)

    report_path = os.path.join(args.output_dir, "summary_report.md")
    create_summary_report(all_results, report_path, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Analysis Complete!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"  - CSV: {csv_path}")
    logger.info(f"  - JSON: {json_path}")
    logger.info(f"  - Report: {report_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
