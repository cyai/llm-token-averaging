"""
Shared utilities used by all individual method runner scripts.

Provides:
  setup_logging              – timestamped file + stdout logging
  collect_embeddings         – extract and cache embeddings from the model once
  run_analyses_for_averaged  – run all 5 analysis modules on a pair of embedding dicts
  flatten_results_to_rows    – convert nested results to a flat list of CSV rows
  export_results_to_csv      – write rows to CSV
  export_results_to_json     – write nested dict to JSON
  create_summary_report      – write a Markdown report for one method
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from utils.embedding_extractor import extract_embeddings
from utils.data_loader import get_data_iterator
from analysis import (
    VarianceAnalysis,
    NormAnalysis,
    InformationTheoryAnalysis,
    SpectralAnalysis,
    RankAnalysis,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, prefix: str = "run") -> logging.Logger:
    """
    Configure root logging to write both to a timestamped file and to stdout.

    Args:
        log_dir: directory where the log file is created
        prefix: filename prefix (default 'run')

    Returns:
        A configured Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding collection
# ---------------------------------------------------------------------------

def collect_embeddings(
    model,
    tokenizer,
    num_sequences: int,
    max_length: int,
    batch_size: int,
    device: str,
    logger: logging.Logger,
) -> Dict[str, np.ndarray]:
    """
    Run the model on `num_sequences` sequences from WikiText-103 and collect
    the hidden states of every layer.

    Args:
        model: EmbeddingExtractorModel instance (with forward hooks registered)
        tokenizer: HuggingFace tokenizer
        num_sequences: number of sequences to process
        max_length: maximum token length per sequence
        batch_size: batch size for inference
        device: torch device string
        logger: logger instance

    Returns:
        Dict mapping layer names (e.g. "embedding", "layer_0" …) to
        numpy arrays of shape [n_sequences, max_length, hidden_dim].
    """
    logger.info(f"Collecting embeddings from {num_sequences} sequences …")

    layer_embeddings: Dict[str, List[np.ndarray]] = {}
    sequences_processed = 0

    data_iter = get_data_iterator(
        tokenizer=tokenizer,
        num_sequences=num_sequences,
        max_length=max_length,
        batch_size=batch_size,
    )

    for batch in tqdm(
        data_iter,
        desc="Extracting embeddings",
        total=num_sequences // batch_size,
    ):
        if sequences_processed >= num_sequences:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        batch_emb = extract_embeddings(model, input_ids, attention_mask, device)

        for layer_name, emb_tensor in batch_emb.items():
            layer_embeddings.setdefault(layer_name, [])
            layer_embeddings[layer_name].append(emb_tensor.cpu().numpy())

        sequences_processed += input_ids.shape[0]

    logger.info("Concatenating batches …")
    result: Dict[str, np.ndarray] = {}
    for layer_name, arrays in layer_embeddings.items():
        result[layer_name] = np.concatenate(arrays, axis=0)
        logger.info(f"  {layer_name}: {result[layer_name].shape}")

    return result


# ---------------------------------------------------------------------------
# Core analysis runner
# ---------------------------------------------------------------------------

def run_analyses_for_averaged(
    original_embeddings: Dict[str, np.ndarray],
    averaged_embeddings: Dict[str, np.ndarray],
    k_label: Any,
    output_dir: str,
    logger: logging.Logger,
) -> Dict[str, Dict]:
    """
    Run all five analysis modules (variance, norm, information_theory, spectral,
    rank) on every layer that appears in both `original_embeddings` and
    `averaged_embeddings`.

    Args:
        original_embeddings: layer_name → [n, seq, dim]
        averaged_embeddings: layer_name → [n, new_seq, dim]  (already averaged)
        k_label: label used in output subdirectory names and plot titles.
                 Can be an int (e.g. 4) or a descriptive string (e.g. "dyn_alt").
        output_dir: root output directory for this analysis run
        logger: logger instance

    Returns:
        Dict[layer_name, {variance, norm, information_theory, spectral, rank}]
    """
    variance_analyzer = VarianceAnalysis(
        max_covariance_distance=config.VARIANCE_COVARIANCE_MAX_DISTANCE
    )
    norm_analyzer = NormAnalysis()
    info_theory_analyzer = InformationTheoryAnalysis(n_bins=config.ENTROPY_BINS)
    spectral_analyzer = SpectralAnalysis(window_size=config.SPECTRAL_WINDOW_SIZE)
    rank_analyzer = RankAnalysis(
        explained_variance_threshold=config.SVD_EXPLAINED_VARIANCE_THRESHOLD
    )

    all_results: Dict[str, Dict] = {}

    common_layers = [
        lname for lname in original_embeddings if lname in averaged_embeddings
    ]

    for layer_name in tqdm(common_layers, desc=f"Analysing layers (k={k_label})"):
        orig = original_embeddings[layer_name]   # [n, seq, dim]
        avg = averaged_embeddings[layer_name]    # [n, new_seq, dim]

        # Analysis modules expect 3-D arrays; add batch axis if needed
        if orig.ndim == 2:
            orig = orig[np.newaxis]
        if avg.ndim == 2:
            avg = avg[np.newaxis]

        layer_out: Dict[str, Any] = {}

        for name, analyzer, method in [
            ("variance",         variance_analyzer,    "analyze"),
            ("norm",             norm_analyzer,        "analyze"),
            ("information_theory", info_theory_analyzer, "analyze"),
            ("spectral",         spectral_analyzer,    "analyze"),
            ("rank",             rank_analyzer,        "analyze"),
        ]:
            try:
                layer_out[name] = getattr(analyzer, method)(
                    orig, avg, layer_name, k_label, output_dir
                )
            except Exception as exc:
                logger.error(
                    f"{name} analysis failed for {layer_name}, k={k_label}: {exc}"
                )
                layer_out[name] = {"error": str(exc)}

        all_results[layer_name] = layer_out

    return all_results


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def flatten_results_to_rows(
    results: Dict[str, Dict],
    method_name: str,
    extra_meta: Optional[Dict] = None,
) -> List[Dict]:
    """
    Flatten a nested results dict (layer_name → analysis_name → metrics)
    into a list of flat dicts suitable for building a DataFrame / CSV.

    Args:
        results: output of run_analyses_for_averaged
        method_name: identifier for this method (e.g. "dynamic_alternating")
        extra_meta: optional dict of extra columns to add to every row
                    (e.g. {"k": 4, "stride": 2})

    Returns:
        List of flat row dicts.
    """
    rows: List[Dict] = []
    extra_meta = extra_meta or {}

    for layer_name, layer_results in results.items():
        row: Dict[str, Any] = {"method": method_name, "layer": layer_name}
        row.update(extra_meta)

        # Variance
        v = layer_results.get("variance", {})
        shrinkage = v.get("shrinkage", {})
        row["variance_shrinkage_factor"] = shrinkage.get("shrinkage_factor", np.nan)
        row["variance_reduction"] = shrinkage.get("variance_reduction", np.nan)

        # Norm
        n = layer_results.get("norm", {})
        norm_shrinkage = n.get("shrinkage", {})
        row["norm_shrinkage_factor"] = norm_shrinkage.get("shrinkage_factor", np.nan)
        row["norm_reduction"] = norm_shrinkage.get("norm_reduction", np.nan)

        # Information theory
        it = layer_results.get("information_theory", {})
        retention = it.get("retention", {})
        row["info_retention_ratio"] = retention.get("retention_ratio", np.nan)
        row["info_loss"] = retention.get("information_loss", np.nan)

        # Spectral
        sp = layer_results.get("spectral", {})
        energy_loss = sp.get("energy_loss", {})
        row["spectral_total_energy_loss_pct"] = energy_loss.get(
            "total_energy_loss_percentage", np.nan
        )
        row["spectral_high_freq_loss_pct"] = energy_loss.get(
            "high_freq_loss_percentage", np.nan
        )

        # Rank
        rk = layer_results.get("rank", {})
        rank_red = rk.get("rank_reduction", {})
        row["effective_rank_original"] = rank_red.get("original_effective_rank", np.nan)
        row["effective_rank_averaged"] = rank_red.get("averaged_effective_rank", np.nan)
        row["rank_reduction"] = rank_red.get("rank_reduction", np.nan)

        rows.append(row)

    return rows


def export_results_to_csv(
    rows: List[Dict],
    output_path: str,
    logger: logging.Logger,
) -> None:
    """Write a list of flat row dicts to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"CSV exported to {output_path}")


def export_results_to_json(
    results: Any,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """Serialise a (possibly nested) results dict to JSON."""

    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(_convert(results), fh, indent=2)
    logger.info(f"JSON exported to {output_path}")


def create_summary_report(
    all_results: Dict,
    method_name: str,
    output_path: str,
    logger: logging.Logger,
    extra_info: Optional[str] = None,
) -> None:
    """
    Write a Markdown summary report for one averaging method.

    Args:
        all_results: top-level results dict (keys are config labels,
                     values are the output of run_analyses_for_averaged)
        method_name: human-readable method name
        output_path: path to the .md file to write
        logger: logger instance
        extra_info: optional additional Markdown content to append
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write(f"# Token Averaging – {method_name} – Summary Report\n\n")
        fh.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        fh.write("## Configuration\n\n")
        fh.write(f"- Model: {config.MODEL_NAME}\n")
        fh.write(f"- Dataset: {config.DATASET_NAME} ({config.DATASET_CONFIG})\n\n")

        fh.write("## Configurations analysed\n\n")
        for label in all_results:
            fh.write(f"- `{label}`\n")

        fh.write("\n## Analysis modules run per configuration\n\n")
        fh.write("1. Variance Analysis\n")
        fh.write("2. Norm Analysis\n")
        fh.write("3. Information Theory Analysis\n")
        fh.write("4. Spectral Analysis\n")
        fh.write("5. Rank Analysis\n")

        fh.write(
            "\nDetailed metrics are in the `metrics/` subdirectory (CSV + JSON).\n"
        )
        fh.write("Plots are in the `plots/` subdirectory.\n")

        if extra_info:
            fh.write(f"\n## Notes\n\n{extra_info}\n")

    logger.info(f"Summary report written to {output_path}")
