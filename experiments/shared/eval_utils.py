"""
Shared evaluation and persistence utilities for LLM averaging experiments.

compute_perplexity  – run a forward pass over WikiText-103 test split and
                      return exp(mean NLL)
save_results        – write a list of result dicts to CSV + JSON
setup_exp_logging   – configure file + stdout logging
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
from datetime import datetime
from typing import List, Optional

import torch
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.data_loader import get_data_iterator
from experiments.shared.averaged_lm import AveragedLanguageModel


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_exp_logging(log_dir: str, prefix: str = "experiment") -> logging.Logger:
    """
    Configure root logging to write both to a timestamped file and to stdout.
    Returns the root logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger()


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: AveragedLanguageModel,
    tokenizer,
    num_sequences: int,
    device: str,
    split: str = "test",
    max_length: int = 512,
    batch_size: int = 4,
) -> float:
    """
    Evaluate the model on WikiText-103 and return perplexity.

    The model is called in eval mode.  Each forward pass returns (loss, logits)
    where loss is the mean NLL over the compressed sequence.  We accumulate
    the sum of NLL values and the total number of valid tokens to compute a
    corpus-level PPL.

    Returns:
        perplexity (float) — exp(mean_nll)
    """
    model.eval()
    data_iter = get_data_iterator(
        tokenizer=tokenizer,
        num_sequences=num_sequences,
        max_length=max_length,
        batch_size=batch_size,
        split=split,
    )

    total_nll = 0.0
    total_batches = 0
    expected_batches = max(1, num_sequences // batch_size)

    with tqdm(data_iter, total=expected_batches, desc="  eval", leave=False) as pbar:
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            try:
                loss, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                if torch.isfinite(loss):
                    total_nll += loss.item()
                    total_batches += 1
                    pbar.set_postfix({"nll": f"{loss.item():.3f}"})
            except Exception as exc:
                logging.getLogger().warning(f"Skipping batch due to error: {exc}")
                continue

    if total_batches == 0:
        return float("inf")

    mean_nll = total_nll / total_batches
    return math.exp(mean_nll)


def compute_perplexity_with_grad(
    model: AveragedLanguageModel,
    tokenizer,
    num_sequences: int,
    device: str,
    split: str = "train",
    max_length: int = 512,
    batch_size: int = 4,
) -> float:
    """
    Same as compute_perplexity but without @torch.no_grad() — used when
    perplexity is computed mid-training to avoid disabling grad tracking.
    """
    model.eval()
    data_iter = get_data_iterator(
        tokenizer=tokenizer,
        num_sequences=num_sequences,
        max_length=max_length,
        batch_size=batch_size,
        split=split,
    )

    total_nll = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in data_iter:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            try:
                loss, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                if torch.isfinite(loss):
                    total_nll += loss.item()
                    total_batches += 1
            except Exception:
                continue

    if total_batches == 0:
        return float("inf")
    return math.exp(total_nll / total_batches)


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_results(rows: List[dict], csv_path: str) -> None:
    """
    Write a list of result dicts to CSV and a matching JSON file.

    Args:
        rows:     list of dicts (all must share the same keys)
        csv_path: path to the output CSV file; JSON written alongside it
    """
    if not rows:
        logging.getLogger().warning("save_results called with empty rows list")
        return

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    json_path = csv_path.replace(".csv", ".json")

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as fh:
        json.dump(rows, fh, indent=2)

    logging.getLogger().info(f"Saved {len(rows)} results → {csv_path}")
    logging.getLogger().info(f"Saved {len(rows)} results → {json_path}")


def make_result_row(
    experiment: str,
    config_name: str,
    method_family: str,
    nominal_k: int,
    compression_ratio: float,
    ppl: float,
    model_name: str,
    num_sequences: int,
    train_steps: int = 0,
    ppl_before: Optional[float] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Build a standardised result row dict.

    Args:
        experiment:       "zero_shot" | "from_scratch" | "finetune"
        config_name:      registry key, e.g. "uniform_k2"
        method_family:    "uniform" | "dynamic" | "overlapping" | "weighted" | "learnable" | "baseline"
        nominal_k:        representative window size
        compression_ratio: fraction of original sequence length removed
        ppl:              evaluated perplexity (after training, if applicable)
        model_name:       HuggingFace model identifier
        num_sequences:    sequences used for evaluation
        train_steps:      steps trained (0 for zero-shot)
        ppl_before:       perplexity before training (finetune only, else None)
        extra:            additional columns to include
    """
    row = {
        "experiment": experiment,
        "config_name": config_name,
        "method": method_family,
        "nominal_k": nominal_k,
        "compression_ratio": round(compression_ratio, 4),
        "ppl": round(ppl, 4),
        "model_name": model_name,
        "num_sequences": num_sequences,
        "train_steps": train_steps,
    }
    if ppl_before is not None:
        row["ppl_before"] = round(ppl_before, 4)
        row["ppl_delta"] = round(ppl - ppl_before, 4)
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: AveragedLanguageModel, path: str, step: int) -> None:
    """Save model state dict and step number."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"step": step, "state_dict": model.state_dict()}, path)
    logging.getLogger().info(f"Checkpoint saved → {path}  (step {step})")


def load_checkpoint(model: AveragedLanguageModel, path: str) -> int:
    """Load state dict from checkpoint, return the saved step number."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    return ckpt.get("step", 0)
