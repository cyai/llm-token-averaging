"""
Data loading utilities for WikiText-103.
"""

import logging
from datasets import load_dataset
from typing import Iterator, Dict, List
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def load_wikitext103(split: str = "train", streaming: bool = False):
    """
    Load WikiText-103 dataset.

    Args:
        split: Dataset split ("train", "validation", or "test")
        streaming: Whether to use streaming mode

    Returns:
        Dataset object
    """
    logger.info(f"Loading WikiText-103 dataset, split: {split}")

    dataset = load_dataset(
        "wikitext", "wikitext-103-v1", split=split, streaming=streaming
    )

    logger.info("Dataset loaded successfully")
    return dataset


def get_data_iterator(
    tokenizer: AutoTokenizer,
    num_sequences: int = 1000,
    max_length: int = 512,
    batch_size: int = 8,
    split: str = "train",
) -> Iterator[Dict[str, torch.Tensor]]:
    """
    Create an iterator over tokenized sequences from WikiText-103.

    Args:
        tokenizer: Tokenizer for encoding text
        num_sequences: Number of sequences to process
        max_length: Maximum sequence length
        batch_size: Batch size for processing
        split: Dataset split

    Yields:
        Dictionary with 'input_ids' and 'attention_mask' tensors
    """
    dataset = load_wikitext103(split=split, streaming=True)

    sequences_processed = 0
    batch_texts = []

    for example in dataset:
        text = example["text"].strip()

        # Skip empty lines and very short texts
        if len(text) < 50:
            continue

        batch_texts.append(text)

        # Process batch when full
        if len(batch_texts) >= batch_size:
            # Tokenize batch with fixed padding to max_length
            encoded = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            yield encoded

            sequences_processed += len(batch_texts)
            batch_texts = []

            if sequences_processed >= num_sequences:
                break

    # Process remaining texts
    if batch_texts and sequences_processed < num_sequences:
        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        yield encoded

    logger.info(f"Processed {sequences_processed} sequences")
