"""
FineWeb streaming data loader for Chinchilla-scale training.

Streams HuggingFaceFW/fineweb (sample-10BT subset) and packs tokenized text
into fixed-length non-padded chunks of `seq_len` tokens using the Pythia
GPT-NeoX tokenizer.  No local bulk download is needed — HuggingFace datasets
handles streaming transparently.

Held-out eval shard
-------------------
FineWeb sample-10BT only ships a "train" split.  We create a reproducible
eval shard by skipping the first `eval_skip_docs` documents and taking the
next `eval_docs` documents.  Training uses the remaining documents.

Usage
-----
    from experiments.chinchilla.fineweb_loader import build_dataloaders

    train_dl, eval_dl = build_dataloaders(tokenizer, seq_len=1024,
                                          batch_size=8, eval_batches=64)
    for batch in train_dl:
        input_ids = batch          # LongTensor [B, seq_len]
"""

from __future__ import annotations

import os
import sys
from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import PreTrainedTokenizerBase

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINEWEB_REPO   = "HuggingFaceFW/fineweb"
FINEWEB_SUBSET = "sample-10BT"

# Number of documents reserved as a held-out eval shard.
# These are *skipped* during training and used for eval only.
EVAL_DOCS = 5_000          # ~50M tokens at typical FineWeb doc lengths
EVAL_SKIP_DOCS = 0         # take first EVAL_DOCS docs for eval


# ---------------------------------------------------------------------------
# Core packing utilities
# ---------------------------------------------------------------------------

def _tokenize_and_pack(
    text_iter: Iterator[str],
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
) -> Iterator[torch.Tensor]:
    """
    Tokenize documents from `text_iter` and pack them into non-padded chunks
    of exactly `seq_len` tokens.  Documents are concatenated with an EOS token
    between them; remainder tokens at the end of the stream are discarded.

    Yields 1-D LongTensors of shape [seq_len].
    """
    buf: list[int] = []
    eos = tokenizer.eos_token_id or 0

    for text in text_iter:
        if not text or not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        buf.append(eos)

        while len(buf) >= seq_len:
            chunk = buf[:seq_len]
            buf = buf[seq_len:]
            yield torch.tensor(chunk, dtype=torch.long)


def _doc_stream(split: str = "train", skip: int = 0, take: Optional[int] = None):
    """
    Yield raw text strings from FineWeb sample-10BT.

    Args:
        split : dataset split ("train" — the only one available)
        skip  : number of leading documents to skip
        take  : if not None, stop after this many documents
    """
    from datasets import load_dataset

    ds = load_dataset(
        FINEWEB_REPO,
        name=FINEWEB_SUBSET,
        split=split,
        streaming=True,
    )
    count = 0
    for i, example in enumerate(ds):
        if i < skip:
            continue
        yield example["text"]
        count += 1
        if take is not None and count >= take:
            break


# ---------------------------------------------------------------------------
# IterableDatasets
# ---------------------------------------------------------------------------

class FineWebTrainDataset(IterableDataset):
    """
    Streaming training dataset.  Skips the first EVAL_DOCS documents (held
    out for eval) and packs the rest into seq_len-token chunks indefinitely,
    cycling through the stream.

    Attributes:
        tokenizer : Pythia GPT-NeoX tokenizer
        seq_len   : tokens per chunk (default 1024)
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, seq_len: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:  # cycle forever; caller stops by token count
            for chunk in _tokenize_and_pack(
                _doc_stream(skip=EVAL_DOCS),
                self.tokenizer,
                self.seq_len,
            ):
                yield chunk


class FineWebEvalDataset(IterableDataset):
    """
    Fixed eval shard: the first EVAL_DOCS documents of FineWeb, packed into
    seq_len-token chunks.  Iterating twice gives the same chunks each time.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, seq_len: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield from _tokenize_and_pack(
            _doc_stream(skip=EVAL_SKIP_DOCS, take=EVAL_DOCS),
            self.tokenizer,
            self.seq_len,
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int = 1024,
    batch_size: int = 8,
    eval_batches: int = 64,
    num_workers: int = 0,
):
    """
    Build (train_dataloader, eval_dataloader) for FineWeb sample-10BT.

    The train dataloader cycles indefinitely; stop training by token count,
    not by epoch.  The eval dataloader is finite (eval_batches × batch_size
    sequences drawn from the fixed eval shard).

    Args:
        tokenizer    : Pythia GPT-NeoX tokenizer
        seq_len      : tokens per sequence (default 1024)
        batch_size   : sequences per batch
        eval_batches : number of batches used for each eval pass
        num_workers  : DataLoader worker processes (0 = main process only)

    Returns:
        (train_dl, eval_dl)
    """
    train_ds = FineWebTrainDataset(tokenizer, seq_len)
    eval_ds  = FineWebEvalDataset(tokenizer, seq_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dl, eval_dl


def estimate_total_batches(target_tokens: int, seq_len: int, batch_size: int) -> int:
    """Return the number of training steps needed to consume `target_tokens`."""
    tokens_per_step = batch_size * seq_len
    return (target_tokens + tokens_per_step - 1) // tokens_per_step
