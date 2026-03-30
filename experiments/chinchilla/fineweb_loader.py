"""
FineWeb streaming data loader for Chinchilla-scale training.

Uses OLM's HuggingFaceTextDataset + OLM's DataLoader(distributed=True) so
that multi-GPU sharding is handled automatically by the OLM library.

When OLM is not installed, or if its HuggingFace dataset API does not support
the required arguments, we fall back to a hand-rolled streaming loader that
reproduces the same behaviour.

Multi-GPU note
--------------
OLM's DataLoader(distributed=True) internally creates a DistributedSampler
that assigns each GPU a disjoint shard of the dataset, exactly as described in
https://github.com/openlanguagemodel/openlanguagemodel/blob/main/docs/datasets-and-training.md

Held-out eval shard
-------------------
FineWeb sample-10BT only ships a "train" split.  The first EVAL_DOCS
documents are reserved as a held-out eval shard; training skips those.

Usage
-----
    from experiments.chinchilla.fineweb_loader import build_dataloaders
    train_dl, eval_dl = build_dataloaders(tokenizer, seq_len=1024,
                                          batch_size=16, distributed=True)
"""

from __future__ import annotations

import os
import sys
from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINEWEB_REPO   = "HuggingFaceFW/fineweb"
FINEWEB_SUBSET = "sample-10BT"

# First EVAL_DOCS documents are reserved as the held-out eval shard.
EVAL_DOCS = 5_000   # ~50M tokens at typical FineWeb document lengths


# ---------------------------------------------------------------------------
# Fallback: hand-rolled streaming loader (used when OLM dataset unavailable)
# ---------------------------------------------------------------------------

def _doc_stream(
    skip: int = 0,
    take: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
    cache_dir: Optional[str] = None,
) -> Iterator[str]:
    """Yield raw text strings from FineWeb, sharded across world_size workers."""
    from datasets import load_dataset

    load_kwargs = dict(
        path=FINEWEB_REPO,
        name=FINEWEB_SUBSET,
        split="train",
        streaming=True,
    )
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    ds = load_dataset(**load_kwargs)
    yielded  = 0
    shard_i  = 0
    for abs_i, example in enumerate(ds):
        if abs_i < skip:
            continue
        if shard_i % world_size == rank:
            yield example["text"]
            yielded += 1
            if take is not None and yielded >= take:
                break
        shard_i += 1


def _tokenize_and_pack(
    text_iter: Iterator[str],
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
) -> Iterator[torch.Tensor]:
    """Tokenize + pack into non-padded seq_len-token chunks."""
    buf: list[int] = []
    eos = tokenizer.eos_token_id or 0
    for text in text_iter:
        if not text or not text.strip():
            continue
        buf.extend(tokenizer.encode(text, add_special_tokens=False))
        buf.append(eos)
        while len(buf) >= seq_len:
            yield torch.tensor(buf[:seq_len], dtype=torch.long)
            buf = buf[seq_len:]


class _FallbackTrainDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len, rank=0, world_size=1, cache_dir=None):
        super().__init__()
        self.tokenizer  = tokenizer
        self.seq_len    = seq_len
        self.rank       = rank
        self.world_size = world_size
        self.cache_dir  = cache_dir

    def __iter__(self):
        while True:
            yield from _tokenize_and_pack(
                _doc_stream(skip=EVAL_DOCS, rank=self.rank,
                            world_size=self.world_size, cache_dir=self.cache_dir),
                self.tokenizer, self.seq_len,
            )


class _FallbackEvalDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len, cache_dir=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len   = seq_len
        self.cache_dir = cache_dir

    def __iter__(self):
        yield from _tokenize_and_pack(
            _doc_stream(skip=0, take=EVAL_DOCS, rank=0, world_size=1,
                        cache_dir=self.cache_dir),
            self.tokenizer, self.seq_len,
        )


# ---------------------------------------------------------------------------
# OLM-native loader (preferred)
# ---------------------------------------------------------------------------

def _build_olm_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    batch_size: int,
    eval_batches: int,
    num_workers: int,
    distributed: bool,
    cache_dir: Optional[str] = None,
):
    """
    Build train + eval loaders using OLM's HuggingFaceTextDataset and
    OLM's DataLoader(distributed=True).

    OLM's DataLoader with distributed=True automatically creates a
    DistributedSampler that shards data across GPUs — no manual rank
    offset arithmetic needed.
    """
    from olm.data.datasets import HuggingFaceTextDataset
    from olm.data.datasets import DataLoader as OLMDataLoader

    # Training dataset — FineWeb sample-10BT, skipping the eval shard
    # OLM's HuggingFaceTextDataset handles streaming + tokenization + packing.
    olm_kwargs = {}
    if cache_dir:
        olm_kwargs["cache_dir"] = cache_dir

    train_ds = HuggingFaceTextDataset(
        repo_id=FINEWEB_REPO,
        name=FINEWEB_SUBSET,
        tokenizer=tokenizer,
        context_length=seq_len,
        streaming=True,
        shuffle=True,
        skip=EVAL_DOCS,
        **olm_kwargs,
    )

    eval_ds = HuggingFaceTextDataset(
        repo_id=FINEWEB_REPO,
        name=FINEWEB_SUBSET,
        tokenizer=tokenizer,
        context_length=seq_len,
        streaming=True,
        shuffle=False,
        take=EVAL_DOCS,
        **olm_kwargs,
    )

    # OLM's DataLoader:
    #   distributed=True  → DistributedSampler (each GPU gets unique data)
    #   persistent_workers=True, pin_memory=True (enabled automatically)
    train_dl = OLMDataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=distributed,
    )
    eval_dl = OLMDataLoader(
        eval_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=False,    # eval is rank-0 only in train.py
    )

    return train_dl, eval_dl


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int = 1024,
    batch_size: int = 16,
    eval_batches: int = 64,
    num_workers: int = 2,
    rank: int = 0,
    world_size: int = 1,
    distributed: bool = False,
    cache_dir: Optional[str] = None,
):
    """
    Build (train_dataloader, eval_dataloader) for FineWeb sample-10BT.

    Prefers OLM's native HuggingFaceTextDataset + DataLoader(distributed=True).
    Falls back to the hand-rolled streaming loader if OLM is not available.

    Args:
        tokenizer    : Pythia GPT-NeoX tokenizer
        seq_len      : tokens per sequence (default 1024)
        batch_size   : sequences per batch *per GPU*
        eval_batches : kept for API compatibility (not used by OLM loader)
        num_workers  : DataLoader worker processes per rank
        rank         : this process's rank (for fallback sharding)
        world_size   : total number of processes (for fallback sharding)
        distributed  : enable distributed sampling (True when using torchrun)
        cache_dir    : local directory for HuggingFace dataset cache;
                       overrides ~/.cache/huggingface when set

    Returns:
        (train_dl, eval_dl)
    """
    # --- Try OLM-native path first ------------------------------------------
    try:
        loaders = _build_olm_dataloaders(
            tokenizer=tokenizer,
            seq_len=seq_len,
            batch_size=batch_size,
            eval_batches=eval_batches,
            num_workers=num_workers,
            distributed=distributed,
            cache_dir=cache_dir,
        )
        if rank == 0:
            print("[fineweb_loader] Using OLM HuggingFaceTextDataset + "
                  "OLM DataLoader(distributed={})".format(distributed), flush=True)
        return loaders
    except Exception as olm_err:
        if rank == 0:
            print(f"[fineweb_loader] OLM dataset unavailable ({olm_err}); "
                  f"falling back to hand-rolled loader.", flush=True)

    # --- Fallback: hand-rolled streaming loader -----------------------------
    from torch.utils.data import DataLoader

    train_ds = _FallbackTrainDataset(tokenizer, seq_len, rank=rank,
                                     world_size=world_size, cache_dir=cache_dir)
    eval_ds  = _FallbackEvalDataset(tokenizer, seq_len, cache_dir=cache_dir)

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          num_workers=num_workers, pin_memory=True)
    eval_dl  = DataLoader(eval_ds,  batch_size=batch_size,
                          num_workers=num_workers, pin_memory=True)

    return train_dl, eval_dl


def estimate_total_batches(
    target_tokens: int,
    seq_len: int,
    batch_size: int,
    world_size: int = 1,
) -> int:
    """Optimizer steps needed to consume target_tokens across all GPUs."""
    tokens_per_global_step = batch_size * world_size * seq_len
    return (target_tokens + tokens_per_global_step - 1) // tokens_per_global_step
