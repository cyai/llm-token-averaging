"""
FineWeb streaming data loader for Chinchilla-scale training.

Streams HuggingFaceFW/fineweb (sample-10BT subset) and packs tokenized text
into fixed-length non-padded chunks of `seq_len` tokens using the Pythia
GPT-NeoX tokenizer.  No local bulk download is needed — HuggingFace datasets
handles streaming transparently.

Multi-GPU / DDP sharding
------------------------
When training with torchrun across N GPUs each GPU should see a disjoint
slice of the document stream.  Pass `rank` and `world_size` to
`build_dataloaders`; each rank will pick up every world_size-th document
starting from `rank`, ensuring no data duplication across GPUs.

Held-out eval shard
-------------------
FineWeb sample-10BT only ships a "train" split.  We create a reproducible
eval shard by taking the first `EVAL_DOCS` documents.  Training skips those
docs and uses the rest.

Usage (single-GPU)
------------------
    from experiments.chinchilla.fineweb_loader import build_dataloaders
    train_dl, eval_dl = build_dataloaders(tokenizer, seq_len=1024, batch_size=8)

Usage (multi-GPU — called inside torchrun worker)
--------------------------------------------------
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train_dl, eval_dl = build_dataloaders(
        tokenizer, seq_len=1024, batch_size=8,
        rank=local_rank, world_size=world_size,
    )
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

# First EVAL_DOCS documents are reserved as the held-out eval shard.
EVAL_DOCS = 5_000   # ~50M tokens at typical FineWeb document lengths


# ---------------------------------------------------------------------------
# Core document stream helpers
# ---------------------------------------------------------------------------

def _doc_stream(
    skip: int = 0,
    take: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[str]:
    """
    Yield raw text strings from FineWeb sample-10BT.

    When world_size > 1 each rank receives a disjoint interleaved slice of
    the stream: rank r takes every world_size-th document starting at offset r
    (after the initial `skip` documents are consumed).

    Args:
        skip       : absolute number of leading documents to skip before
                     any sharding.  Used to exclude the eval shard.
        take       : stop after yielding this many documents (used for eval).
        rank       : this worker's rank (0-based)
        world_size : total number of parallel workers
    """
    from datasets import load_dataset

    ds = load_dataset(
        FINEWEB_REPO,
        name=FINEWEB_SUBSET,
        split="train",
        streaming=True,
    )

    yielded = 0
    shard_idx = 0   # index within the post-skip document stream

    for abs_idx, example in enumerate(ds):
        if abs_idx < skip:
            continue

        # Shard: this rank only takes documents where shard_idx % world_size == rank
        if shard_idx % world_size == rank:
            yield example["text"]
            yielded += 1
            if take is not None and yielded >= take:
                break

        shard_idx += 1


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


# ---------------------------------------------------------------------------
# IterableDatasets
# ---------------------------------------------------------------------------

class FineWebTrainDataset(IterableDataset):
    """
    Infinite streaming training dataset backed by FineWeb sample-10BT.

    Skips the first EVAL_DOCS documents (held out for eval) and packs the
    rest into seq_len-token chunks indefinitely, cycling through the stream.

    In multi-GPU mode each rank is assigned an interleaved shard of the
    document stream so every GPU sees different data at every step.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int = 1024,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:   # cycle; caller stops by token count
            yield from _tokenize_and_pack(
                _doc_stream(
                    skip=EVAL_DOCS,
                    rank=self.rank,
                    world_size=self.world_size,
                ),
                self.tokenizer,
                self.seq_len,
            )


class FineWebEvalDataset(IterableDataset):
    """
    Fixed eval shard: the first EVAL_DOCS documents, packed into seq_len chunks.
    All ranks read the same eval shard (evaluation is rank-0-only in train.py).
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, seq_len: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield from _tokenize_and_pack(
            _doc_stream(skip=0, take=EVAL_DOCS, rank=0, world_size=1),
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
    rank: int = 0,
    world_size: int = 1,
):
    """
    Build (train_dataloader, eval_dataloader) for FineWeb sample-10BT.

    The train dataloader cycles indefinitely; stop training by token count.
    The eval dataloader is finite (eval_batches × batch_size sequences drawn
    from the fixed eval shard).

    Args:
        tokenizer    : Pythia GPT-NeoX tokenizer
        seq_len      : tokens per sequence (default 1024)
        batch_size   : sequences per batch *per GPU*
        eval_batches : number of batches used for each eval pass
        num_workers  : DataLoader worker processes (0 = main process only)
        rank         : this process's rank (0 for single-GPU)
        world_size   : total number of parallel processes (1 for single-GPU)

    Returns:
        (train_dl, eval_dl)
    """
    train_ds = FineWebTrainDataset(tokenizer, seq_len, rank=rank, world_size=world_size)
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


def estimate_total_batches(
    target_tokens: int,
    seq_len: int,
    batch_size: int,
    world_size: int = 1,
) -> int:
    """
    Return the number of optimizer steps needed to consume `target_tokens`
    across all GPUs combined.

    Each step processes `batch_size * world_size * seq_len` tokens globally.
    """
    tokens_per_global_step = batch_size * world_size * seq_len
    return (target_tokens + tokens_per_global_step - 1) // tokens_per_global_step
