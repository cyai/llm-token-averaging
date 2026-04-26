"""
FineWeb data loader for Chinchilla-scale training.

Loading priority
----------------
1. Local binary cache  (fastest — numpy memmap, zero network overhead)
2. OLM HuggingFaceTextDataset  (fast — library-managed streaming)
3. Hand-rolled HF streaming  (slowest — network + on-the-fly tokenisation)

Pre-tokenise once (recommended before any multi-GPU run):

    python -m experiments.chinchilla.fineweb_loader \
        --data_dir /data/fineweb \
        --tokenizer EleutherAI/pythia-70m \
        --num_proc 16

This writes two files:
    /data/fineweb/train.bin   — flat uint16 array, all training tokens
    /data/fineweb/eval.bin    — flat uint16 array, eval tokens

Subsequent calls to build_dataloaders() with data_dir= set will skip all
network I/O and read directly from disk via numpy memmap.

Multi-GPU note
--------------
The local-bin path uses PyTorch DistributedSampler so each GPU sees a
disjoint, non-overlapping slice of every epoch — no manual rank arithmetic.
num_workers can be set generously (e.g. 4) since workers just do memmap reads.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINEWEB_REPO = "HuggingFaceFW/fineweb"
FINEWEB_SUBSET = "sample-10BT"

# First EVAL_DOCS documents are reserved as the held-out eval shard.
EVAL_DOCS = 5_000  # ~50M tokens at typical FineWeb document lengths

DTYPE = np.uint16  # GPT-NeoX vocab = 50 257, fits in uint16


# ---------------------------------------------------------------------------
# 1.  LOCAL BINARY CACHE  (fastest path)
# ---------------------------------------------------------------------------


class _LocalBinDataset(Dataset):
    """
    Map-style dataset backed by a pre-tokenised flat binary file.

    The file is memory-mapped so the OS handles caching; multiple workers
    can read from it concurrently with zero data duplication.

    Returns plain (seq_len,) int64 tensors — compatible with the training
    loop's `batch.to(device)` pattern.
    """

    def __init__(self, path: Path, seq_len: int):
        self.seq_len = seq_len
        # +1 so we can return seq_len tokens (the LM shifts internally)
        self.data = np.memmap(path, dtype=DTYPE, mode="r")
        # Number of complete, non-overlapping windows
        self.n_seqs = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len].astype(np.int64)
        return torch.from_numpy(chunk)


def _build_local_dataloaders(
    data_dir: Path,
    seq_len: int,
    batch_size: int,
    num_workers: int,
    distributed: bool,
):
    """Build fast train + eval DataLoaders from pre-tokenised .bin files."""
    from torch.utils.data import DistributedSampler

    train_path = data_dir / "train.bin"
    eval_path = data_dir / "eval.bin"

    train_ds = _LocalBinDataset(train_path, seq_len)
    eval_ds = _LocalBinDataset(eval_path, seq_len)

    train_sampler = (
        DistributedSampler(train_ds, shuffle=True, drop_last=True)
        if distributed
        else None
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # shuffle only when not using DDP sampler
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    return train_dl, eval_dl


# ---------------------------------------------------------------------------
# 2.  DATASET PREPARATION  (run once, produces the .bin files above)
# ---------------------------------------------------------------------------


def _tokenize_batch(args):
    """Top-level function (picklable) used by multiprocessing pool."""
    texts, tokenizer_name, eos_id = args
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    ids: list[int] = []
    for text in texts:
        if text and text.strip():
            ids.extend(tok.encode(text, add_special_tokens=False))
            ids.append(eos_id)
    return ids


def prepare_dataset(
    data_dir: str | Path,
    tokenizer_name: str = "EleutherAI/pythia-70m",
    num_proc: int = 4,
    chunk_size: int = 1_000,
    max_train_tokens: Optional[int] = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """
    Stream FineWeb sample-10BT, tokenise it, and write flat uint16 binary
    files to data_dir/train.bin and data_dir/eval.bin.

    Uses streaming=True so only the documents actually needed are fetched —
    no need to download the full ~27 GB dataset when you only need 320M tokens.

    Args:
        data_dir         : directory to write the .bin files
        tokenizer_name   : HuggingFace tokenizer identifier
        num_proc         : parallel tokenisation workers (unused in streaming
                           mode — tokenisation is done in the main thread to
                           avoid multiprocessing + streaming conflicts)
        chunk_size       : documents buffered before flushing to disk
        max_train_tokens : stop after this many train tokens (None = full dataset)
        force            : re-create files even if they already exist

    Returns:
        (train_path, eval_path)
    """
    from datasets import load_dataset

    data_dir   = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.bin"
    eval_path  = data_dir / "eval.bin"

    if train_path.exists() and eval_path.exists() and not force:
        train_tok = len(np.memmap(train_path, dtype=DTYPE, mode="r"))
        eval_tok  = len(np.memmap(eval_path,  dtype=DTYPE, mode="r"))
        if max_train_tokens is None or train_tok >= max_train_tokens:
            print(
                f"[fineweb_loader] Cache already exists — "
                f"train: {train_tok/1e6:.0f}M tok, eval: {eval_tok/1e6:.0f}M tok.\n"
                f"  Pass force=True to re-build.",
                flush=True,
            )
            return train_path, eval_path
        print(
            f"[fineweb_loader] Cache has {train_tok/1e6:.0f}M train tokens "
            f"but {max_train_tokens/1e6:.0f}M requested — rebuilding.",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    eos_id    = tokenizer.eos_token_id or 0

    budget_str = f"{max_train_tokens/1e6:.0f}M" if max_train_tokens else "full dataset"
    print(
        f"[fineweb_loader] Streaming {FINEWEB_REPO}/{FINEWEB_SUBSET} "
        f"(target: {budget_str} train tokens) …",
        flush=True,
    )

    def _stream_shard(out_path: Path, label: str,
                      skip: int, max_docs: Optional[int],
                      max_tokens: Optional[int]):
        """Stream, tokenise in chunks, write to out_path. Returns tokens written."""
        total    = 0
        docs_seen = 0
        buf: list[str] = []

        def _flush(buf, max_remaining):
            ids = _tokenize_batch((buf, tokenizer_name, eos_id))
            if max_remaining is not None:
                ids = ids[:max_remaining]
            arr = np.array(ids, dtype=DTYPE)
            arr.tofile(f)
            return len(arr)

        max_tok_str = f"{max_tokens/1e6:.0f}M" if max_tokens else "∞"
        max_doc_str = f"{max_docs:,}" if max_docs else "∞"
        print(
            f"  [{label}] writing (max_docs={max_doc_str}, max_tokens={max_tok_str}) …",
            flush=True,
        )

        with open(out_path, "wb") as f:
            ds = load_dataset(
                FINEWEB_REPO, name=FINEWEB_SUBSET, split="train", streaming=True
            )
            for abs_i, example in enumerate(ds):
                if abs_i < skip:
                    continue

                buf.append(example["text"])
                docs_seen += 1

                # flush when buffer is full or we've hit the doc limit
                at_doc_limit = (max_docs is not None and docs_seen >= max_docs)
                if len(buf) >= chunk_size or at_doc_limit:
                    remaining = None if max_tokens is None else max_tokens - total
                    written   = _flush(buf, remaining)
                    buf       = []
                    total    += written
                    print(
                        f"  [{label}] {docs_seen:,} docs | "
                        f"{total/1e6:.1f}M/{max_tok_str} tokens",
                        flush=True,
                    )
                    if at_doc_limit or (max_tokens and total >= max_tokens):
                        break

            # flush any remainder
            if buf and (max_tokens is None or total < max_tokens):
                remaining = None if max_tokens is None else max_tokens - total
                total    += _flush(buf, remaining)

        print(
            f"  [{label}] Done — {docs_seen:,} docs, "
            f"{total/1e6:.1f}M tokens → {out_path}",
            flush=True,
        )
        return total

    _stream_shard(eval_path,  "eval",  skip=0,
                  max_docs=EVAL_DOCS, max_tokens=None)
    _stream_shard(train_path, "train", skip=EVAL_DOCS,
                  max_docs=None,      max_tokens=max_train_tokens)

    return train_path, eval_path


# ---------------------------------------------------------------------------
# 3.  OLM-NATIVE STREAMING LOADER  (second preference)
# ---------------------------------------------------------------------------


def _build_olm_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    batch_size: int,
    num_workers: int,
    distributed: bool,
):
    from olm.data.datasets import HuggingFaceTextDataset
    from olm.data.datasets import DataLoader as OLMDataLoader

    def _make_hf_dataset(skip=None, take=None, shuffle=False):
        common = dict(
            name=FINEWEB_SUBSET,
            tokenizer=tokenizer,
            context_length=seq_len,
            streaming=True,
            shuffle=shuffle,
        )
        if skip is not None:
            common["skip"] = skip
        if take is not None:
            common["take"] = take
        for path_kwarg in ("path", "repo_id", "dataset_path", "dataset"):
            try:
                return HuggingFaceTextDataset(**{path_kwarg: FINEWEB_REPO}, **common)
            except TypeError:
                continue
        raise RuntimeError(
            "Could not construct OLM HuggingFaceTextDataset: none of "
            "(path, repo_id, dataset_path, dataset) were accepted."
        )

    train_ds = _make_hf_dataset(skip=EVAL_DOCS, shuffle=True)
    eval_ds = _make_hf_dataset(take=EVAL_DOCS, shuffle=False)

    train_dl = OLMDataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=distributed,
    )
    eval_dl = OLMDataLoader(
        eval_ds, batch_size=batch_size, num_workers=num_workers, distributed=False
    )
    return train_dl, eval_dl


# ---------------------------------------------------------------------------
# 4.  HAND-ROLLED STREAMING FALLBACK  (last resort)
# ---------------------------------------------------------------------------


def _doc_stream(
    skip: int = 0,
    take: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(FINEWEB_REPO, name=FINEWEB_SUBSET, split="train", streaming=True)
    yielded = 0
    shard_i = 0
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
    def __init__(self, tokenizer, seq_len, rank=0, world_size=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            eff_rank = self.rank * worker_info.num_workers + worker_info.id
            eff_ws = self.world_size * worker_info.num_workers
        else:
            eff_rank = self.rank
            eff_ws = self.world_size
        while True:
            yield from _tokenize_and_pack(
                _doc_stream(skip=EVAL_DOCS, rank=eff_rank, world_size=eff_ws),
                self.tokenizer,
                self.seq_len,
            )


class _FallbackEvalDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
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
    batch_size: int = 16,
    eval_batches: int = 64,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    distributed: bool = False,
    data_dir: Optional[str | Path] = None,
):
    """
    Build (train_dataloader, eval_dataloader) for FineWeb sample-10BT.

    Loading priority:
      1. Local binary cache (data_dir/train.bin + eval.bin) — fastest
      2. OLM HuggingFaceTextDataset                         — fast streaming
      3. Hand-rolled HF streaming                           — slow fallback

    Args:
        tokenizer  : Pythia GPT-NeoX tokenizer
        seq_len    : tokens per sequence
        batch_size : sequences per batch *per GPU*
        num_workers: DataLoader workers per rank
                     (4 is fine for local bin; use 0 for streaming fallback)
        rank       : this process's global rank
        world_size : total number of processes
        distributed: enable DistributedSampler (True when using torchrun)
        data_dir   : path to directory containing train.bin / eval.bin;
                     if None or files missing, falls back to streaming
    """
    is_main = rank == 0

    # ── 1. Local binary cache ────────────────────────────────────────────────
    if data_dir is not None:
        data_dir = Path(data_dir)
        train_path = data_dir / "train.bin"
        eval_path = data_dir / "eval.bin"
        if train_path.exists() and eval_path.exists():
            if is_main:
                tok_count = len(np.memmap(train_path, dtype=DTYPE, mode="r"))
                print(
                    f"[fineweb_loader] Using local binary cache "
                    f"({tok_count/1e9:.2f}B train tokens) — "
                    f"num_workers={num_workers}",
                    flush=True,
                )
            return _build_local_dataloaders(
                data_dir,
                seq_len,
                batch_size,
                num_workers,
                distributed,
            )
        elif is_main:
            print(
                f"[fineweb_loader] data_dir={data_dir} set but .bin files "
                f"not found — run prepare_dataset() first. Falling back to streaming.",
                flush=True,
            )

    # ── 2. OLM-native streaming ──────────────────────────────────────────────
    try:
        loaders = _build_olm_dataloaders(
            tokenizer=tokenizer,
            seq_len=seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
            distributed=distributed,
        )
        if is_main:
            print("[fineweb_loader] Using OLM HuggingFaceTextDataset.", flush=True)
        return loaders
    except Exception as olm_err:
        if is_main:
            print(
                f"[fineweb_loader] OLM unavailable ({olm_err}); "
                f"falling back to hand-rolled streaming (slow — "
                f"run prepare_dataset() to fix this).",
                flush=True,
            )

    # ── 3. Hand-rolled streaming fallback ────────────────────────────────────
    # Force num_workers=0 for streaming: multiple workers re-stream the same
    # shard independently (bug fixed above), but 0 is still safer and simpler.
    stream_workers = 0
    if is_main and num_workers > 0:
        print(
            f"[fineweb_loader] Overriding num_workers={num_workers}→0 "
            f"for streaming fallback.",
            flush=True,
        )

    train_ds = _FallbackTrainDataset(
        tokenizer, seq_len, rank=rank, world_size=world_size
    )
    eval_ds = _FallbackEvalDataset(tokenizer, seq_len)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, num_workers=stream_workers, pin_memory=True
    )
    eval_dl = DataLoader(
        eval_ds, batch_size=batch_size, num_workers=stream_workers, pin_memory=True
    )
    return train_dl, eval_dl


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def estimate_total_batches(
    target_tokens: int,
    seq_len: int,
    batch_size: int,
    world_size: int = 1,
) -> int:
    tokens_per_global_step = batch_size * world_size * seq_len
    return (target_tokens + tokens_per_global_step - 1) // tokens_per_global_step


# ---------------------------------------------------------------------------
# CLI  — python -m experiments.chinchilla.fineweb_loader --data_dir /data/fw
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Pre-tokenise FineWeb sample-10BT to local binary files."
    )
    p.add_argument(
        "--data_dir", required=True, help="Output directory for train.bin / eval.bin"
    )
    p.add_argument("--tokenizer", default="EleutherAI/pythia-70m")
    p.add_argument(
        "--num_proc", type=int, default=mp.cpu_count(),
        help="Tokenisation workers (default: all CPUs)",
    )
    p.add_argument(
        "--max_train_tokens", type=int, default=None,
        help="Stop after this many train tokens, e.g. 320000000. "
             "Omit to tokenise the full ~10B-token dataset.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-build even if .bin files already exist",
    )
    args = p.parse_args()

    train_p, eval_p = prepare_dataset(
        data_dir         = args.data_dir,
        tokenizer_name   = args.tokenizer,
        num_proc         = args.num_proc,
        max_train_tokens = args.max_train_tokens,
        force            = args.force,
    )
    print(f"\nDone.\n  train → {train_p}\n  eval  → {eval_p}")
