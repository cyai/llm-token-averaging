"""
FineWeb data loader for Chinchilla-scale training.

Loading priority
----------------
1. Local binary cache  (fastest — numpy memmap, zero network overhead)
2. OLM HuggingFaceTextDataset  (fast — library-managed streaming)
3. Hand-rolled HF streaming  (slowest — one shard at a time)

Pre-tokenise once (recommended before any multi-GPU run):

    python -m experiments.chinchilla.fineweb_loader \
        --data_dir /data/fineweb \
        --tokenizer EleutherAI/pythia-70m \
        --max_train_tokens 600_000_000 \
        --num_proc 16 \
        --dl_workers 8

How it works
------------
- Lists all parquet shards for sample-10BT on the HF Hub
- Downloads exactly the shards needed (in parallel threads)
- Tokenises each shard in parallel with a multiprocessing pool
- Splits the result into eval.bin (first EVAL_DOCS docs) and train.bin

This avoids streaming one-shard-at-a-time: for 600M tokens you download
~7 shards in parallel instead of waiting for sequential HTTP fetches.
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

FINEWEB_REPO   = "HuggingFaceFW/fineweb"
FINEWEB_SUBSET = "sample-10BT"

EVAL_DOCS = 5_000        # first N docs reserved for eval
DTYPE     = np.uint16    # GPT-NeoX vocab 50 257 fits in uint16


# ---------------------------------------------------------------------------
# 1.  LOCAL BINARY CACHE  (fastest path)
# ---------------------------------------------------------------------------

class _LocalBinDataset(Dataset):
    """
    Map-style dataset backed by a pre-tokenised flat binary file.
    numpy memmap means multiple DataLoader workers can read concurrently
    with zero data duplication or extra memory.
    """

    def __init__(self, path: Path, seq_len: int):
        self.seq_len = seq_len
        self.data    = np.memmap(path, dtype=DTYPE, mode="r")
        self.n_seqs  = (len(self.data) - 1) // seq_len

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
    from torch.utils.data import DistributedSampler

    train_ds = _LocalBinDataset(data_dir / "train.bin", seq_len)
    eval_ds  = _LocalBinDataset(data_dir / "eval.bin",  seq_len)

    train_sampler = (
        DistributedSampler(train_ds, shuffle=True, drop_last=True)
        if distributed else None
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
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
    )
    return train_dl, eval_dl


# ---------------------------------------------------------------------------
# 2.  DATASET PREPARATION — parallel download + multiprocess tokenisation
# ---------------------------------------------------------------------------

def _list_shard_paths() -> list[str]:
    """Return sorted parquet shard paths for sample-10BT on HF Hub."""
    from huggingface_hub import list_repo_tree
    paths = []
    for item in list_repo_tree(FINEWEB_REPO, repo_type="dataset", recursive=True):
        p = item.path if hasattr(item, "path") else str(item)
        if "10BT" in p and p.endswith(".parquet"):
            paths.append(p)
    return sorted(paths)


def _download_shards_parallel(
    shard_paths: list[str],
    local_dir: Path,
    max_workers: int = 8,
) -> list[Path]:
    """Download HF parquet shards in parallel. Returns local paths in order."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    def _dl(shard_path):
        local = hf_hub_download(
            repo_id=FINEWEB_REPO,
            repo_type="dataset",
            filename=shard_path,
            local_dir=str(local_dir),
        )
        return shard_path, Path(local)

    print(
        f"  Downloading {len(shard_paths)} shard(s) with {max_workers} threads …",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_dl, p): p for p in shard_paths}
        for i, fut in enumerate(as_completed(futures), 1):
            sp, lp = fut.result()
            results[sp] = lp
            print(f"  [{i}/{len(shard_paths)}] ✓ {sp}", flush=True)

    return [results[p] for p in shard_paths]


def _tokenize_parquet_to_file(args) -> tuple[str, int]:
    """
    Top-level picklable worker: tokenise one parquet shard and write
    tokens directly to a .bin temp file beside the parquet.

    Returns (tmp_path, token_count) — no large data crosses the IPC pipe.
    """
    parquet_path, tokenizer_name, eos_id = args
    import pyarrow.parquet as pq

    out_path = str(parquet_path) + ".tok.bin"
    tok   = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    table = pq.read_table(parquet_path, columns=["text"])
    texts = table.column("text").to_pylist()
    total = 0
    CHUNK = 10_000   # docs per flush to keep memory low

    with open(out_path, "wb") as f:
        buf: list[int] = []
        for i, text in enumerate(texts):
            if text and text.strip():
                buf.extend(tok.encode(text, add_special_tokens=False))
                buf.append(eos_id)
            if len(buf) >= CHUNK * 256 or i == len(texts) - 1:
                arr = np.array(buf, dtype=np.uint16)
                arr.tofile(f)
                total += len(arr)
                buf   = []

    return out_path, total


def prepare_dataset(
    data_dir: str | Path,
    tokenizer_name: str = "EleutherAI/pythia-70m",
    num_proc: int = 8,
    max_train_tokens: Optional[int] = None,
    dl_workers: int = 8,
    force: bool = False,
) -> tuple[Path, Path]:
    """
    Download only the FineWeb parquet shards needed, tokenise in parallel,
    and write flat uint16 binary files (train.bin / eval.bin).

    For 600M tokens this downloads ~7 shards in parallel (~2-5 min)
    instead of streaming 600M tokens one-by-one (~30 min).

    Args:
        data_dir         : output directory
        tokenizer_name   : HuggingFace tokenizer id
        num_proc         : tokenisation worker processes
        max_train_tokens : token budget for train.bin (None = full 10B)
        dl_workers       : parallel shard download threads
        force            : rebuild even if .bin files already exist
    """
    import pyarrow.parquet as pq

    data_dir   = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.bin"
    eval_path  = data_dir / "eval.bin"

    if train_path.exists() and eval_path.exists() and not force:
        train_tok = len(np.memmap(train_path, dtype=DTYPE, mode="r"))
        eval_tok  = len(np.memmap(eval_path,  dtype=DTYPE, mode="r"))
        if max_train_tokens is None or train_tok >= max_train_tokens:
            print(
                f"[fineweb_loader] Cache exists — "
                f"train: {train_tok/1e6:.0f}M tok, eval: {eval_tok/1e6:.0f}M tok.\n"
                f"  Pass --force to rebuild.",
                flush=True,
            )
            return train_path, eval_path
        print(
            f"[fineweb_loader] Cache only has {train_tok/1e6:.0f}M train tokens "
            f"({max_train_tokens/1e6:.0f}M requested) — rebuilding.",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    eos_id    = tokenizer.eos_token_id or 0

    # ── Step 1: list shards ──────────────────────────────────────────────────
    print("[fineweb_loader] Listing shards on HF Hub …", flush=True)
    all_shards = _list_shard_paths()
    if not all_shards:
        raise RuntimeError(
            f"No parquet files found for {FINEWEB_REPO}/{FINEWEB_SUBSET}. "
            "Check your HF credentials and dataset name."
        )
    print(f"  Found {len(all_shards)} shards.", flush=True)

    # ── Step 2: decide how many shards to fetch ──────────────────────────────
    # FineWeb sample-10BT has ~100M tokens/shard.
    # +2 buffer to avoid off-by-one at boundary.
    TOKENS_PER_SHARD_EST = 100_000_000
    if max_train_tokens is None:
        n_shards = len(all_shards)
    else:
        n_shards = min(len(all_shards),
                       max_train_tokens // TOKENS_PER_SHARD_EST + 2)

    budget_str = f"{max_train_tokens/1e6:.0f}M" if max_train_tokens else "all"
    print(
        f"[fineweb_loader] Need {n_shards} shard(s) for {budget_str} train tokens.",
        flush=True,
    )

    # ── Step 3: parallel download ────────────────────────────────────────────
    raw_dir     = data_dir / "raw"
    local_files = _download_shards_parallel(
        all_shards[:n_shards], raw_dir, max_workers=dl_workers
    )

    # ── Step 4: parallel tokenisation → temp .bin files (no IPC bottleneck) ──
    print(
        f"[fineweb_loader] Tokenising {len(local_files)} shard(s) "
        f"with {num_proc} workers …",
        flush=True,
    )
    jobs = [(str(f), tokenizer_name, eos_id) for f in local_files]
    tok_files: list[Path] = []
    tok_counts: list[int] = []
    with mp.Pool(num_proc) as pool:
        for i, (tmp_path, n_tok) in enumerate(
            pool.imap(_tokenize_parquet_to_file, jobs), 1
        ):
            tok_files.append(Path(tmp_path))
            tok_counts.append(n_tok)
            print(f"  [{i}/{len(jobs)}] {n_tok/1e6:.1f}M tokens → {tmp_path}", flush=True)

    # ── Step 5: doc-level eval/train split, concat temp files → .bin ─────────
    print("[fineweb_loader] Splitting eval/train and writing .bin files …", flush=True)

    # Re-tokenise only the first EVAL_DOCS docs from shard 0 → eval.bin.
    # We stream shard 0 in batches so we never load the full 2 GB parquet.
    # The pre-tokenised tok_files[0] is used for train (skipping eval tokens).
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    pf0 = pq.ParquetFile(local_files[0])

    eval_ids:  list[int] = []
    docs_seen: int = 0
    for batch in pf0.iter_batches(batch_size=1_000, columns=["text"]):
        for text in batch.column("text").to_pylist():
            if docs_seen >= EVAL_DOCS:
                break
            if text and text.strip():
                eval_ids.extend(tok.encode(text, add_special_tokens=False) + [eos_id])
                docs_seen += 1
        if docs_seen >= EVAL_DOCS:
            break

    eval_token_count = len(eval_ids)
    np.array(eval_ids, dtype=DTYPE).tofile(eval_path)
    print(f"  eval  → {eval_path}  ({eval_token_count/1e6:.1f}M tokens)", flush=True)
    del eval_ids

    # Write train.bin: shard 0 from tok_files[0] (skip eval tokens at head),
    # then remaining shards in full until max_train_tokens is reached.
    with open(train_path, "wb") as out_f:
        total_train = 0
        for shard_i, (tok_f, n_tok) in enumerate(zip(tok_files, tok_counts)):
            data = np.fromfile(tok_f, dtype=DTYPE)
            if shard_i == 0:
                data = data[eval_token_count:]   # skip the eval head
            remaining = (
                max_train_tokens - total_train
                if max_train_tokens else len(data)
            )
            data = data[:remaining]
            out_f.write(data.tobytes())
            total_train += len(data)
            print(
                f"  shard {shard_i}: +{len(data)/1e6:.1f}M tokens "
                f"(total {total_train/1e6:.1f}M)",
                flush=True,
            )
            if max_train_tokens and total_train >= max_train_tokens:
                break

    # Clean up temp token files
    for tok_f in tok_files:
        tok_f.unlink(missing_ok=True)

    print(
        f"[fineweb_loader] Done.\n"
        f"  eval  → {eval_path}  ({eval_token_count/1e6:.1f}M tokens)\n"
        f"  train → {train_path}  ({total_train/1e6:.1f}M tokens)",
        flush=True,
    )
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
            name=FINEWEB_SUBSET, tokenizer=tokenizer,
            context_length=seq_len, streaming=True, shuffle=shuffle,
        )
        if skip is not None: common["skip"] = skip
        if take is not None: common["take"] = take
        for kwarg in ("path", "repo_id", "dataset_path", "dataset"):
            try:
                return HuggingFaceTextDataset(**{kwarg: FINEWEB_REPO}, **common)
            except TypeError:
                continue
        raise RuntimeError("OLM HuggingFaceTextDataset: no valid path kwarg found.")

    train_dl = OLMDataLoader(
        _make_hf_dataset(skip=EVAL_DOCS, shuffle=True),
        batch_size=batch_size, num_workers=num_workers, distributed=distributed,
    )
    eval_dl = OLMDataLoader(
        _make_hf_dataset(take=EVAL_DOCS, shuffle=False),
        batch_size=batch_size, num_workers=num_workers, distributed=False,
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
    ds      = load_dataset(FINEWEB_REPO, name=FINEWEB_SUBSET,
                           split="train", streaming=True)
    yielded = 0
    shard_i = 0
    for abs_i, ex in enumerate(ds):
        if abs_i < skip:
            continue
        if shard_i % world_size == rank:
            yield ex["text"]
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
        self.tokenizer  = tokenizer
        self.seq_len    = seq_len
        self.rank       = rank
        self.world_size = world_size

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        if wi is not None:
            eff_rank = self.rank * wi.num_workers + wi.id
            eff_ws   = self.world_size * wi.num_workers
        else:
            eff_rank = self.rank
            eff_ws   = self.world_size
        while True:
            yield from _tokenize_and_pack(
                _doc_stream(skip=EVAL_DOCS, rank=eff_rank, world_size=eff_ws),
                self.tokenizer, self.seq_len,
            )


class _FallbackEvalDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len   = seq_len

    def __iter__(self):
        yield from _tokenize_and_pack(
            _doc_stream(skip=0, take=EVAL_DOCS),
            self.tokenizer, self.seq_len,
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

    Priority: local .bin cache → OLM streaming → hand-rolled streaming.

    Pass data_dir= to use the fast local binary cache produced by
    prepare_dataset(). With local cache, num_workers=4 is appropriate.
    For streaming fallbacks, num_workers is overridden to 0.
    """
    is_main = (rank == 0)

    # ── 1. Local binary cache ────────────────────────────────────────────────
    if data_dir is not None:
        data_dir   = Path(data_dir)
        train_path = data_dir / "train.bin"
        eval_path  = data_dir / "eval.bin"
        if train_path.exists() and eval_path.exists():
            if is_main:
                n = len(np.memmap(train_path, dtype=DTYPE, mode="r"))
                print(
                    f"[fineweb_loader] Local cache — "
                    f"{n/1e6:.0f}M train tokens, num_workers={num_workers}",
                    flush=True,
                )
            return _build_local_dataloaders(
                data_dir, seq_len, batch_size, num_workers, distributed,
            )
        elif is_main:
            print(
                f"[fineweb_loader] data_dir={data_dir} set but .bin files missing.\n"
                f"  Run: python -m experiments.chinchilla.fineweb_loader "
                f"--data_dir {data_dir} --max_train_tokens <N>\n"
                f"  Falling back to streaming.",
                flush=True,
            )

    # ── 2. OLM-native streaming ──────────────────────────────────────────────
    try:
        loaders = _build_olm_dataloaders(
            tokenizer=tokenizer, seq_len=seq_len, batch_size=batch_size,
            num_workers=num_workers, distributed=distributed,
        )
        if is_main:
            print("[fineweb_loader] Using OLM HuggingFaceTextDataset.", flush=True)
        return loaders
    except Exception as e:
        if is_main:
            print(
                f"[fineweb_loader] OLM unavailable ({e}); "
                f"hand-rolled streaming (slow — run prepare_dataset() to fix).",
                flush=True,
            )

    # ── 3. Hand-rolled streaming fallback (num_workers forced to 0) ──────────
    if is_main and num_workers > 0:
        print(
            f"[fineweb_loader] Overriding num_workers={num_workers}→0 "
            f"for streaming fallback.",
            flush=True,
        )
    train_ds = _FallbackTrainDataset(tokenizer, seq_len, rank=rank, world_size=world_size)
    eval_ds  = _FallbackEvalDataset(tokenizer, seq_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    eval_dl  = DataLoader(eval_ds,  batch_size=batch_size, num_workers=0, pin_memory=True)
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
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Pre-tokenise FineWeb sample-10BT to local binary files.\n"
                    "Downloads only the shards needed (parallel), then tokenises\n"
                    "in parallel — much faster than streaming.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir",          required=True,
                   help="Output directory for train.bin / eval.bin")
    p.add_argument("--tokenizer",         default="EleutherAI/pythia-70m")
    p.add_argument("--num_proc",          type=int, default=mp.cpu_count(),
                   help="Tokenisation workers (default: all CPUs)")
    p.add_argument("--dl_workers",        type=int, default=8,
                   help="Parallel shard download threads (default: 8)")
    p.add_argument("--max_train_tokens",  type=int, default=None,
                   help="Stop after N train tokens, e.g. 600000000. "
                        "Omit to tokenise the full ~10B-token dataset.")
    p.add_argument("--force",             action="store_true",
                   help="Rebuild even if .bin files already exist")
    args = p.parse_args()

    train_p, eval_p = prepare_dataset(
        data_dir         = args.data_dir,
        tokenizer_name   = args.tokenizer,
        num_proc         = args.num_proc,
        max_train_tokens = args.max_train_tokens,
        dl_workers       = args.dl_workers,
        force            = args.force,
    )
    print(f"\nDone.\n  train → {train_p}\n  eval  → {eval_p}")
