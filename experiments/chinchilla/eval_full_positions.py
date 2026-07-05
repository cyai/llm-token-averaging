"""
Full-position (offset-ensemble) evaluation for token-averaging models.

Motivation
----------
A k=1 model predicts every token; a k-averaged model trained with disjoint
windows only predicts every k-th token (the first token of each next window).
Their eval losses are therefore not directly comparable: the averaged model
assigns no probability to (k-1)/k of the positions.

This script closes that gap with an *offset ensemble*: the averaged model is
run k times per sequence, shifting the window grid by o = 0..k-1 tokens.
Offset o predicts token positions o+k, o+2k, ... — so across all k passes,
(almost) every position in the sequence is predicted exactly once, each
conditioned on its full compressed prefix.  The combined per-token NLL is a
genuine full-sequence language-modelling loss, directly comparable to k=1.

For an exactly-equal comparison, the k=1 baseline is additionally scored
restricted to the identical position set covered by the ensemble.

Both models are evaluated on the SAME sequences from eval.bin, in the same
deterministic order.

NOTE: checkpoints trained with a fixed offset-0 window grid have never seen
offsets > 0, so those passes are off-distribution.  The per-offset breakdown
printed by this script quantifies exactly how much that costs.

Usage
-----
    # side-by-side 125M comparison (recommended: one invocation, same data)
    python experiments/chinchilla/eval_full_positions.py \
        --models model1_125m avg_125m_k2 \
        --ckpts  experiments/chinchilla/results/model1_125m/checkpoints/final.pt \
                 experiments/chinchilla/results/avg_125m_k2/checkpoints/final.pt \
        --data_dir /data/fineweb --seq_len 1024 --batch_size 16

    # start in background:
    nohup python experiments/chinchilla/eval_full_positions.py \
        --models model1_125m avg_125m_k2 \
        --ckpts  experiments/chinchilla/results/model1_125m/checkpoints/final.pt \
                 experiments/chinchilla/results/avg_125m_k2/checkpoints/final.pt \
        --data_dir /data/fineweb --seq_len 1024 --batch_size 16 \
        --out results/eval_full_positions.json > results/eval_full_positions.log 2>&1 &

    # single model
    python experiments/chinchilla/eval_full_positions.py \
        --models avg_125m_k2 --data_dir /data/fineweb

Checkpoint format (see train.py): torch.save dict with keys
    step, tokens_seen, cumulative_flops, model, optimizer, scheduler
where state["model"] is the raw module state dict:
    k=1 : OLMTransformerBody keys        (embed_in.* / body.* / embed_out.*)
    k>1 : OLMAveragedLanguageModel keys  (backbone.embed_in.* / ...)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.chinchilla.model_configs import get_config, ModelConfig
from experiments.chinchilla.fineweb_loader import DTYPE
from experiments.shared.olm_model import OLMTransformerBody


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg: ModelConfig, ckpt_path: Path, vocab_size: int,
               device: str) -> OLMTransformerBody:
    """
    Build an OLMTransformerBody and load checkpoint weights into it.

    We always build UNTIED and load the raw state dict: a tied checkpoint
    stores the shared weight under both the embed_in and embed_out keys, so
    loading untied reproduces it exactly.  (Building tied and then loading an
    untied checkpoint would silently overwrite the input embedding with the
    LM head weight — so we never tie here.)
    """
    backbone = OLMTransformerBody(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        context_length=cfg.context_len,
    )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state

    # Averaged models are saved as OLMAveragedLanguageModel → strip "backbone."
    if any(key.startswith("backbone.") for key in sd):
        sd = {k[len("backbone."):]: v for k, v in sd.items()
              if k.startswith("backbone.")}

    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys loading {ckpt_path}: {missing}")
    if unexpected:
        print(f"  [warn] ignored unexpected keys: {unexpected}", flush=True)

    meta = {}
    if isinstance(state, dict):
        meta = {k: state[k] for k in ("step", "tokens_seen") if k in state}
    print(f"  loaded {ckpt_path}  {meta}", flush=True)

    backbone.to(device)
    backbone.eval()
    return backbone


# ---------------------------------------------------------------------------
# Per-position NLL computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def nll_baseline(backbone: OLMTransformerBody,
                 ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    k=1 forward.  Returns (nll [B, T-1], positions [T-1]) where positions[j]
    is the 0-based index of the token being predicted (1..T-1).
    """
    logits = backbone(ids)                       # [B, T, V]
    logits = logits[:, :-1]                      # predict ids[:, 1:]
    labels = ids[:, 1:]
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        reduction="none",
    ).view(ids.size(0), -1)                      # [B, T-1]
    positions = torch.arange(1, ids.size(1), device=ids.device)
    return nll, positions


@torch.no_grad()
def nll_averaged_offset(backbone: OLMTransformerBody, ids: torch.Tensor,
                        k: int, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One offset pass of the k-averaged model.

    Window grid starts at `offset`: window j covers raw positions
    [offset + j*k, offset + (j+1)*k - 1].  Position j of the transformer
    output predicts the first token of window j+1, i.e. raw position
    offset + (j+1)*k  — matching the training convention
    (labels = input_ids[:, k::k]) exactly when offset == 0.

    Returns (nll [B, P], positions [P]) with positions = offset+k, offset+2k, ...
    """
    ids_o = ids[:, offset:]
    n = ids_o.size(1) // k                       # number of complete windows
    ids_o = ids_o[:, : n * k]

    hidden = backbone.embed_in(ids_o)            # [B, n*k, D]
    B, _, D = hidden.shape
    avg = hidden.reshape(B, n, k, D).mean(dim=2) # [B, n, D]

    out = backbone.body(avg[:, :-1])             # [B, n-1, D]
    logits = backbone.embed_out(out)             # [B, n-1, V]

    labels = ids[:, offset + k :: k][:, : logits.size(1)]   # [B, <= n-1]
    logits = logits[:, : labels.size(1)]

    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        reduction="none",
    ).view(B, -1)                                # [B, P]
    positions = offset + k + k * torch.arange(labels.size(1), device=ids.device)
    return nll, positions


# ---------------------------------------------------------------------------
# Evaluation loops
# ---------------------------------------------------------------------------

def _autocast(device: str):
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                          enabled=(device_type == "cuda"))


@torch.no_grad()
def evaluate_model(backbone: OLMTransformerBody, k: int, batches,
                   device: str, seq_len: int) -> dict:
    """
    Evaluate one model over the eval batches.

    k == 1 : single pass, per-position NLL for positions 1..T-1.
    k >= 2 : k offset passes; combined they cover positions k..~T-1.

    Returns a dict with summed NLL / counts per position (so a baseline can
    later be restricted to any position subset) and per-offset stats.
    """
    pos_nll_sum = torch.zeros(seq_len, dtype=torch.float64)   # index = predicted position
    pos_count   = torch.zeros(seq_len, dtype=torch.float64)
    offset_sums   = {o: 0.0 for o in range(k)} if k > 1 else {}
    offset_counts = {o: 0   for o in range(k)} if k > 1 else {}

    n_seqs = 0
    for ids in batches:
        ids = ids.to(device)
        n_seqs += ids.size(0)
        with _autocast(device):
            if k == 1:
                nll, positions = nll_baseline(backbone, ids)
                pos_nll_sum[positions.cpu()] += nll.sum(dim=0).double().cpu()
                pos_count[positions.cpu()]   += ids.size(0)
            else:
                for o in range(k):
                    nll, positions = nll_averaged_offset(backbone, ids, k, o)
                    pos_nll_sum[positions.cpu()] += nll.sum(dim=0).double().cpu()
                    pos_count[positions.cpu()]   += ids.size(0)
                    offset_sums[o]   += nll.sum().item()
                    offset_counts[o] += nll.numel()

    return {
        "k": k,
        "n_seqs": n_seqs,
        "pos_nll_sum": pos_nll_sum,
        "pos_count": pos_count,
        "offset_sums": offset_sums,
        "offset_counts": offset_counts,
    }


def summarize(res: dict, restrict_to: torch.Tensor | None = None) -> dict:
    """Mean NLL (and ppl) over all covered positions, optionally restricted."""
    covered = res["pos_count"] > 0
    if restrict_to is not None:
        covered = covered & restrict_to
    total_nll = res["pos_nll_sum"][covered].sum().item()
    total_cnt = res["pos_count"][covered].sum().item()
    mean = total_nll / max(total_cnt, 1)
    return {
        "mean_nll": mean,
        "ppl": float(np.exp(mean)),
        "n_predictions": int(total_cnt),
        "n_positions": int(covered.sum().item()),
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def eval_batches(data_dir: Path | None, tokenizer_name: str, seq_len: int,
                 batch_size: int, max_batches: int | None):
    """
    Yield eval batches deterministically.  Prefers {data_dir}/eval.bin
    (memmap, exact same data every run); falls back to HF streaming of the
    same first-5000-docs eval slice used in training.
    """
    if data_dir is not None and (Path(data_dir) / "eval.bin").exists():
        data = np.memmap(Path(data_dir) / "eval.bin", dtype=DTYPE, mode="r")
        n_seqs = (len(data) - 1) // seq_len
        print(f"[eval] eval.bin: {len(data)/1e6:.1f}M tokens → {n_seqs} seqs "
              f"of {seq_len}", flush=True)
        count = 0
        for start in range(0, n_seqs, batch_size):
            idx = range(start, min(start + batch_size, n_seqs))
            chunk = np.stack([
                data[i * seq_len:(i + 1) * seq_len].astype(np.int64) for i in idx
            ])
            yield torch.from_numpy(chunk)
            count += 1
            if max_batches is not None and count >= max_batches:
                return
    else:
        print("[eval] no eval.bin — falling back to HF streaming "
              "(same first-5000-doc eval slice as training)", flush=True)
        from experiments.chinchilla.fineweb_loader import _FallbackEvalDataset
        from torch.utils.data import DataLoader
        tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        ds = _FallbackEvalDataset(tok, seq_len)
        dl = DataLoader(ds, batch_size=batch_size, num_workers=0)
        for i, batch in enumerate(dl):
            if max_batches is not None and i >= max_batches:
                return
            yield batch if isinstance(batch, torch.Tensor) else batch["input_ids"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Offset-ensemble (full-position) evaluation for "
                    "token-averaging models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--models", nargs="+", required=True,
                   help="Model config names, e.g. model1_125m avg_125m_k2. "
                        "When a k=1 and a k>1 model are both given, the k=1 "
                        "model is also scored restricted to the k>1 model's "
                        "covered positions.")
    p.add_argument("--ckpts", nargs="+", default=None,
                   help="Checkpoint paths, parallel to --models. Defaults to "
                        "results/<name>/checkpoints/final.pt")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Directory containing eval.bin (recommended).")
    p.add_argument("--seq_len", type=int, default=1024,
                   help="Raw tokens per eval sequence. Must match training "
                        "(1024 for the current 50M/125M runs).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_batches", type=int, default=None,
                   help="Limit eval batches (default: all of eval.bin).")
    p.add_argument("--tokenizer_name", type=str, default="EleutherAI/pythia-70m")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None,
                   help="Optional path to write results as JSON.")
    args = p.parse_args()

    if args.ckpts is not None and len(args.ckpts) != len(args.models):
        p.error("--ckpts must have one path per --models entry")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    results = {}
    for i, name in enumerate(args.models):
        cfg = get_config(name)
        ckpt = Path(args.ckpts[i]) if args.ckpts else (
            _ROOT / "experiments" / "chinchilla" / "results" / cfg.name
            / "checkpoints" / "final.pt"
        )
        if not ckpt.exists():
            raise FileNotFoundError(
                f"No checkpoint at {ckpt} — pass --ckpts explicitly.")

        print(f"\n[{name}] k={cfg.averaging_k}  "
              f"d={cfg.d_model} h={cfg.n_heads} l={cfg.n_layers}", flush=True)
        model = load_model(cfg, ckpt, vocab_size, args.device)

        batches = eval_batches(
            Path(args.data_dir) if args.data_dir else None,
            args.tokenizer_name, args.seq_len, args.batch_size,
            args.max_batches,
        )
        res = evaluate_model(model, cfg.averaging_k, batches,
                             args.device, args.seq_len)
        results[name] = res

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 72)
    print("RESULTS  (mean NLL in nats/token; identical eval sequences)")
    print("=" * 72)

    avg_names = [n for n in results if results[n]["k"] > 1]
    base_names = [n for n in results if results[n]["k"] == 1]

    report = {}
    for name, res in results.items():
        s = summarize(res)
        report[name] = {"all_covered": s}
        print(f"\n{name}  (k={res['k']}, {res['n_seqs']} seqs)")
        print(f"  covered positions : {s['n_positions']} / {args.seq_len}"
              f"  ({s['n_predictions']:,} predictions)")
        print(f"  mean NLL          : {s['mean_nll']:.4f}   ppl {s['ppl']:.2f}")
        if res["k"] > 1:
            for o in sorted(res["offset_sums"]):
                m = res["offset_sums"][o] / max(res["offset_counts"][o], 1)
                report[name][f"offset_{o}"] = m
                tag = "  (training offset)" if o == 0 else \
                      "  (off-distribution for offset-0-trained ckpts)"
                print(f"  offset {o} NLL      : {m:.4f}{tag}")

    # k=1 restricted to the averaged model's covered positions → exact
    # apples-to-apples target sets.
    for bname in base_names:
        for aname in avg_names:
            mask = results[aname]["pos_count"] > 0
            s = summarize(results[bname], restrict_to=mask)
            report[bname][f"restricted_to_{aname}"] = s
            print(f"\n{bname} restricted to positions covered by {aname}:")
            print(f"  mean NLL          : {s['mean_nll']:.4f}   ppl {s['ppl']:.2f}"
                  f"   ({s['n_predictions']:,} predictions)")

    if args.out:
        def _clean(d):
            return {k: (_clean(v) if isinstance(v, dict) else v)
                    for k, v in d.items()}
        Path(args.out).write_text(json.dumps(_clean(report), indent=2))
        print(f"\nJSON written to {args.out}")


if __name__ == "__main__":
    main()
