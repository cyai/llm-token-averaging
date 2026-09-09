"""
Offset-ensemble rescoring of Hub-hosted checkpoints.

Why this exists
---------------
Every run before the 500M pair logged only the native offset-0 eval loss. For
k > 1 that scores one token per group, roughly 1/k of the positions, so it is
comparable within a fixed k but not across k. The cross-k numbers in the
paper's Tables 2, 4 and 5 therefore rest on a metric that does not mean the
same thing in each column.

This script fixes that offline. It pulls each final.pt from the Hub, rebuilds
the model exactly as train.py did (including the learned pooling modules), and
scores it with the offset ensemble: the averaged model runs k times with the
window grid shifted by o = 0..k-1, so the union of passes predicts (nearly)
every position exactly once, each conditioned on its full compressed prefix.
The prediction-count-weighted mean is a full-sequence NLL directly comparable
across k.

Difference from eval_full_positions.py
--------------------------------------
That script reimplements the averaging inline as a hardcoded mean, so it
silently produces wrong numbers for the learned and weighted poolers. This one
instantiates the real OLMAveragedLanguageModel with the run's method config and
calls its forward, so it is correct for every pooling rule in the ablation.

Usage
-----
    # everything the paper needs, in dependency order
    python experiments/chinchilla/rescore_hf.py \
        --groups pairs pooling context \
        --data_dir /data/fineweb \
        --out experiments/chinchilla/results/rescore_full_position.json

    # one group, keeping downloads for a rerun
    python experiments/chinchilla/rescore_hf.py --groups pooling \
        --data_dir /data/fineweb --keep_downloads

    # dry run: print the manifest and download sizes, fetch nothing
    python experiments/chinchilla/rescore_hf.py --groups pairs --dry_run
"""

from __future__ import annotations

import argparse
import json
import shutil
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
from experiments.chinchilla.eval_full_positions import eval_batches
from experiments.shared.averaged_lm import build_method_config
from experiments.shared.olm_model import (
    OLMTransformerBody,
    OLMAveragedLanguageModel,
)

HF_PREFIX = "FAIRC/token-averaging-"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
# (config_name, hub_repo_suffix, seq_len, group, note)
#
# seq_len is the RAW sequence length the run trained at, which is also the
# length we must score at: it sets how many raw tokens each eval sequence
# holds and therefore what the loss is an average over.
#
# Deliberately excluded:
#   model1_50m            final.pt on the Hub is a 0-byte broken LFS pointer.
#   avg_50m_k2 (main)     Hub copy is an untied-embedding model from a
#                         different run than the one the paper cites.
#   avg_50m_k8 (main)     same, untied; matched_mean-avg_50m_k8 is the tied one.
# The 50M more-data and more-context arms come from run_50m_pair.sh instead,
# which logs this metric during training.

MANIFEST: list[tuple[str, str, int, str, str]] = [
    # --- the two large more-data pairs -----------------------------------
    ("model1_250m", "model1_250m", 1024, "pairs", "250M k=1"),
    ("avg_250m_k2", "avg_250m_k2", 1024, "pairs", "250M k=2"),
    ("model1_500m", "model1_500m", 1024, "pairs", "500M k=1"),
    ("avg_500m_k2", "avg_500m_k2", 1024, "pairs", "500M k=2"),

    # --- pooling comparison, matched protocol ----------------------------
    ("avg_50m_k2", "matched_mean-avg_50m_k2", 1024, "pooling", "k=2 mean"),
    ("avg_50m_k2_learnable_pos", "avg_50m_k2_learnable_pos", 1024, "pooling",
     "k=2 learned: content + position"),
    ("avg_50m_k2_learnable", "avg_50m_k2_learnable", 1024, "pooling",
     "k=2 learned: content only"),
    ("avg_50m_k2_wexp", "avg_50m_k2_wexp", 1024, "pooling",
     "k=2 fixed recency"),
    ("avg_50m_k2_ov4s2", "avg_50m_k2_ov4s2", 1024, "pooling",
     "k=2 overlapping w4s2"),
    ("avg_50m_k2_word", "avg_50m_k2_word", 1024, "pooling",
     "k=2 word-aligned (see caveat)"),

    ("avg_50m_k4", "matched_mean-avg_50m_k4", 1024, "pooling", "k=4 mean"),
    ("avg_50m_k4_learnable_pos", "avg_50m_k4_learnable_pos", 1024, "pooling",
     "k=4 learned: content + position"),
    ("avg_50m_k4_learnable", "avg_50m_k4_learnable", 1024, "pooling",
     "k=4 learned: content only"),
    ("avg_50m_k4_wexp", "avg_50m_k4_wexp", 1024, "pooling",
     "k=4 fixed recency"),

    ("avg_50m_k8", "matched_mean-avg_50m_k8", 1024, "pooling", "k=8 mean"),
    ("avg_50m_k8_learnable_pos", "avg_50m_k8_learnable_pos", 1024, "pooling",
     "k=8 learned: content + position"),
    ("avg_50m_k8_learnable", "avg_50m_k8_learnable", 1024, "pooling",
     "k=8 learned: content only"),
    ("avg_50m_k8_wexp", "avg_50m_k8_wexp", 1024, "pooling",
     "k=8 fixed recency"),

    # --- the surviving 2048-context arm ----------------------------------
    # Its averaged counterpart is lost, so this only becomes a comparison
    # once run_50m_pair.sh runs 3 and 4 finish. Scored here so the old and
    # new k=1 numbers can be sanity-checked against each other.
    ("model1_50m_tied_2nctx", "model1_50m_tied_2ctx", 2048, "context",
     "50M k=1 full attention @2048"),
]


# ---------------------------------------------------------------------------
# Model construction and loading
# ---------------------------------------------------------------------------

def _strip_prefix(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _checkpoint_is_tied(sd: dict) -> bool:
    """
    Decide tying from the checkpoint itself rather than from the config.

    Some early runs were trained untied and uploaded under a config that now
    says tied. Building tied and loading an untied checkpoint would silently
    overwrite the input embedding with the LM head, so we always trust the
    file.
    """
    bb = _strip_prefix(sd, "backbone.") or sd
    win = bb.get("embed_in.embedding.weight")
    wout = bb.get("embed_out.blocks.1.weight")
    if win is None or wout is None:
        return True                     # only one matrix stored -> tied
    return torch.equal(win, wout)


def build_and_load(cfg: ModelConfig, ckpt_path: Path, vocab_size: int,
                   device: str) -> tuple[torch.nn.Module, dict]:
    """Rebuild the trained model (backbone + pooling) and load its weights."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state

    tied = _checkpoint_is_tied(sd)
    if tied != bool(cfg.tie_embeddings):
        print(f"  [warn] config says tie_embeddings={cfg.tie_embeddings} but the "
              f"checkpoint is {'tied' if tied else 'untied'}; trusting the file. "
              f"This is a different run than the config describes.", flush=True)

    backbone = OLMTransformerBody(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        context_length=cfg.context_len,
    )
    if tied:
        backbone.tie_embedding_weights()

    k = cfg.averaging_k
    if k > 1:
        method_cfg = build_method_config(cfg.method_name or f"uniform_k{k}")
        model = OLMAveragedLanguageModel(backbone, method_cfg)
        pooling_params = 0
        if method_cfg.learnable_module is not None:
            pooling_params = sum(p.numel()
                                 for p in method_cfg.learnable_module.parameters())
        print(f"  pooling: {method_cfg.name} (family={method_cfg.method_family}, "
              f"{pooling_params:,} pooling params)", flush=True)
    else:
        model = backbone

    missing, unexpected = model.load_state_dict(sd, strict=False)
    # A tied checkpoint legitimately lacks a separate LM head entry.
    missing = [m for m in missing if "embed_out.blocks.1.weight" not in m]
    if missing:
        raise RuntimeError(f"missing keys loading {ckpt_path}: {missing[:8]}")
    if unexpected:
        print(f"  [warn] ignored unexpected keys: {list(unexpected)[:8]}", flush=True)

    meta = {kk: state[kk] for kk in ("step", "tokens_seen", "cumulative_flops")
            if isinstance(state, dict) and kk in state}
    model.to(device).eval()
    return model, meta


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _autocast(device: str):
    dt = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.autocast(device_type=dt, dtype=torch.bfloat16,
                          enabled=(dt == "cuda"))


@torch.no_grad()
def score_averaged(model, k: int, batches, device: str) -> dict:
    """
    Offset-ensemble score for k > 1, using the model's own forward so that
    whatever pooling rule it was trained with is the one applied.

    In eval mode the model pins its internal random offset to 0, so slicing o
    tokens off the front of the input shifts the window grid by o and makes the
    pass predict positions o+k, o+2k, ... This is the same procedure train.py
    uses for its eval_loss_all_pos column.
    """
    nll_sum, n_preds, n_seqs = 0.0, 0, 0
    per_offset = {o: [0.0, 0] for o in range(k)}

    for ids in batches:
        ids = ids.to(device)
        n_seqs += ids.size(0)
        with _autocast(device):
            for o in range(k):
                loss, logits = model(ids[:, o:])
                n = logits.size(0) * logits.size(1)
                nll_sum += loss.item() * n
                n_preds += n
                per_offset[o][0] += loss.item() * n
                per_offset[o][1] += n

    return {
        "k": k,
        "n_seqs": n_seqs,
        "mean_nll": nll_sum / max(n_preds, 1),
        "n_predictions": n_preds,
        "per_offset": {o: v[0] / max(v[1], 1) for o, v in per_offset.items()},
    }


@torch.no_grad()
def score_baseline(model, batches, device: str, seq_len: int) -> dict:
    """
    k = 1 score, kept per-position so it can be restricted afterwards to the
    exact position set an offset ensemble covers (positions >= k).
    """
    pos_nll = torch.zeros(seq_len, dtype=torch.float64)
    pos_cnt = torch.zeros(seq_len, dtype=torch.float64)
    n_seqs = 0

    for ids in batches:
        ids = ids.to(device)
        n_seqs += ids.size(0)
        with _autocast(device):
            logits = model(ids)[:, :-1]
            labels = ids[:, 1:]
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                reduction="none",
            ).view(ids.size(0), -1)
        pos_nll[1:ids.size(1)] += nll.sum(dim=0).double().cpu()
        pos_cnt[1:ids.size(1)] += ids.size(0)

    def _mean(min_pos: int) -> dict:
        sel = pos_cnt > 0
        sel[:min_pos] = False
        tot, cnt = pos_nll[sel].sum().item(), pos_cnt[sel].sum().item()
        return {"mean_nll": tot / max(cnt, 1), "n_predictions": int(cnt)}

    out = {"k": 1, "n_seqs": n_seqs, **_mean(1)}
    # Position sets matching each k we compare against.
    out["restricted"] = {f"pos_ge_{k}": _mean(k) for k in (2, 4, 8)}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Rescore Hub checkpoints on the offset-ensemble metric.")
    p.add_argument("--groups", nargs="+", default=["pairs", "pooling", "context"],
                   choices=["pairs", "pooling", "context"],
                   help="Which blocks of the manifest to score.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Score only these config names (overrides --groups).")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Directory holding eval.bin. Strongly recommended: "
                        "without it the eval split is re-derived by streaming "
                        "and the numbers drift from the published ones.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_batches", type=int, default=None,
                   help="Limit eval batches (default: all of eval.bin).")
    p.add_argument("--download_dir", type=str, default="/tmp/ta_ckpts")
    p.add_argument("--keep_downloads", action="store_true",
                   help="Do not delete each checkpoint after scoring. The "
                        "500M files are 5.8 GB each.")
    p.add_argument("--tokenizer_name", type=str, default="EleutherAI/pythia-70m")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None, help="Write results as JSON.")
    p.add_argument("--dry_run", action="store_true",
                   help="List what would be scored and its download size.")
    args = p.parse_args()

    if args.only:
        todo = [m for m in MANIFEST if m[0] in set(args.only)]
        unknown = set(args.only) - {m[0] for m in MANIFEST}
        if unknown:
            p.error(f"not in the manifest: {sorted(unknown)}")
    else:
        todo = [m for m in MANIFEST if m[3] in set(args.groups)]

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()

    if args.dry_run:
        total = 0
        print(f"{'config':28s} {'repo':34s} {'seq':>5s} {'size':>10s}")
        for name, repo, seq_len, group, note in todo:
            rid = HF_PREFIX + repo
            try:
                files = api.list_repo_tree(rid, path_in_repo="checkpoints",
                                           recursive=True, expand=True)
                size = next((f.size for f in files
                             if f.path.endswith("final.pt")), 0)
            except Exception as exc:
                print(f"{name:28s} {repo:34s} {seq_len:5d}  ERROR {exc}")
                continue
            total += size
            flag = "  <<< BROKEN" if size < 10_000 else ""
            print(f"{name:28s} {repo:34s} {seq_len:5d} {size/1e9:9.2f}G{flag}")
        print(f"\n{len(todo)} checkpoints, {total/1e9:.1f} GB to download.")
        return

    if not args.data_dir or not (Path(args.data_dir) / "eval.bin").exists():
        print("[warn] no eval.bin: the eval split will be re-derived by "
              "streaming, so these numbers are not strictly comparable to the "
              "published ones. Pass --data_dir pointing at the training cache.",
              flush=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vocab_size = len(tok)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dl_root = Path(args.download_dir)
    dl_root.mkdir(parents=True, exist_ok=True)

    report: dict = {}
    for i, (name, repo, seq_len, group, note) in enumerate(todo, 1):
        rid = HF_PREFIX + repo
        print(f"\n[{i}/{len(todo)}] {note}  ({name} <- {rid}, seq_len={seq_len})",
              flush=True)

        local = dl_root / repo
        try:
            ckpt = Path(hf_hub_download(
                repo_id=rid, filename="checkpoints/final.pt",
                local_dir=str(local), repo_type="model",
            ))
        except Exception as exc:
            print(f"  [skip] download failed: {type(exc).__name__}: {exc}",
                  flush=True)
            report[name] = {"error": f"download failed: {exc}"}
            continue

        if ckpt.stat().st_size < 10_000:
            print("  [skip] file is a broken LFS pointer, not a checkpoint.",
                  flush=True)
            report[name] = {"error": "broken LFS pointer"}
            shutil.rmtree(local, ignore_errors=True)
            continue

        try:
            cfg = get_config(name)
            model, meta = build_and_load(cfg, ckpt, vocab_size, args.device)
            batches = eval_batches(
                Path(args.data_dir) if args.data_dir else None,
                args.tokenizer_name, seq_len, args.batch_size, args.max_batches,
            )
            if cfg.averaging_k > 1:
                res = score_averaged(model, cfg.averaging_k, batches, args.device)
                print(f"  full-position NLL : {res['mean_nll']:.4f}   "
                      f"ppl {np.exp(res['mean_nll']):.2f}   "
                      f"({res['n_predictions']:,} predictions)", flush=True)
                for o, v in sorted(res["per_offset"].items()):
                    tag = " (training offset)" if o == 0 else ""
                    print(f"    offset {o}: {v:.4f}{tag}", flush=True)
            else:
                res = score_baseline(model, batches, args.device, seq_len)
                print(f"  all positions     : {res['mean_nll']:.4f}   "
                      f"ppl {np.exp(res['mean_nll']):.2f}", flush=True)
                for kk, v in res["restricted"].items():
                    print(f"    {kk}: {v['mean_nll']:.4f}", flush=True)

            res.update({"note": note, "group": group, "seq_len": seq_len,
                        "repo": rid, "ppl": float(np.exp(res["mean_nll"])),
                        "ckpt_meta": {k: float(v) for k, v in meta.items()}})
            report[name] = res
        except Exception as exc:
            print(f"  [skip] scoring failed: {type(exc).__name__}: {exc}",
                  flush=True)
            report[name] = {"error": f"scoring failed: {exc}"}
        finally:
            del_model = locals().get("model")
            if del_model is not None:
                del del_model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            if not args.keep_downloads:
                shutil.rmtree(local, ignore_errors=True)

    # ---- report ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("FULL-POSITION (offset-ensemble) NLL, nats/token")
    print("=" * 78)
    for group in ("pairs", "pooling", "context"):
        rows = [(n, r) for n, r in report.items()
                if r.get("group") == group and "error" not in r]
        if not rows:
            continue
        print(f"\n-- {group} --")
        for n, r in rows:
            print(f"  {r['note']:36s} k={r['k']:<2} "
                  f"NLL {r['mean_nll']:.4f}  ppl {r['ppl']:8.2f}")

    failed = {n: r["error"] for n, r in report.items() if "error" in r}
    if failed:
        print("\n-- not scored --")
        for n, e in failed.items():
            print(f"  {n:36s} {e}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=float))
        print(f"\nJSON written to {args.out}")


if __name__ == "__main__":
    main()
