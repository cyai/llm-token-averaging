"""
Score checkpoints on the full eval split (all of eval.bin) with the true pooling
rule, so every number in the comparison tables comes from one eval set and one
metric.

Per checkpoint it records:
  nll_all         full-position NLL (offset ensemble for k>1, next-token for k=1)
  nll_pos_ge      the same restricted to target positions >= 2, 4, 8 (the position
                  set an offset ensemble at that k covers), for exact k=1 vs k>1 parity
  nll_offset0     native training metric (offset-0 grid) for k>1
  per_offset      per-offset NLL for k>1
  per_seq_nll     mean NLL of each eval sequence (for paired standard errors)

Rows are appended to a JSONL file; checkpoints already present are skipped.

Inputs (any mix):
  --runs DIR [DIR ...]        local run dirs (results_dir/<run_name>); config is read
                              from run_meta.json (base_config or run_name) unless
                              given as DIR=CONFIG
  --runs_root ROOT            every run dir under ROOT that has a loss_log.csv
  --model CONFIG CKPT [LABEL] explicit checkpoint, CKPT = file, run dir, or
                              hf:<owner>/<repo>[:<file>]  (repeatable)
  --all_checkpoints           also score step_*.pt, not only final.pt
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.chinchilla.model_configs import MODEL_CONFIGS
from experiments.rebuttal.common import (
    append_jsonl, default_device, eq2_flops_per_raw_token, eval_batches,
    load_model, per_position_nll, read_jsonl, read_run_meta,
)


def config_for_run(run_dir: Path) -> str | None:
    meta_path = run_dir / "run_meta.json"
    names = []
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
        names += [meta.get("base_config"), meta.get("run_name")]
    names.append(run_dir.name)
    for n in names:
        if not n:
            continue
        n = re.sub(r"_seed\d+$", "", n)
        if n in MODEL_CONFIGS:
            return n
    return None


def checkpoints_of(run_dir: Path, all_ckpts: bool) -> list[Path]:
    ck = run_dir / "checkpoints"
    out = []
    if all_ckpts:
        out += sorted(ck.glob("step_*.pt"))
    if (ck / "final.pt").exists():
        out.append(ck / "final.pt")
    return out


@torch.no_grad()
def score(m, data_dir: str, seq_len: int, batch_size: int, max_seqs: int | None) -> dict:
    device = str(m.device)
    k = m.k
    tot_sum = 0.0
    tot_cnt = 0
    ge_sum = {2: 0.0, 4: 0.0, 8: 0.0}
    ge_cnt = {2: 0, 4: 0, 8: 0}
    off_sum = np.zeros(k)
    off_cnt = np.zeros(k)
    per_seq = []
    positions = torch.arange(seq_len, device=device)
    for ids in eval_batches(data_dir, seq_len, batch_size, max_seqs):
        ids = ids.to(device)
        nll = per_position_nll(m, ids)
        valid = ~torch.isnan(nll)
        nll0 = torch.where(valid, nll, torch.zeros_like(nll))
        tot_sum += nll0.sum().item()
        tot_cnt += int(valid.sum())
        for g in ge_sum:
            sel = valid & (positions >= g).view(1, -1)
            ge_sum[g] += nll0[sel].sum().item()
            ge_cnt[g] += int(sel.sum())
        if k > 1:
            for o in range(k):
                sel = valid & ((positions % k) == o).view(1, -1)
                off_sum[o] += nll0[sel].sum().item()
                off_cnt[o] += int(sel.sum())
        per_seq += (nll0.sum(1) / valid.sum(1).clamp_min(1)).tolist()
    out = {
        "nll_all": tot_sum / max(tot_cnt, 1),
        "n_predictions": tot_cnt,
        "n_seqs": len(per_seq),
        "nll_pos_ge": {str(g): ge_sum[g] / max(ge_cnt[g], 1) for g in ge_sum},
        "per_seq_nll": [round(x, 5) for x in per_seq],
    }
    if k > 1:
        per_offset = (off_sum / np.maximum(off_cnt, 1)).tolist()
        out["per_offset"] = per_offset
        out["nll_offset0"] = per_offset[0]
    else:
        out["nll_offset0"] = out["nll_all"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--runs_root", nargs="*", default=[])
    p.add_argument("--model", nargs="+", action="append", default=[],
                   metavar=("CONFIG", "CKPT"))
    p.add_argument("--all_checkpoints", action="store_true")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_seqs", type=int, default=None,
                   help="Default: all eval sequences (3,339 at seq_len 1024).")
    p.add_argument("--seq_len", type=int, default=None,
                   help="Override the scoring length (default: the run's raw_seq_len).")
    p.add_argument("--device", default=default_device())
    p.add_argument("--download_dir", default="/tmp/ta_ckpts")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    jobs: list[tuple[str, str, str]] = []
    run_dirs: list[tuple[Path, str | None]] = []
    for spec in args.runs:
        d, _, cfg = spec.partition("=")
        run_dirs.append((Path(d), cfg or None))
    for root in args.runs_root:
        for log in sorted(Path(root).rglob("loss_log.csv")):
            run_dirs.append((log.parent, None))
    for d, cfg in run_dirs:
        cfg = cfg or config_for_run(d)
        if cfg is None:
            print(f"[skip] cannot infer config for {d}; pass {d}=CONFIG", flush=True)
            continue
        for ck in checkpoints_of(d, args.all_checkpoints):
            jobs.append((cfg, str(ck), d.name))
    for entry in args.model:
        if len(entry) < 2:
            p.error("--model needs CONFIG CKPT [LABEL]")
        cfg, ck = entry[0], entry[1]
        label = entry[2] if len(entry) > 2 else Path(ck.split(":")[-1]).stem
        jobs.append((cfg, ck, label))

    done = {(r.get("ckpt"), r.get("max_seqs"), r.get("seq_len")) for r in read_jsonl(args.out)
            if "error" not in r}
    print(f"{len(jobs)} checkpoint(s) to consider; {len(done)} already scored in {args.out}",
          flush=True)
    if args.dry_run:
        for j in jobs:
            print("  ", j)
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    for i, (cfg_name, ck, label) in enumerate(jobs, 1):
        cfg = MODEL_CONFIGS[cfg_name]
        print(f"\n[{i}/{len(jobs)}] {label}: {cfg_name} <- {ck}", flush=True)
        local_meta = {} if ck.startswith("hf:") else read_run_meta(
            Path(ck) / "checkpoints" / "final.pt" if Path(ck).is_dir() else Path(ck))
        seq_len = args.seq_len or int(local_meta.get("raw_seq_len") or cfg.context_len)
        if (str(ck), args.max_seqs, seq_len) in done and not args.force:
            print("  already scored, skipping", flush=True)
            continue
        try:
            t0 = time.time()
            m = load_model(cfg_name, ck, device=args.device, download_dir=args.download_dir)
            meta = m.meta.get("run_meta") or local_meta
            res = score(m, args.data_dir, seq_len, args.batch_size, args.max_seqs)
            tokens = m.meta.get("tokens_seen")
            flops = m.meta.get("flops_eq2")
            if flops is None and tokens is not None:
                flops = float(tokens) * eq2_flops_per_raw_token(cfg, seq_len, m.k)
            row = {
                "label": label, "config": cfg_name, "ckpt": str(ck), "k": m.k,
                "seq_len": seq_len, "max_seqs": args.max_seqs,
                "step": m.meta.get("step"), "tokens_seen": tokens, "flops_eq2": flops,
                "seed": meta.get("seed"), "two_stage_k": meta.get("two_stage_k"),
                "run_name": meta.get("run_name"), "elapsed_s": time.time() - t0, **res,
            }
            append_jsonl(args.out, row)
            extra = f"  offset0 {res['nll_offset0']:.4f}" if m.k > 1 else ""
            print(f"  full-position NLL {res['nll_all']:.4f} over {res['n_seqs']} seqs"
                  f"{extra}  ({time.time() - t0:.0f}s)", flush=True)
        except Exception as exc:
            print(f"  [error] {type(exc).__name__}: {exc}", flush=True)
            append_jsonl(args.out, {"label": label, "config": cfg_name, "ckpt": str(ck),
                                    "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
