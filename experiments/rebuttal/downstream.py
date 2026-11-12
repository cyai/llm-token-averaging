"""
Downstream zero-shot evals (LAMBADA, HellaSwag, PIQA, ARC-Easy, WinoGrande) with
the pooling rule each run was trained with.

experiments/benchmarks/eval_downstream.py rebuilds every k>1 model with mean
pooling and drops the learned pooling weights, so its numbers for learned and
weighted pooling runs are wrong. This script reuses its task code but swaps the
scoring function for the offset ensemble from common.py, which calls the run's
own pooling module.

Each model writes <out_dir>/<label>.json with accuracy, n, and the binomial
standard error sqrt(acc(1-acc)/n) per task.

    python experiments/rebuttal/downstream.py --out_dir results/rebuttal/downstream \
        --runs results/rebuttal/seeds/model1_50m_seed2 ... \
        --model avg_500m_k2 hf:cyai/token-averaging-avg_500m_k2 500m_k2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import experiments.benchmarks.eval_downstream as ed
from experiments.rebuttal.common import default_device, full_position_logprobs, load_model
from experiments.rebuttal.rescore import config_for_run

_CURRENT = {}


def _patched_get_logprobs(backbone, input_ids, k):
    return full_position_logprobs(_CURRENT["model"], input_ids)


ed.get_logprobs = _patched_get_logprobs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--runs", nargs="*", default=[],
                   help="Local run dirs (DIR or DIR=CONFIG); label = dir name.")
    p.add_argument("--model", nargs="+", action="append", default=[],
                   metavar=("CONFIG", "CKPT"), help="CONFIG CKPT [LABEL]")
    p.add_argument("--tasks", default="lambada,hellaswag,piqa,arc_easy,winogrande")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=default_device())
    p.add_argument("--download_dir", default="/tmp/ta_ckpts")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    jobs: list[tuple[str, str, str]] = []
    for spec in args.runs:
        d, _, cfg = spec.partition("=")
        d = Path(d)
        cfg = cfg or config_for_run(d)
        if cfg is None:
            print(f"[skip] cannot infer config for {d}", flush=True)
            continue
        jobs.append((cfg, str(d), d.name))
    for entry in args.model:
        cfg, ck = entry[0], entry[1]
        label = entry[2] if len(entry) > 2 else cfg
        jobs.append((cfg, ck, label))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ed.TOKENIZER_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (cfg_name, ck, label) in enumerate(jobs, 1):
        out_path = out_dir / f"{label}.json"
        existing = json.loads(out_path.read_text()) if out_path.exists() else None
        if existing and not args.force and all(t in existing.get("tasks", {}) for t in tasks):
            print(f"[{i}/{len(jobs)}] {label}: done already", flush=True)
            continue
        print(f"\n[{i}/{len(jobs)}] {label}: {cfg_name} <- {ck}", flush=True)
        try:
            m = load_model(cfg_name, ck, device=args.device, download_dir=args.download_dir)
        except Exception as exc:
            print(f"  [error] load failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        _CURRENT["model"] = m
        result = existing if (existing and not args.force) else {
            "label": label, "config": cfg_name, "ckpt": str(ck), "k": m.k,
            "pooling": m.method_cfg.name if m.method_cfg else None,
            "params_M": sum(p.numel() for p in m.backbone.parameters()) / 1e6,
            "seed": (m.meta.get("run_meta") or {}).get("seed"),
            "limit": args.limit, "tasks": {},
        }
        max_len = m.cfg.context_len
        for task in tasks:
            if task in result["tasks"] and not args.force:
                continue
            t0 = time.time()
            r = ed.TASK_FNS[task](m.backbone, m.k, tokenizer, args.device, max_len, args.limit)
            acc, n = r.get("acc", 0.0), max(r.get("n", 0), 1)
            r["se"] = math.sqrt(acc * (1 - acc) / n)
            r["elapsed_s"] = time.time() - t0
            result["tasks"][task] = r
            extra = f", ppl={r['ppl']:.2f}" if "ppl" in r else ""
            print(f"  {task}: acc={acc:.4f} ± {r['se']:.4f} (n={n}){extra} "
                  f"[{r['elapsed_s']:.0f}s]", flush=True)
            out_path.write_text(json.dumps(result, indent=2, default=float))
        del m
        _CURRENT.clear()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
