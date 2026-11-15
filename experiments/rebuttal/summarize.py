"""
Tables for the non-training outputs:

  downstream   per-model accuracy ± binomial SE, plus seed-mean ± seed-SE for
               labels that differ only in _seed<N>
  partial      single-cache (partial-group) vs offset-ensemble NLL by residue class
  decode       decoding benchmark, with throughput and KV size relative to k1
  train        training-throughput benchmark
  rescore      full-eval-set NLL of every scored checkpoint

    python experiments/rebuttal/summarize.py --results experiments/rebuttal/results
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.rebuttal.common import read_jsonl

TASKS = ["lambada", "hellaswag", "piqa", "arc_easy", "winogrande"]


def downstream(res: Path) -> list[str]:
    files = sorted((res / "downstream").glob("*.json"))
    if not files:
        return []
    rows = [json.loads(f.read_text()) for f in files]
    out = ["## Downstream (accuracy ± binomial SE)", "",
           "| model | k | pooling | " + " | ".join(TASKS) + " | LAMBADA ppl |",
           "|---|---|---|" + "---|" * len(TASKS) + "---|"]
    for r in rows:
        cells = []
        for t in TASKS:
            tr = r["tasks"].get(t)
            cells.append(f"{tr['acc']:.3f} ± {tr['se']:.3f}" if tr else "-")
        ppl = r["tasks"].get("lambada", {}).get("ppl")
        out.append(f"| {r['label']} | {r['k']} | {r.get('pooling') or '-'} | " + " | ".join(cells)
                   + f" | {ppl:.2f} |" if ppl else f"| {r['label']} | {r['k']} | "
                   f"{r.get('pooling') or '-'} | " + " | ".join(cells) + " | - |")

    groups = defaultdict(list)
    for r in rows:
        base = re.sub(r"_seed\d+$", "", r["label"])
        if base != r["label"]:
            groups[base].append(r)
    if groups:
        out += ["", "### Seed means (mean ± SE over seeds)", "",
                "| model | seeds | " + " | ".join(TASKS) + " |", "|---|---|" + "---|" * len(TASKS)]
        for g, rs in sorted(groups.items()):
            cells = []
            for t in TASKS:
                v = [x["tasks"][t]["acc"] for x in rs if t in x["tasks"]]
                if len(v) > 1:
                    m = sum(v) / len(v)
                    sd = math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))
                    cells.append(f"{m:.3f} ± {sd / math.sqrt(len(v)):.3f}")
                elif v:
                    cells.append(f"{v[0]:.3f}")
                else:
                    cells.append("-")
            out.append(f"| {g} | {len(rs)} | " + " | ".join(cells) + " |")
    return out + [""]


def partial(res: Path) -> list[str]:
    rows = read_jsonl(res / "partial_group.jsonl") + read_jsonl(res / "partial_group_window.jsonl")
    if not rows:
        return []
    out = ["## Single-cache decoding with a tentative partial group", "",
           "NLL on the common target set. `ensemble` = k caches (paper metric); "
           "`single_*` = one cache of T/k entries; `baseline` = k=1 model.", ""]
    for r in rows:
        k = r["k"]
        cols = ["all"] + [f"r{i}" for i in range(k)]
        out += [f"**{r['label']}** (k={k}, {r['pooling']}, {r['n_seqs']} seqs, "
                f"walker check {r.get('walker_max_abs_logit_diff', float('nan')):.1e})", "",
                "| method | " + " | ".join(cols) + " | Δ all vs ensemble |",
                "|---|" + "---|" * (len(cols) + 1)]
        ens = r["ensemble"]["all"]
        for key in [kk for kk in r if kk in ("ensemble", "baseline") or kk.startswith("single_")]:
            vals = r[key]
            out.append(f"| {key} | " + " | ".join(
                f"{vals[c]:.4f}" if vals.get(c) is not None else "-" for c in cols)
                + f" | {vals['all'] - ens:+.4f} |")
        out.append("")
    return out


def decode(res: Path) -> list[str]:
    rows = [r for r in read_jsonl(res / "bench_decode.jsonl")]
    if not rows:
        return []
    ref = {}
    for r in rows:
        if r["mode"] == "k1" and "error" not in r:
            ref[(r["config"], r["prompt_len"], r["batch"])] = r
    out = ["## Decoding benchmark", "",
           "| config | prompt | batch | mode | k | prefill s | ms/step | tok/s | vs k1 | KV GiB | "
           "KV vs k1 | peak GiB |", "|---" * 12 + "|"]
    for r in rows:
        if "error" in r:
            out.append(f"| {r['config']} | {r['prompt_len']} | {r['batch']} | {r['mode']} | {r['k']} | "
                       f"{r['error']} |" + " |" * 7)
            continue
        b = ref.get((r["config"], r["prompt_len"], r["batch"]))
        sp = f"{r['decode_tokens_per_s'] / b['decode_tokens_per_s']:.2f}x" if b else "-"
        kv = f"{r['kv_cache_bytes'] / b['kv_cache_bytes']:.2f}x" if b else "-"
        out.append(
            f"| {r['config']} | {r['prompt_len']} | {r['batch']} | {r['mode']} | {r['k']} | "
            f"{r['prefill_s']:.3f} | {r['decode_ms_per_step']:.2f} | {r['decode_tokens_per_s']:.0f} | "
            f"{sp} | {r['kv_cache_bytes'] / 2**30:.2f} | {kv} | {r['peak_mem_bytes'] / 2**30:.2f} |")
    return out + [""]


def train(res: Path) -> list[str]:
    rows = read_jsonl(res / "bench_train.jsonl")
    if not rows:
        return []
    out = ["## Training throughput (one GPU, synthetic data)", "",
           "| config | k | two-stage k | batch | raw len | ms/step | raw tok/s | TFLOP/s | peak GiB | "
           "budget h (1 GPU) |", "|---" * 10 + "|"]
    for r in rows:
        if "error" in r:
            out.append(f"| {r['config']} | {r['k']} | {r['two_stage_k']} | {r['batch']} | "
                       f"{r['raw_seq_len']} | {r['error']} |" + " |" * 4)
            continue
        out.append(f"| {r['config']} | {r['k']} | {r['two_stage_k']} | {r['batch']} | "
                   f"{r['raw_seq_len']} | {r['step_s'] * 1e3:.1f} | {r['raw_tokens_per_s']:.0f} | "
                   f"{r['model_tflops_per_s']:.1f} | {r['peak_mem_gib']:.1f} | "
                   f"{r['projected_hours_1gpu']:.1f} |")
    return out + [""]


def rescore(res: Path) -> list[str]:
    out = []
    for name in ("rescore_hub.jsonl", "rescore_local.jsonl", "rescore_hub_curves.jsonl"):
        rows = [r for r in read_jsonl(res / name) if "error" not in r]
        errs = [r for r in read_jsonl(res / name) if "error" in r]
        if not rows and not errs:
            continue
        out += [f"## {name}: full eval set", "",
                "| label | config | k | step | tokens | FLOPs (Eq. 2) | full-position NLL | "
                "pos≥2 | pos≥4 | pos≥8 | offset-0 | seqs |", "|---" * 12 + "|"]
        for r in rows:
            ge = r.get("nll_pos_ge", {})
            out.append(
                f"| {r['label']} | {r['config']} | {r['k']} | {r.get('step') or '-'} | "
                f"{(r.get('tokens_seen') or 0) / 1e9:.3f}B | {r.get('flops_eq2') or 0:.3e} | "
                f"{r['nll_all']:.4f} | {ge.get('2', float('nan')):.4f} | {ge.get('4', float('nan')):.4f} | "
                f"{ge.get('8', float('nan')):.4f} | {r['nll_offset0']:.4f} | {r['n_seqs']} |")
        for r in errs:
            out.append(f"| {r['label']} | {r['config']} | error: {r['error'][:80]} |" + " |" * 9)
        out.append("")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(Path(__file__).resolve().parent / "results"))
    p.add_argument("--out", default=None, help="Markdown file (default: <results>/tables.md)")
    args = p.parse_args()
    res = Path(args.results)
    lines = ["# Rebuttal evaluation tables", ""]
    for fn in (rescore, downstream, partial, decode, train):
        lines += fn(res)
    out = Path(args.out) if args.out else res / "tables.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
