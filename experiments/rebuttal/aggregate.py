"""
Seed statistics, mean ± SE learning curves, and equal-loss compute ratios.

For every group of runs (same configuration, different seeds) it reports
  final loss   last periodic eval row (256 eval sequences; the extra final row,
               which uses twice as many eval batches, is excluded so every
               number comes from the same eval set) and, when rescore JSONLs are
               given, the full-eval-set (3,339 sequences) value
  mean, std, SE = std/sqrt(n), n
for both the native metric (eval_loss) and the full-position metric
(eval_loss_all_pos, equal to eval_loss for k=1).

For every (baseline, method) pair it reports the equal-loss compute ratio
  FLOPs_baseline(end) / FLOPs_method(first reaching the baseline's final loss)
from the seed-mean curves (FLOPs from Eq. 2), and the mean ± SE of the
per-seed-pair ratios when seeds match. For two-stage runs only rows after the
switch to token-level training are used.

    python experiments/rebuttal/aggregate.py --out experiments/rebuttal/results/analysis \
        --rescore experiments/rebuttal/results/rescore_local.jsonl \
                  experiments/rebuttal/results/rescore_hub.jsonl
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.chinchilla.model_configs import MODEL_CONFIGS
from experiments.rebuttal.common import eq2_flops_per_raw_token, read_jsonl

CH = Path(__file__).resolve().parent.parent / "chinchilla"
RB = "results_rebuttal"
HUB_LOGS = Path(__file__).resolve().parent / "results" / "hf_logs"
HUB_OWNER = "FAIRC/token-averaging-"

HUB_RUNS: dict[str, str] = {
    "seed_replicates-model1_50m_seed1": "model1_50m",
    "seed_replicates-avg_50m_k2_seed1": "avg_50m_k2",
    "update_control-model1_50m": "model1_50m",
    "matched_mean-avg_50m_k2": "avg_50m_k2",
    "matched_mean-avg_50m_k4": "avg_50m_k4",
    "matched_mean-avg_50m_k8": "avg_50m_k8",
    "avg_50m_k4_learnable_pos": "avg_50m_k4_learnable_pos",
    "avg_50m_k8_learnable_pos": "avg_50m_k8_learnable_pos",
}

GROUPS: dict[str, list[str]] = {
    "k1": [f"{RB}/seeds/model1_50m_seed*", "hub:seed_replicates-model1_50m_seed1"],
    "k2 mean": [f"{RB}/seeds/avg_50m_k2_seed*", "hub:seed_replicates-avg_50m_k2_seed1"],
    "k2 learned c+p": [f"{RB}/seeds/avg_50m_k2_learnable_pos_seed*"],
    "k2 mean, batch 32 (k1 #updates)": [f"{RB}/update/avg_50m_k2_b32_seed*"],
    "k1, batch 8 (k2 #updates)": [f"{RB}/update/model1_50m_b8_seed*"],
    "k2 mean, batch 8": [f"{RB}/update/avg_50m_k2_b8_seed*"],
    "k1, batch 8 (July run)": ["hub:update_control-model1_50m"],
    "k2 mean (July matched_mean run)": ["hub:matched_mean-avg_50m_k2"],
    "PLT K=4": [f"{RB}/two_stage/plt_50m_k4_l067_seed*"],
    "PLT K=2": [f"{RB}/two_stage/plt_50m_k2_l067_seed*"],
    "PLT K=4, K*T context": [f"{RB}/two_stage/plt_50m_k4_ctx_seed*"],
    "PLT K=2, K*T context": [f"{RB}/two_stage/plt_50m_k2_ctx_seed*"],
    "TST s=4 r=0.3": [f"{RB}/two_stage/tst_50m_s4_r03_seed*"],
    "TST s=8 r=0.3": [f"{RB}/two_stage/tst_50m_s8_r03_seed*"],
    "k2, 1B unique x2 epochs": [f"{RB}/data/avg_50m_k2_uniq1b_seed*"],
    "k1, 1B unique x2 epochs (2x FLOPs)": [f"{RB}/data/model1_50m_2x_uniq1b_seed*"],
    "k4 learned c+p, mean init": [f"{RB}/pooling/avg_50m_k4_learnable_posuni_seed*"],
    "k8 learned c+p, mean init": [f"{RB}/pooling/avg_50m_k8_learnable_posuni_seed*"],
    "k4 learned c+p, recency init": ["hub:avg_50m_k4_learnable_pos"],
    "k8 learned c+p, recency init": ["hub:avg_50m_k8_learnable_pos"],
    "k4 mean": ["hub:matched_mean-avg_50m_k4"],
    "k8 mean": ["hub:matched_mean-avg_50m_k8"],
}
for _lr in ("6e-4", "1.2e-3", "2.4e-3"):
    GROUPS[f"k1 lr {_lr}"] = [f"{RB}/lr/model1_50m_lr{_lr}_seed*"]
    GROUPS[f"k2 lr {_lr}"] = [f"{RB}/lr/avg_50m_k2_lr{_lr}_seed*"]
    GROUPS[f"k1 b8 lr {_lr}"] = [f"{RB}/lr_b8/model1_50m_b8_lr{_lr}_seed*"]
    GROUPS[f"k2 b8 lr {_lr}"] = [f"{RB}/lr_b8/avg_50m_k2_b8_lr{_lr}_seed*"]

PAIRS: list[tuple[str, str]] = [
    ("k1", "k2 mean"),
    ("k1", "k2 learned c+p"),
    ("k1", "k2 mean, batch 32 (k1 #updates)"),
    ("k1, batch 8 (k2 #updates)", "k2 mean"),
    ("k1, batch 8 (k2 #updates)", "k2 mean, batch 8"),
    ("k1", "PLT K=4"),
    ("k1", "PLT K=2"),
    ("k1", "PLT K=4, K*T context"),
    ("k1", "PLT K=2, K*T context"),
    ("k1", "TST s=4 r=0.3"),
    ("k1", "TST s=8 r=0.3"),
    ("k1", "k2, 1B unique x2 epochs"),
] + [(f"k1 lr {lr}", f"k2 lr {lr}") for lr in ("6e-4", "1.2e-3", "2.4e-3")] + [
    (f"k1 b8 lr {lr}", f"k2 b8 lr {lr}") for lr in ("6e-4", "1.2e-3", "2.4e-3")]


# ---------------------------------------------------------------------------

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def fetch_hub_logs(repo: str) -> Path:
    from huggingface_hub import hf_hub_download, list_repo_files

    d = HUB_LOGS / repo
    if not (d / "loss_log.csv").exists():
        files = list_repo_files(HUB_OWNER + repo)
        for name in ("loss_log.csv", "run_meta.json", "config.json"):
            if name in files:
                hf_hub_download(HUB_OWNER + repo, name, local_dir=str(d))
    return d


def load_run(run_dir: Path, cfg_name: str | None = None) -> dict | None:
    log = run_dir / "loss_log.csv"
    if not log.exists():
        return None
    meta = {}
    if (run_dir / "run_meta.json").exists():
        meta = json.loads((run_dir / "run_meta.json").read_text())
    with open(log) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    cfg_name = cfg_name or meta.get("base_config") or re.sub(
        r"_seed\d+$", "", meta.get("run_name") or run_dir.name)
    cfg = MODEL_CONFIGS.get(cfg_name)
    budget = meta.get("planned_tokens") or meta.get("target_tokens") or (cfg.target_tokens if cfg else None)
    complete = (
        (run_dir / "checkpoints" / "final.pt").exists()
        or bool(meta.get("total_steps_planned")
                and int(rows[-1]["step"]) >= int(meta["total_steps_planned"]))
        or bool(budget and _f(rows[-1]["tokens_seen"]) >= 0.999 * float(budget))
    )
    periodic = rows[:-1] if complete and len(rows) > 1 else rows
    k = int(meta.get("averaging_k") or (cfg.averaging_k if cfg else 1))
    seq_len = int(meta.get("raw_seq_len") or 1024)
    per_tok = eq2_flops_per_raw_token(cfg, seq_len, k) if cfg else float("nan")
    switch = meta.get("stage_switch_step")

    step = np.array([int(r["step"]) for r in periodic])
    tokens = np.array([_f(r["tokens_seen"]) for r in periodic])
    native = np.array([_f(r["eval_loss"]) for r in periodic])
    allpos = np.array([_f(r.get("eval_loss_all_pos")) for r in periodic])
    if k == 1:
        allpos = np.where(np.isnan(allpos), native, allpos)
    if "flops_eq2" in periodic[0] and periodic[0]["flops_eq2"] not in (None, ""):
        flops = np.array([_f(r["flops_eq2"]) for r in periodic])
    else:
        flops = tokens * per_tok
    usable = np.ones_like(step, dtype=bool)
    if switch:
        usable = step > int(switch)
    return {
        "dir": run_dir, "name": run_dir.name, "seed": meta.get("seed"), "k": k,
        "complete": complete, "step": step, "tokens": tokens, "flops": flops,
        "native": native, "allpos": allpos, "usable": usable, "meta": meta,
    }


def rescore_lookup(paths: list[str]) -> dict[str, dict]:
    out = {}
    for p in paths:
        for r in read_jsonl(p):
            if "error" in r or r.get("max_seqs") is not None:
                continue
            if not str(r.get("ckpt", "")).endswith("final.pt") and not r.get("label", "").startswith("hub-"):
                continue
            if ":checkpoints/step_" in str(r.get("ckpt", "")):
                continue
            out[r["label"]] = r
    return out


def rescore_for(run: dict, table: dict[str, dict]) -> dict | None:
    d: Path = run["dir"]
    parent = d.parent.name.removeprefix("results_")
    for cand in (d.name, f"hub-{parent}-{d.name}", f"hub-{d.name}"):
        if cand in table:
            return table[cand]
    if parent == "matched_mean":
        return table.get(f"hub-matched_mean-{d.name}")
    return None


def stats(vals: list[float]) -> dict:
    v = np.array([x for x in vals if x is not None and not math.isnan(x)])
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "se": None}
    sd = float(v.std(ddof=1)) if n > 1 else None
    return {"n": n, "mean": float(v.mean()), "std": sd,
            "se": sd / math.sqrt(n) if sd is not None else None, "values": v.tolist()}


def last_usable(run: dict, key: str) -> float:
    m = run["usable"] & ~np.isnan(run[key])
    return float(run[key][m][-1]) if m.any() else float("nan")


def first_crossing(flops: np.ndarray, loss: np.ndarray, target: float) -> float | None:
    """Smallest FLOPs at which the (linearly interpolated) curve reaches target."""
    for i in range(len(loss)):
        if loss[i] <= target:
            if i == 0:
                return float(flops[0])
            f0, f1, l0, l1 = flops[i - 1], flops[i], loss[i - 1], loss[i]
            return float(f0 + (l0 - target) / max(l0 - l1, 1e-12) * (f1 - f0))
    return None


def mean_curve(runs: list[dict], key: str, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    ys = []
    for r in runs:
        m = r["usable"] & ~np.isnan(r[key])
        if m.sum() < 2:
            continue
        x, y = r["flops"][m], r[key][m]
        yi = np.interp(grid, x, y, left=np.nan, right=np.nan)
        ys.append(yi)
    if not ys:
        return np.full_like(grid, np.nan), np.full_like(grid, np.nan), 0
    Y = np.vstack(ys)
    n = np.sum(~np.isnan(Y), axis=0)
    mu = np.nanmean(Y, axis=0)
    sd = np.nanstd(Y, axis=0, ddof=1) if len(ys) > 1 else np.full_like(grid, np.nan)
    return mu, sd / np.sqrt(np.maximum(n, 1)), len(ys)


def equal_loss(base: list[dict], meth: list[dict], key: str) -> dict:
    base_final = [last_usable(r, key) for r in base if r["complete"]]
    meth_c = [r for r in meth if r["complete"]]
    if not base_final or not meth_c:
        return {}
    target = float(np.nanmean(base_final))
    base_flops = float(np.mean([r["flops"][r["usable"]][-1] for r in base if r["complete"]]))
    lo = max(min(r["flops"][r["usable"]][0] for r in meth_c), 1.0)
    hi = max(r["flops"][r["usable"]][-1] for r in meth_c)
    grid = np.geomspace(lo, hi, 600)
    mu, _, n = mean_curve(meth_c, key, grid)
    ok = ~np.isnan(mu)
    cross = first_crossing(grid[ok], mu[ok], target) if ok.any() else None
    out = {"target": target, "baseline_flops": base_flops, "n_base": len(base_final),
           "n_method": n, "method_flops_at_target": cross,
           "ratio_mean_curve": base_flops / cross if cross else None}
    by_seed_b = {r["seed"]: r for r in base if r["complete"]}
    ratios = []
    for r in meth_c:
        b = by_seed_b.get(r["seed"])
        if b is None:
            continue
        t = last_usable(b, key)
        m = r["usable"] & ~np.isnan(r[key])
        c = first_crossing(r["flops"][m], r[key][m], t)
        if c:
            ratios.append(b["flops"][b["usable"]][-1] / c)
    if ratios:
        s = stats(ratios)
        out.update({"ratio_per_seed_mean": s["mean"], "ratio_per_seed_se": s["se"],
                    "ratio_per_seed_n": s["n"], "ratio_per_seed_values": ratios})
    return out


def fmt(s: dict, digits: int = 4) -> str:
    if not s or s.get("mean") is None:
        return "-"
    if s.get("se") is None:
        return f"{s['mean']:.{digits}f} (n=1)"
    return f"{s['mean']:.{digits}f} ± {s['se']:.{digits}f} (sd {s['std']:.{digits}f}, n={s['n']})"


def plot(groups: dict[str, list[dict]], pairs, key: str, out_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] matplotlib unavailable ({exc}); skipping figures")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for base, meth in pairs:
        rb = [r for r in groups.get(base, []) if r["usable"].any()]
        rm = [r for r in groups.get(meth, []) if r["usable"].any()]
        if not rb or not rm:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for label, runs, color in ((base, rb, "#1f77b4"), (meth, rm, "#d62728")):
            lo = max(min(r["flops"][r["usable"]][0] for r in runs if r["usable"].any()), 1.0)
            hi = max(r["flops"][r["usable"]][-1] for r in runs if r["usable"].any())
            grid = np.geomspace(lo, hi, 400)
            mu, se, n = mean_curve(runs, key, grid)
            ax.plot(grid, mu, color=color, label=f"{label} (n={n})")
            if n > 1:
                ax.fill_between(grid, mu - se, mu + se, color=color, alpha=0.25, linewidth=0)
        ax.set_xscale("log")
        ax.set_xlabel("training FLOPs (Eq. 2)")
        ax.set_ylabel("full-position eval loss" if key == "allpos" else "native eval loss")
        finals = [last_usable(r, key) for r in rb + rm]
        finals = [f for f in finals if not math.isnan(f)]
        if finals:
            ax.set_ylim(min(finals) - 0.05, min(finals) + 0.6)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
        fname = out_dir / f"curve_{key}_{re.sub(r'[^A-Za-z0-9]+', '_', base)}__{re.sub(r'[^A-Za-z0-9]+', '_', meth)}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=160)
        plt.close(fig)
        files.append(str(fname))

    names = [g for g, rs in groups.items() if rs]
    if names:
        fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(names) + 1.2))
        for i, g in enumerate(names):
            vals = [last_usable(r, key) for r in groups[g] if r["complete"]]
            vals = [v for v in vals if not math.isnan(v)]
            if not vals:
                continue
            ax.scatter(vals, [i] * len(vals), s=14, color="#555555")
            s = stats(vals)
            if s["se"] is not None:
                ax.errorbar(s["mean"], i, xerr=s["se"], fmt="o", color="#d62728", capsize=3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("final full-position eval loss" if key == "allpos" else "final native eval loss")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fname = out_dir / f"finals_{key}.png"
        fig.savefig(fname, dpi=160)
        plt.close(fig)
        files.append(str(fname))
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chinchilla_dir", default=str(CH))
    p.add_argument("--rescore", nargs="*", default=[])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    root = Path(args.chinchilla_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rtable = rescore_lookup(args.rescore)

    groups: dict[str, list[dict]] = {}
    for g, patterns in GROUPS.items():
        runs = []
        seen = set()
        for pat in patterns:
            if pat.startswith("hub:"):
                repo = pat[len("hub:"):]
                r = load_run(fetch_hub_logs(repo), HUB_RUNS[repo])
                if r is not None:
                    runs.append(r)
                continue
            for d in sorted(glob.glob(str(root / pat))):
                d = Path(d)
                if d in seen or not d.is_dir():
                    continue
                seen.add(d)
                r = load_run(d)
                if r is not None:
                    runs.append(r)
        groups[g] = runs

    summary = {"groups": {}, "pairs": []}
    lines = ["# Rebuttal runs: seed statistics and equal-loss ratios", "",
             "Final losses are the last periodic eval (256 sequences). `full eval` is the "
             "rescored value on all 3,339 eval sequences. Mean ± SE over seeds.", ""]
    lines += ["| group | runs (complete) | native | full-position | full eval (rescored) |",
              "|---|---|---|---|---|"]
    for g, runs in groups.items():
        if not runs:
            continue
        done = [r for r in runs if r["complete"]]
        nat = stats([last_usable(r, "native") for r in done])
        alp = stats([last_usable(r, "allpos") for r in done])
        res = stats([(rescore_for(r, rtable) or {}).get("nll_all", float("nan")) for r in done])
        summary["groups"][g] = {"runs": [str(r["dir"]) for r in runs],
                                "complete": [str(r["dir"]) for r in done],
                                "native": nat, "allpos": alp, "full_eval": res}
        lines.append(f"| {g} | {len(runs)} ({len(done)}) | {fmt(nat)} | {fmt(alp)} | {fmt(res)} |")

    lines += ["", "## Equal-loss compute ratios (full-position metric, Eq. 2 FLOPs)", "",
              "| baseline | method | target | ratio (seed-mean curve) | ratio per seed pair |",
              "|---|---|---|---|---|"]
    for base, meth in PAIRS:
        rb, rm = groups.get(base, []), groups.get(meth, [])
        if not rb or not rm:
            continue
        for key in ("allpos", "native"):
            e = equal_loss(rb, rm, key)
            if not e:
                continue
            summary["pairs"].append({"baseline": base, "method": meth, "metric": key, **e})
            if key != "allpos":
                continue
            rc = f"{e['ratio_mean_curve']:.3f}x" if e.get("ratio_mean_curve") else "not reached"
            rs = (f"{e['ratio_per_seed_mean']:.3f} ± {e['ratio_per_seed_se'] or 0:.3f} "
                  f"(n={e['ratio_per_seed_n']})" if e.get("ratio_per_seed_mean") else "-")
            lines.append(f"| {base} | {meth} | {e['target']:.4f} | {rc} | {rs} |")

    lines += ["", "## Seed differences", "",
              "Difference of seed means (method − baseline), full-position metric, with "
              "SE = sqrt(SE_b² + SE_m²).", "", "| baseline | method | Δ | SE |", "|---|---|---|---|"]
    for base, meth in PAIRS:
        sb = summary["groups"].get(base, {}).get("allpos")
        sm = summary["groups"].get(meth, {}).get("allpos")
        if not sb or not sm or sb["mean"] is None or sm["mean"] is None:
            continue
        se = math.sqrt((sb["se"] or 0) ** 2 + (sm["se"] or 0) ** 2) if (sb["se"] and sm["se"]) else None
        lines.append(f"| {base} | {meth} | {sm['mean'] - sb['mean']:+.4f} | "
                     f"{se:.4f} |" if se else f"| {base} | {meth} | {sm['mean'] - sb['mean']:+.4f} | - |")

    figs = plot(groups, PAIRS, "allpos", out / "figures") + plot(groups, PAIRS, "native", out / "figures")
    summary["figures"] = figs
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out / 'summary.md'}, {out / 'summary.json'}, {len(figs)} figure(s)")


if __name__ == "__main__":
    main()
