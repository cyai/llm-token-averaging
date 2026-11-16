"""
All rebuttal runs, and a small GPU queue that executes them.

Each job is one process on one GPU. A job is skipped when its `done` path
exists (final.pt for training, a marker file otherwise), and waits until every
path in `needs` exists, so evaluation jobs start as soon as the run they
evaluate finishes. Training jobs pass --resume, so killing the queue and
starting it again continues from the last checkpoint.

    # what would run, in order
    python experiments/rebuttal/jobs.py --sets all --list

    # run on 8 GPUs
    python experiments/rebuttal/jobs.py --sets all --gpus 0,1,2,3,4,5,6,7

    # a subset
    python experiments/rebuttal/jobs.py --sets seeds update --gpus 0,1,2,3

Sets (priority order of `all`):
  check       decode-path correctness test on a tiny model (CPU, seconds)
  bench       training-throughput and decoding benchmarks
  rescore_hub full-eval-set rescoring of every existing checkpoint on the Hub
  seeds       50M seed replicates: k=1, k=2 mean, k=2 learned (content+position)
  update      update-count controls: k=2 at batch 32 (same #updates as k=1),
              k=1 at batch 8 (same #updates as k=2)
  two_stage   patch-level training (K=2,4) and token superposition (s=4,8),
              FLOPs-matched to the 50M k=1 baseline
  lr          learning-rate sweep for k=1 and k=2
  data        data-constrained: k=2 for 2 epochs over 1B unique tokens
  pooling     learned content+position pooling at k=4/8 started from mean pooling
  kv          partial-group single-cache quality (+ partial-group fine-tune)
  downstream  downstream evals with the correct pooling, Hub checkpoints
  hub_curves  intermediate 250M/500M Hub checkpoints on the full eval set (large downloads)
  lr_b8       learning-rate sweep for k=1 and k=2 at batch 8
  kv_ft2      partial-group fine-tunes at the end-of-pretraining LR, with standard-loss controls
  kv_window   single cache whose open slot holds the last k raw tokens (no fine-tuning)
Every training job also adds a rescore job and a downstream job for its run.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PY = sys.executable
TRAIN = "experiments/chinchilla/train.py"
RB_DIR = "experiments/rebuttal"
HF = "hf:FAIRC/token-averaging-"


@dataclass
class Job:
    name: str
    cmd: list[str]
    done: Path
    needs: list[Path] = field(default_factory=list)
    set: str = ""
    marker: bool = False
    cpu: bool = False
    cleanup: list[str] = field(default_factory=list)


class Plan:
    def __init__(self, results: Path, data_dir: str, out: Path):
        self.results = results
        self.data_dir = data_dir
        self.out = out
        self.hub_dir = out / "_hub_ckpts"
        self.jobs: list[Job] = []

    def marker(self, name: str) -> Path:
        return self.out / "_done" / name

    def add(self, job: Job) -> None:
        self.jobs.append(job)

    def train(self, set_name: str, sub: str, config: str, seed: int, run_name: str | None = None,
              batch: int = 16, eval_batches: int | None = None, extra: list[str] | None = None,
              downstream: bool = True) -> Path:
        run_name = run_name or config
        run_dir = self.results / sub / (f"{run_name}_seed{seed}" if seed else run_name)
        eval_batches = eval_batches or max(1, 256 // batch)
        cmd = [PY, TRAIN, "--model", config, "--run_name", run_name, "--seed", str(seed),
               "--batch_size", str(batch), "--seq_len", "1024", "--log_steps", "100",
               "--eval_batches", str(eval_batches), "--num_workers", "4",
               "--data_dir", self.data_dir, "--checkpoint_steps", "10000",
               "--keep_last_checkpoints", "1", "--results_dir", str(self.results / sub),
               "--resume"] + (extra or [])
        final = run_dir / "checkpoints" / "final.pt"
        tag = run_dir.name
        self.add(Job(f"train:{sub}/{tag}", cmd, final, set=set_name,
                     cleanup=[str(run_dir / "checkpoints" / "step_*.pt")]))
        self.add(Job(
            f"rescore:{sub}/{tag}",
            [PY, f"{RB_DIR}/rescore.py", "--data_dir", self.data_dir,
             "--out", str(self.out / "rescore_local.jsonl"), "--runs", f"{run_dir}={config}"],
            self.marker(f"rescore_{sub}_{tag}"), needs=[final], set=set_name, marker=True))
        if downstream:
            self.add(Job(
                f"downstream:{sub}/{tag}",
                [PY, f"{RB_DIR}/downstream.py", "--out_dir", str(self.out / "downstream"),
                 "--runs", f"{run_dir}={config}"],
                self.marker(f"downstream_{sub}_{tag}"), needs=[final], set=set_name, marker=True))
        return run_dir

    def script(self, set_name: str, name: str, args: list[str], needs: list[Path] | None = None,
               cpu: bool = False, cleanup: list[str] | None = None) -> None:
        if any(a.startswith("hf:") for a in args):
            args = args + ["--download_dir", str(self.hub_dir)]
        self.add(Job(name, [PY] + args, self.marker(name.replace(":", "_").replace("/", "_")),
                     needs=needs or [], set=set_name, marker=True, cpu=cpu, cleanup=cleanup or []))


HUB_POOLING = [
    ("avg_50m_k2", "matched_mean-avg_50m_k2"),
    ("avg_50m_k2_learnable_pos", "avg_50m_k2_learnable_pos"),
    ("avg_50m_k2_learnable", "avg_50m_k2_learnable"),
    ("avg_50m_k2_wexp", "avg_50m_k2_wexp"),
    ("avg_50m_k4", "matched_mean-avg_50m_k4"),
    ("avg_50m_k4_learnable_pos", "avg_50m_k4_learnable_pos"),
    ("avg_50m_k4_learnable", "avg_50m_k4_learnable"),
    ("avg_50m_k4_wexp", "avg_50m_k4_wexp"),
    ("avg_50m_k8", "matched_mean-avg_50m_k8"),
    ("avg_50m_k8_learnable_pos", "avg_50m_k8_learnable_pos"),
    ("avg_50m_k8_learnable", "avg_50m_k8_learnable"),
    ("avg_50m_k8_wexp", "avg_50m_k8_wexp"),
]
HUB_PAIRS = [
    ("model1_50m", "seed_replicates-model1_50m_seed1"),
    ("avg_50m_k2", "seed_replicates-avg_50m_k2_seed1"),
    ("model1_50m", "update_control-model1_50m"),
    ("model1_125m", "model1_125m"),
    ("avg_125m_k2", "avg_125m_k2"),
    ("model1_250m", "model1_250m"),
    ("avg_250m_k2", "avg_250m_k2"),
    ("model1_500m", "model1_500m"),
    ("avg_500m_k2", "avg_500m_k2"),
    ("model1_50m_tied_2nctx", "model1_50m_tied_2ctx"),
]


def hub_label(repo: str) -> str:
    return "hub-" + repo


def build_plan(results: Path, data_dir: str, out: Path) -> Plan:
    pl = Plan(results, data_dir, out)

    # --- check / bench ------------------------------------------------------
    pl.script("check", "check:decode", [f"{RB_DIR}/bench_decode.py", "--check", "--device", "cpu"],
              cpu=True)
    pl.script("bench", "bench:train_50m", [
        f"{RB_DIR}/bench_train.py", "--configs", "model1_50m", "avg_50m_k2", "avg_50m_k4",
        "avg_50m_k8", "--batch_size", "16", "--out", str(out / "bench_train.jsonl")])
    pl.script("bench", "bench:train_large", [
        f"{RB_DIR}/bench_train.py", "--configs", "model1_250m", "avg_250m_k2", "model1_500m",
        "avg_500m_k2", "--batch_size", "16", "--out", str(out / "bench_train.jsonl")])
    pl.script("bench", "bench:train_two_stage", [
        f"{RB_DIR}/bench_train.py", "--configs", "model1_50m", "--two_stage_k", "4",
        "--batch_size", "16", "--out", str(out / "bench_train.jsonl")])
    pl.script("bench", "bench:decode_500m", [
        f"{RB_DIR}/bench_decode.py", "--configs", "model1_500m", "--k", "2",
        "--prompt_lens", "2048", "8192", "32768", "--gen", "256", "--batch_sizes", "1", "8", "32",
        "--out", str(out / "bench_decode.jsonl")])
    pl.script("bench", "bench:decode_1b", [
        f"{RB_DIR}/bench_decode.py", "--configs", "model1_1b", "--k", "2",
        "--prompt_lens", "2048", "8192", "32768", "--gen", "256", "--batch_sizes", "1", "8", "32",
        "--out", str(out / "bench_decode.jsonl")])
    pl.script("bench", "bench:decode_500m_k4", [
        f"{RB_DIR}/bench_decode.py", "--configs", "model1_500m", "--k", "4",
        "--modes", "rotate", "single", "--prompt_lens", "8192", "32768", "--gen", "256",
        "--batch_sizes", "1", "8", "32", "--out", str(out / "bench_decode.jsonl")])

    # --- rescoring of existing Hub checkpoints --------------------------------
    for cfg, repo in HUB_PAIRS + HUB_POOLING:
        pl.script("rescore_hub", f"rescore_hub:{repo}", [
            f"{RB_DIR}/rescore.py", "--data_dir", data_dir, "--out", str(out / "rescore_hub.jsonl"),
            "--model", cfg, HF + repo, hub_label(repo)])

    # --- seeds ----------------------------------------------------------------
    for seed in (2, 3, 4, 5):
        pl.train("seeds", "seeds", "model1_50m", seed)
        pl.train("seeds", "seeds", "avg_50m_k2", seed)
    for seed in (1, 2, 3):
        pl.train("seeds", "seeds", "avg_50m_k2_learnable_pos", seed)

    # --- update-count controls ------------------------------------------------
    for seed in (1, 2, 3):
        pl.train("update", "update", "avg_50m_k2", seed, run_name="avg_50m_k2_b32", batch=32)
        pl.train("update", "update", "model1_50m", seed, run_name="model1_50m_b8", batch=8)
        pl.train("update", "update", "avg_50m_k2", seed, run_name="avg_50m_k2_b8", batch=8)

    # --- PLT / TST (FLOPs-matched to the 50M k=1 baseline, Eq. 2) -------------
    two_stage = [
        ("plt_50m_k4_l067", ["--two_stage_k", "4", "--two_stage_frac", "0.6667",
                             "--target_tokens", "2074679113"]),
        ("plt_50m_k2_l067", ["--two_stage_k", "2", "--two_stage_frac", "0.6667",
                             "--target_tokens", "1556009335"]),
        ("tst_50m_s4_r03", ["--two_stage_k", "4", "--two_stage_frac", "0.3",
                            "--two_stage_seq_mult", "4", "--total_steps", "61036"]),
        ("tst_50m_s8_r03", ["--two_stage_k", "8", "--two_stage_frac", "0.3",
                            "--two_stage_seq_mult", "8", "--total_steps", "61036"]),
    ]
    for seed in (1, 2):
        for run_name, extra in two_stage:
            pl.train("two_stage", "two_stage", "model1_50m", seed, run_name=run_name, extra=extra)

    # --- PLT as in Shao et al.: K*T raw context in stage 1, same raw tokens per
    # step, optimizer and schedule reset at the switch; steps solve
    # N * (lambda/K + 1 - lambda) = 61036 so Eq. 2 FLOPs match k=1 ------------
    for k, steps in ((2, 91554), (4, 122072)):
        for seed in (1, 2):
            pl.train("plt_ctx", "two_stage", "model1_50m", seed, run_name=f"plt_50m_k{k}_ctx",
                     extra=["--two_stage_k", str(k), "--two_stage_frac", "0.6667",
                            "--two_stage_seq_mult", str(k), "--two_stage_batch_size", str(16 // k),
                            "--two_stage_reset_optim", "--total_steps", str(steps),
                            "--target_tokens", str(steps * 16384)])

    # --- learning-rate sweep --------------------------------------------------
    for lr in ("6e-4", "1.2e-3", "2.4e-3"):
        for cfg in ("model1_50m", "avg_50m_k2"):
            for seed in ((1, 2, 3) if lr == "1.2e-3" else (1,)):
                pl.train("lr", "lr", cfg, seed, run_name=f"{cfg}_lr{lr}", extra=["--lr", lr],
                         downstream=lr == "1.2e-3")
    for seed in (1, 2):
        for lr in ("1.2e-3", "6e-4", "2.4e-3"):
            for cfg in ("model1_50m", "avg_50m_k2"):
                pl.train("lr_b8", "lr_b8", cfg, seed, run_name=f"{cfg}_b8_lr{lr}", batch=8,
                         extra=["--lr", lr])

    # --- data-constrained -----------------------------------------------------
    pl.train("data", "data", "avg_50m_k2", 1, run_name="avg_50m_k2_uniq1b",
             extra=["--max_unique_tokens", "1000000000"])
    pl.train("data", "data", "model1_50m", 1, run_name="model1_50m_2x_uniq1b",
             extra=["--max_unique_tokens", "1000000000", "--target_tokens", "2000000000"])

    # --- matched-init pooling -------------------------------------------------
    pl.train("pooling", "pooling", "avg_50m_k4_learnable_posuni", 1)
    pl.train("pooling", "pooling", "avg_50m_k8_learnable_posuni", 1)

    # --- KV: partial-group single cache -------------------------------------
    kv_models = [
        ("avg_50m_k2", HF + "seed_replicates-avg_50m_k2_seed1", "50m_k2_seed1"),
        ("avg_50m_k2_learnable_pos", HF + "avg_50m_k2_learnable_pos", "50m_k2_lpos"),
        ("avg_50m_k4", HF + "matched_mean-avg_50m_k4", "50m_k4_mean"),
        ("avg_50m_k8", HF + "matched_mean-avg_50m_k8", "50m_k8_mean"),
    ]
    for cfg, ck, label in kv_models:
        pl.script("kv", f"kv:partial_{label}", [
            f"{RB_DIR}/eval_partial_group.py", "--model", cfg, ck, label,
            "--baseline", "model1_50m", HF + "seed_replicates-model1_50m_seed1",
            "--data_dir", data_dir, "--out", str(out / "partial_group.jsonl")])
    for cfg, base, label in (("avg_250m_k2", "model1_250m", "250m_k2"),
                             ("avg_500m_k2", "model1_500m", "500m_k2")):
        pl.script("kv", f"kv:partial_{label}", [
            f"{RB_DIR}/eval_partial_group.py", "--model", cfg, HF + cfg, label,
            "--baseline", base, HF + base, "--batch_size", "4",
            "--data_dir", data_dir, "--out", str(out / "partial_group.jsonl")])
    for cfg, ck, label, base, base_ck, bs in (
            [(c, ck, lb, "model1_50m", HF + "seed_replicates-model1_50m_seed1", "8")
             for c, ck, lb in kv_models]
            + [("avg_250m_k2", HF + "avg_250m_k2", "250m_k2", "model1_250m", HF + "model1_250m", "4"),
               ("avg_500m_k2", HF + "avg_500m_k2", "500m_k2", "model1_500m", HF + "model1_500m", "4")]
            + [("avg_50m_k2_learnable_pos",
                str(results / "seeds" / f"avg_50m_k2_learnable_pos_seed{s}"), f"50m_k2_lpos_seed{s}",
                "model1_50m", HF + "seed_replicates-model1_50m_seed1", "8") for s in (1, 2, 3)]
            + [(f"avg_50m_k{k}_learnable_pos", HF + f"avg_50m_k{k}_learnable_pos", f"50m_k{k}_lpos",
                "model1_50m", HF + "seed_replicates-model1_50m_seed1", "8") for k in (4, 8)]
            + [(f"avg_50m_k{k}_{p}", HF + f"avg_50m_k{k}_{p}", f"50m_k{k}_{p}",
                "model1_50m", HF + "seed_replicates-model1_50m_seed1", "8")
               for p in ("learnable", "wexp") for k in (2, 4, 8)]
            + [(f"avg_50m_k{k}_learnable_posuni",
                str(results / "pooling" / f"avg_50m_k{k}_learnable_posuni_seed1"), f"50m_k{k}_lposuni",
                "model1_50m", HF + "seed_replicates-model1_50m_seed1", "8") for k in (4, 8)]):
        pl.script("kv_window", f"kv:window_{label}", [
            f"{RB_DIR}/eval_partial_group.py", "--model", cfg, ck, f"{label}_window",
            "--baseline", base, base_ck, "--modes", "renorm,window", "--batch_size", bs,
            "--data_dir", data_dir, "--out", str(out / "partial_group_window.jsonl")])
    ft_variants = [("kv", "ft", None, ["--lam", "1"]),
                   ("kv_ft2", "ft_lowlr", "low", ["--lam", "1", "--warmup", "500"]),
                   ("kv_ft2", "ctrl_lowlr", "low", ["--lam", "0", "--warmup", "500"])]
    for cfg, ck, label, bs, low_lr in (
            ("avg_50m_k2", HF + "seed_replicates-avg_50m_k2_seed1", "50m_k2_seed1", 16, "2e-5"),
            ("avg_50m_k4", HF + "matched_mean-avg_50m_k4", "50m_k4_mean", 16, "2e-5"),
            ("avg_500m_k2", HF + "avg_500m_k2", "500m_k2", 8, "1.2e-5")):
        for set_name, suffix, lr, extra in ft_variants:
            tag = f"{label}_{suffix}"
            ft_dir = results / "partial_ft" / tag
            ft_final = ft_dir / "checkpoints" / "final.pt"
            pl.add(Job(f"kv:finetune_{label}" if suffix == "ft" else f"kv:finetune_{tag}", [
                PY, f"{RB_DIR}/finetune_partial_group.py", "--config", cfg, "--ckpt", ck,
                "--data_dir", data_dir, "--out_dir", str(ft_dir), "--steps", "3000",
                "--batch_size", str(bs), "--download_dir", str(pl.hub_dir)]
                + (["--lr", low_lr] if lr == "low" else []) + extra, ft_final, set=set_name))
            base = "model1_500m" if "500m" in cfg else "model1_50m"
            base_ck = HF + ("model1_500m" if "500m" in cfg else "seed_replicates-model1_50m_seed1")
            pl.script(set_name, f"kv:partial_{tag}", [
                f"{RB_DIR}/eval_partial_group.py", "--model", cfg, str(ft_dir), tag,
                "--baseline", base, base_ck, "--batch_size", "4" if "500m" in cfg else "8",
                "--data_dir", data_dir, "--out", str(out / "partial_group.jsonl")], needs=[ft_final])
            pl.script(set_name, f"kv:rescore_{tag}", [
                f"{RB_DIR}/rescore.py", "--data_dir", data_dir,
                "--out", str(out / "rescore_local.jsonl"), "--runs", f"{ft_dir}={cfg}"],
                needs=[ft_final])

    # --- downstream on Hub checkpoints (correct pooling) ----------------------
    for cfg, repo in HUB_POOLING + HUB_PAIRS:
        if cfg == "model1_50m_tied_2nctx":
            continue
        pl.script("downstream", f"downstream_hub:{repo}", [
            f"{RB_DIR}/downstream.py", "--out_dir", str(out / "downstream"),
            "--model", cfg, HF + repo, hub_label(repo)])

    # --- intermediate large checkpoints on the full eval set ------------------
    for cfg, steps in (("model1_500m", range(50_000, 300_001, 50_000)),
                       ("avg_500m_k2", range(50_000, 600_001, 50_000)),
                       ("model1_250m", range(50_000, 150_001, 50_000)),
                       ("avg_250m_k2", range(50_000, 300_001, 50_000))):
        for s in steps:
            pl.script("hub_curves", f"hub_curves:{cfg}_{s}", [
                f"{RB_DIR}/rescore.py", "--data_dir", data_dir,
                "--out", str(out / "rescore_hub_curves.jsonl"),
                "--model", cfg, f"{HF}{cfg}:checkpoints/step_{s:08d}.pt", f"hub-{cfg}-step{s}"],
                cleanup=[str(pl.hub_dir / f"FAIRC__token-averaging-{cfg}" / "checkpoints"
                             / f"step_{s:08d}.pt")])
    return pl


SET_ORDER = ["check", "bench", "rescore_hub", "seeds", "update", "two_stage", "kv",
             "downstream", "lr", "data", "pooling", "hub_curves", "lr_b8", "kv_ft2", "kv_window", "plt_ctx"]


def order_jobs(jobs: list[Job], sets: list[str]) -> list[Job]:
    """Stable sort by set priority. A run's eval jobs sit right after its training
    job, so once the run finishes they are picked before later training jobs."""
    rank = {s: i for i, s in enumerate(SET_ORDER)}
    return sorted((j for j in jobs if j.set in sets), key=lambda j: rank[j.set])


def gpu_mem_used_mb() -> dict[str, int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    used = {}
    for line in out.strip().splitlines():
        idx, mem = (s.strip() for s in line.split(","))
        used[idx] = int(mem)
    return used


def run_queue(jobs: list[Job], gpus: list[str], log_dir: Path, poll: float = 15.0,
              max_foreign_mb: int | None = None) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs if not j.done.exists()]
    running: dict[str, tuple[subprocess.Popen, Job, object]] = {}
    free = list(gpus)
    failed: list[str] = []
    last_shared: list[str] = []

    def stop(*_):
        for proc, job, fh in running.values():
            proc.terminate()
        sys.exit(1)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f"[queue] {len(pending)} job(s) to run on GPUs {gpus}", flush=True)

    while pending or running:
        for gpu, (proc, job, fh) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del running[gpu]
            if not job.cpu:
                free.append(gpu)
            if rc == 0:
                if job.marker:
                    job.done.parent.mkdir(parents=True, exist_ok=True)
                    job.done.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
                for pattern in job.cleanup:
                    for path in glob.glob(pattern):
                        os.remove(path)
                print(f"[queue] done   {job.name}", flush=True)
            else:
                failed.append(job.name)
                print(f"[queue] FAILED {job.name} (exit {rc}); see {log_dir / safe(job.name)}.log",
                      flush=True)

        started = True
        while started:
            started = False
            if max_foreign_mb is None:
                usable = list(free)
            else:
                used = gpu_mem_used_mb()
                usable = [g for g in free if used.get(g, 0) <= max_foreign_mb]
                shared = sorted(set(free) - set(usable))
                if shared != last_shared:
                    print(f"[queue] skipping GPUs in use by other processes: {shared}", flush=True)
                    last_shared = shared
            for job in list(pending):
                if job.done.exists():
                    pending.remove(job)
                    continue
                if not all(n.exists() for n in job.needs):
                    continue
                if any(rj.name == job.name for _, rj, _ in running.values()):
                    continue
                if job.cpu:
                    slot = f"cpu-{job.name}"
                    env_gpu = ""
                elif usable:
                    slot = usable.pop(0)
                    free.remove(slot)
                    env_gpu = slot
                else:
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=env_gpu, PYTHONUNBUFFERED="1")
                fh = open(log_dir / f"{safe(job.name)}.log", "a")
                fh.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')}  {shlex.join(job.cmd)}\n")
                fh.flush()
                proc = subprocess.Popen(job.cmd, cwd=ROOT, env=env, stdout=fh,
                                        stderr=subprocess.STDOUT)
                running[slot] = (proc, job, fh)
                pending.remove(job)
                print(f"[queue] start  {job.name}  on {'cpu' if job.cpu else 'GPU ' + slot}",
                      flush=True)
                started = True

        if not running and pending and not any(all(n.exists() for n in j.needs) for j in pending):
            blocked = [j.name for j in pending]
            print(f"[queue] {len(blocked)} job(s) blocked on missing inputs "
                  f"(their training failed or was not selected):", flush=True)
            for b in blocked:
                print(f"   {b}", flush=True)
            break
        time.sleep(poll)

    print(f"[queue] finished; {len(failed)} failure(s)", flush=True)
    for f in failed:
        print(f"   {f}", flush=True)


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", default=["all"])
    p.add_argument("--gpus", default="0")
    p.add_argument("--data_dir", default=os.environ.get("DATA_DIR", "/data/fineweb"))
    p.add_argument("--results", default=os.environ.get(
        "RB_RESULTS", str(ROOT / "experiments" / "chinchilla" / "results_rebuttal")))
    p.add_argument("--out", default=os.environ.get(
        "RB_OUT", str(ROOT / "experiments" / "rebuttal" / "results")))
    p.add_argument("--only", default=None, help="Regex on job names.")
    p.add_argument("--list", action="store_true")
    p.add_argument("--no_eval_deps", action="store_true",
                   help="Drop the per-run rescore/downstream jobs.")
    p.add_argument("--max_foreign_mb", type=int, default=None,
                   help="Do not start a job on a GPU that other processes already use "
                        "more than this many MiB of.")
    args = p.parse_args()

    sets = SET_ORDER if "all" in args.sets else args.sets
    unknown = set(sets) - set(SET_ORDER)
    if unknown:
        p.error(f"unknown sets {sorted(unknown)}; choose from {SET_ORDER}")
    plan = build_plan(Path(args.results), args.data_dir, Path(args.out))
    jobs = order_jobs(plan.jobs, sets)
    if args.only:
        jobs = [j for j in jobs if re.search(args.only, j.name)]
    if args.no_eval_deps:
        jobs = [j for j in jobs if not (j.needs and j.name.split(":")[0] in ("rescore", "downstream"))]

    if args.list:
        for j in jobs:
            state = "done" if j.done.exists() else ("waits" if j.needs and not all(
                n.exists() for n in j.needs) else "ready")
            print(f"{state:5s}  [{j.set}] {j.name}")
            print(f"       {shlex.join(j.cmd)}")
        print(f"\n{len(jobs)} job(s); {sum(not j.done.exists() for j in jobs)} not done")
        return
    run_queue(jobs, [g.strip() for g in args.gpus.split(",") if g.strip()],
              Path(args.out) / "logs", max_foreign_mb=args.max_foreign_mb)


if __name__ == "__main__":
    main()
