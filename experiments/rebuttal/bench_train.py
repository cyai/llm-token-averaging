"""
Training throughput per configuration: optimizer-step time, raw tokens/s,
achieved model FLOPs/s (Eq. 2), peak memory, and the projected wall-clock time
for the configuration's full token budget on one GPU.

Uses the same model classes, bf16 autocast, torch.compile and AdamW as
train.py, on random token batches (data loading excluded).

    python experiments/rebuttal/bench_train.py \
        --configs model1_50m avg_50m_k2 avg_50m_k4 avg_50m_k8 model1_500m avg_500m_k2 \
        --batch_size 16 --out results/rebuttal/bench_train.jsonl

Two-stage baselines: add  --two_stage_k 4  with a k=1 config to time the
compressed stage (the token-level stage costs the same as the k=1 config).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.chinchilla.model_configs import get_config
from experiments.chinchilla.train import _build_optimizer, train_flops_per_seq
from experiments.rebuttal.common import append_jsonl, default_device
from experiments.shared.averaged_lm import build_method_config
from experiments.shared.olm_model import (
    OLMAveragedLanguageModel, OLMTransformerBody, OLMTwoStageLanguageModel,
)

VOCAB = 50_277


def build(cfg, two_stage_k: int):
    bb = OLMTransformerBody(VOCAB, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.context_len)
    if cfg.tie_embeddings:
        bb.tie_embedding_weights()
    if two_stage_k > 1:
        return OLMTwoStageLanguageModel(bb, build_method_config(f"uniform_k{two_stage_k}"), two_stage_k), two_stage_k
    if cfg.averaging_k > 1:
        return OLMAveragedLanguageModel(bb, build_method_config(cfg.method_name or f"uniform_k{cfg.averaging_k}")), cfg.averaging_k
    return bb, 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=None, help="Default: config context_len.")
    p.add_argument("--seq_mult", type=int, default=1, help="Raw length multiplier (TST stage 1).")
    p.add_argument("--two_stage_k", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--device", default=default_device())
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dev = args.device
    is_cuda = dev.startswith("cuda")

    for name in args.configs:
        cfg = get_config(name)
        model, k = build(cfg, args.two_stage_k)
        model.to(dev).train()
        fwd_loss = not isinstance(model, OLMTransformerBody)
        if not args.no_compile:
            model = torch.compile(model)
        opt = _build_optimizer(model, lr=cfg.lr)
        seq = (args.seq_len or cfg.context_len) * args.seq_mult
        ids = torch.randint(0, VOCAB, (args.batch_size, seq), device=dev)

        def step():
            with torch.autocast(device_type="cuda" if is_cuda else "cpu", dtype=torch.bfloat16,
                                enabled=is_cuda):
                if fwd_loss:
                    loss, _ = model(ids)
                else:
                    logits = model(ids)[:, :-1]
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        row = {"config": name, "k": k, "two_stage_k": args.two_stage_k, "batch": args.batch_size,
               "raw_seq_len": seq, "compiled": not args.no_compile, "device": dev}
        try:
            for _ in range(args.warmup):
                step()
            if is_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(args.steps):
                step()
            if is_cuda:
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / args.steps
            tokens_per_step = args.batch_size * seq
            _, eq2 = train_flops_per_seq(cfg, seq // k)
            flops_per_step = args.batch_size * eq2
            tok_s = tokens_per_step / dt
            row.update({
                "step_s": dt, "raw_tokens_per_s": tok_s,
                "model_tflops_per_s": flops_per_step / dt / 1e12,
                "peak_mem_gib": (torch.cuda.max_memory_allocated() / 2**30) if is_cuda else 0.0,
                "target_tokens": cfg.target_tokens,
                "projected_hours_1gpu": cfg.target_tokens / tok_s / 3600,
            })
            print(f"{name:28s} k={k} B={args.batch_size} L={seq}  {dt*1e3:7.1f} ms/step  "
                  f"{tok_s:10.0f} tok/s  {row['model_tflops_per_s']:6.1f} TFLOP/s  "
                  f"mem {row['peak_mem_gib']:5.1f} GiB  "
                  f"budget {cfg.target_tokens/1e9:.2f}B -> {row['projected_hours_1gpu']:.1f} h",
                  flush=True)
        except torch.cuda.OutOfMemoryError:
            row["error"] = "oom"
            print(f"{name}: OOM", flush=True)
        if args.out:
            append_jsonl(args.out, row)
        del model, opt
        if is_cuda:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
