"""
Autoregressive decoding benchmark: prefill time, decode throughput, KV-cache
size and peak memory, for three ways of serving a model.

  k1      ordinary model, one cache with one entry per token (T entries)
  rotate  k>1 model, k caches (one per group grid). Token t is predicted by grid
          t mod k from its newest complete group, so each new token costs one
          single-position forward over one cache of ~t/k entries. Total cache
          ~T entries, as for k1; attention reads per token are ~t/k.
  single  k>1 model, one cache on the offset-0 grid (~T/k entries). Between group
          boundaries the incomplete group is pooled into a tentative slot that is
          overwritten until the group completes (see eval_partial_group.py for
          its quality).

Models are randomly initialised with the requested architecture (speed and
memory do not depend on the weights). All modes use the same explicit layer
loop and static preallocated caches, so differences come from the cache
layout and attention length, not from kernel choice.

    python experiments/rebuttal/bench_decode.py --configs model1_500m model1_1b \
        --k 2 --prompt_lens 2048 8192 32768 --gen 256 --batch_sizes 1 8 32 \
        --out results/rebuttal/bench_decode.jsonl

    python experiments/rebuttal/bench_decode.py --check   # correctness on a tiny model
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.chinchilla.model_configs import get_config
from experiments.rebuttal.common import (
    StaticKVCache, append_jsonl, cached_forward, default_device, forward_hidden, layer_modules,
)
from experiments.rebuttal.eval_partial_group import partial_mask
from experiments.shared.olm_model import OLMTransformerBody

VOCAB = 50_277


def build(cfg_name: str, max_positions: int, device, dtype) -> OLMTransformerBody:
    cfg = get_config(cfg_name)
    bb = OLMTransformerBody(VOCAB, cfg.d_model, cfg.n_heads, cfg.n_layers, max_positions)
    bb.tie_embedding_weights()
    return bb.to(device=device, dtype=dtype).eval()


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def decode(bb, layers, mode: str, k: int, tokens: torch.Tensor, P: int, collect: bool = False):
    """
    Teacher-forced decoding of tokens[:, P:] given the prompt tokens[:, :P].
    Returns (prefill_s, decode_s, cache_bytes, logits or None); logits[i] predicts
    tokens[:, P + i].
    """
    device = tokens.device
    B, total = tokens.shape
    G = total - P
    attn = layers[0][1]
    L, H, hd = len(layers), attn.num_heads, attn.head_dim
    dtype = next(bb.parameters()).dtype
    head = bb.embed_out
    emb = bb.embed_in
    out_logits = [] if collect else None

    def pool(ids):
        e = emb(ids)
        return e.reshape(B, -1, k, e.size(-1)).mean(dim=2) if k > 1 else e

    def emit(h):
        lg = head(h[:, -1:])
        if collect:
            out_logits.append(lg[:, 0].float())
        return lg.argmax(-1)

    _sync(device)
    t0 = time.perf_counter()
    if mode == "k1":
        caches = [StaticKVCache(L, B, H, total, hd, device, dtype)]
        emit(cached_forward(bb, emb(tokens[:, :P]), caches[0], layers=layers))
    elif mode == "rotate":
        caches = [StaticKVCache(L, B, H, total // k + 1, hd, device, dtype) for _ in range(k)]
        last = None
        for o in range(k):
            n = (P - o) // k
            h = cached_forward(bb, pool(tokens[:, o:o + n * k]), caches[o], layers=layers)
            if o == P % k:
                last = h
        emit(last)
    elif mode == "single":
        caches = [StaticKVCache(L, B, H, total // k + 2, hd, device, dtype)]
        n = P // k
        h = cached_forward(bb, pool(tokens[:, :n * k]), caches[0], layers=layers)
        r = P % k
        if r:
            h = cached_forward(bb, emb(tokens[:, n * k:P]).mean(1, keepdim=True), caches[0],
                               commit=False, layers=layers)
        emit(h)
    else:
        raise ValueError(mode)
    _sync(device)
    t1 = time.perf_counter()

    for t in range(P + 1, P + G):
        if mode == "k1":
            h = cached_forward(bb, emb(tokens[:, t - 1:t]), caches[0], layers=layers)
        elif mode == "rotate":
            h = cached_forward(bb, pool(tokens[:, t - k:t]), caches[t % k], layers=layers)
        else:
            r = t % k
            if r == 0:
                h = cached_forward(bb, pool(tokens[:, t - k:t]), caches[0], layers=layers)
            else:
                h = cached_forward(bb, emb(tokens[:, t - r:t]).mean(1, keepdim=True), caches[0],
                                   commit=False, layers=layers)
        emit(h)
    _sync(device)
    t2 = time.perf_counter()
    cache_bytes = sum(c.nbytes for c in caches)
    logits = torch.stack(out_logits, 1) if collect else None
    return t1 - t0, t2 - t1, cache_bytes, logits


@torch.no_grad()
def reference_logits(bb, mode: str, k: int, tokens: torch.Tensor, P: int) -> torch.Tensor:
    """Same predictions as decode(), computed with full (uncached) parallel passes."""
    B, total = tokens.shape
    emb_all = bb.embed_in(tokens)
    D = emb_all.size(-1)
    rows = []
    for t in range(P, total):
        if mode == "k1":
            rows.append(bb(tokens[:, :t])[:, -1])
            continue
        if mode == "rotate":
            o = t % k
            n = (t - o) // k
            g = emb_all[:, o:o + n * k].reshape(B, n, k, D).mean(2)
            rows.append(bb.embed_out(bb.body(g))[:, -1])
            continue
        r = t % k
        n = t // k
        g = emb_all[:, :n * k].reshape(B, n, k, D).mean(2)
        if r == 0:
            rows.append(bb.embed_out(bb.body(g))[:, -1])
        else:
            p = emb_all[:, n * k:t].mean(1, keepdim=True)
            x = torch.cat([g, p], 1)
            pos = torch.cat([torch.arange(n), torch.tensor([n])]).to(tokens.device)
            mask = partial_mask(n, 1, tokens.device)
            mask[n, :n] = True
            h = forward_hidden(bb, x, pos, attn_mask=mask)
            rows.append(bb.embed_out(h)[:, -1])
    return torch.stack(rows, 1).float()


def run_check() -> None:
    torch.manual_seed(0)
    bb = OLMTransformerBody(97, 64, 4, 3, 128).eval()
    tokens = torch.randint(0, 97, (2, 41))
    P = 17
    layers = layer_modules(bb)
    ok = True
    for mode, k in (("k1", 1), ("rotate", 2), ("rotate", 4), ("single", 2), ("single", 4)):
        _, _, _, got = decode(bb, layers, mode, k, tokens, P, collect=True)
        ref = reference_logits(bb, mode, k, tokens, P)
        err = (got - ref).abs().max().item()
        ok &= err < 1e-4
        print(f"check {mode:6s} k={k}: max |dlogit| = {err:.2e}", flush=True)
    print("ALL OK" if ok else "MISMATCH", flush=True)
    if not ok:
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", default=["model1_500m"])
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--modes", nargs="+", default=["k1", "rotate", "single"])
    p.add_argument("--prompt_lens", type=int, nargs="+", default=[2048, 8192])
    p.add_argument("--gen", type=int, default=256)
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--repeats", type=int, default=2, help="Timed repeats (best is kept).")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=default_device())
    p.add_argument("--out", default=None)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    if args.check:
        run_check()
        return

    dtype = getattr(torch, args.dtype)
    device = args.device
    for cfg_name in args.configs:
        max_total = max(args.prompt_lens) + args.gen
        bb = build(cfg_name, max_total + 8, device, dtype)
        layers = layer_modules(bb)
        weight_bytes = sum(q.numel() * q.element_size() for q in bb.parameters())
        for P in args.prompt_lens:
            for B in args.batch_sizes:
                for mode in args.modes:
                    k = 1 if mode == "k1" else args.k
                    row = {"config": cfg_name, "mode": mode, "k": k, "prompt_len": P,
                           "gen": args.gen, "batch": B, "dtype": args.dtype,
                           "weight_bytes": weight_bytes, "device": str(device)}
                    try:
                        tokens = torch.randint(0, VOCAB, (B, P + args.gen), device=device)
                        decode(bb, layers, mode, k, tokens[:, : min(P, 256) + 8], min(P, 256))
                        best = None
                        for _ in range(args.repeats):
                            if str(device).startswith("cuda"):
                                torch.cuda.empty_cache()
                                torch.cuda.reset_peak_memory_stats()
                            pre, dec, cache_b, _ = decode(bb, layers, mode, k, tokens, P)
                            peak = (torch.cuda.max_memory_allocated()
                                    if str(device).startswith("cuda") else 0)
                            if best is None or dec < best[1]:
                                best = (pre, dec, cache_b, peak)
                        pre, dec, cache_b, peak = best
                        steps = args.gen - 1
                        row.update({
                            "prefill_s": pre, "decode_s": dec,
                            "decode_ms_per_step": 1e3 * dec / max(steps, 1),
                            "decode_tokens_per_s": B * steps / max(dec, 1e-9),
                            "kv_cache_bytes": cache_b, "peak_mem_bytes": peak,
                        })
                        print(f"{cfg_name} {mode:6s} k={k} P={P:6d} B={B:3d}  prefill {pre:7.3f}s  "
                              f"decode {row['decode_ms_per_step']:7.2f} ms/step  "
                              f"{row['decode_tokens_per_s']:9.1f} tok/s  "
                              f"KV {cache_b/2**30:6.2f} GiB  peak {peak/2**30:6.2f} GiB", flush=True)
                    except torch.cuda.OutOfMemoryError:
                        row["error"] = "oom"
                        print(f"{cfg_name} {mode:6s} k={k} P={P:6d} B={B:3d}  OOM", flush=True)
                        torch.cuda.empty_cache()
                    if args.out:
                        append_jsonl(args.out, row)
        del bb
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
