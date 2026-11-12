"""
Quality of single-cache decoding with a tentative partial group.

Setting. Decoding with one KV cache on the offset-0 grid stores one entry per
complete group (T/k entries). To predict token p = j*k + r with 0 < r < k, the
last r tokens do not form a complete group yet; we pool them into a partial
group P_j^r, place it at RoPE position j, let it attend to the complete groups
G_0..G_{j-1}, and read the prediction from it. When the group completes, its
slot is overwritten by the full group G_j. Predictions with r = 0 are exactly
the offset-0 predictions the model was trained on.

The model never saw partial groups in training, so this is out of
distribution; this script measures how much worse it is than the offset
ensemble (k caches) on the same target positions.

All partial predictions for a fixed r are computed in one parallel pass over
the concatenation [G_0..G_{N-1}, P_0^r..P_{J-1}^r] with a mask in which G
is causal over G and P_j sees G_{<j} and itself. This is exactly what
incremental decoding with the overwritten slot computes.

Output: per checkpoint and pooling mode, NLL by residue class r = p mod k for
  ensemble   offset ensemble (the paper's metric; k caches)
  single     one cache + tentative partial group
  baseline   optional k=1 model on the same positions (--baseline CONFIG CKPT)
plus the all-position means over the common set of targets.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.rebuttal.common import (
    LoadedModel, append_jsonl, autocast, chunked_nll, default_device, eval_batches,
    forward_hidden, layer_modules, load_model, per_position_nll, pool_partial,
)


def partial_mask(n_groups: int, n_partial: int, device) -> torch.Tensor:
    N, J = n_groups, n_partial
    m = torch.zeros(N + J, N + J, dtype=torch.bool, device=device)
    m[:N, :N] = torch.tril(torch.ones(N, N, dtype=torch.bool, device=device))
    j = torch.arange(J, device=device).view(-1, 1)
    i = torch.arange(N, device=device).view(1, -1)
    m[N:, :N] = i < j
    m[N + torch.arange(J), N + torch.arange(J)] = True
    return m


@torch.no_grad()
def single_cache_nll(m: LoadedModel, ids: torch.Tensor, mode: str, layers) -> torch.Tensor:
    """[B, T] NLL from single-cache decoding with partial groups; NaN where undefined."""
    B, T = ids.shape
    k = m.k
    N = T // k
    out = torch.full((B, T), float("nan"), device=ids.device)
    with autocast(str(ids.device)):
        emb = m.backbone.embed_in(ids)
        G = m.pool(emb[:, : N * k], ids[:, : N * k])
        ref_norm = G.float().norm(dim=-1).mean()

        logits_g = m.backbone.embed_out(m.backbone.body(G[:, :-1]))
        pos_g = k * torch.arange(1, N, device=ids.device)
        out[:, pos_g] = chunked_nll(logits_g, ids[:, pos_g])

        for r in range(1, k):
            J = min((T - 1 - r) // k + 1, N)
            windows = emb[:, : J * k].reshape(B, J, k, -1)[:, :, :r]
            P = pool_partial(m, windows, r, mode=mode, ref_norm=ref_norm).to(G.dtype)
            X = torch.cat([G, P], dim=1)
            positions = torch.cat([torch.arange(N, device=ids.device),
                                   torch.arange(J, device=ids.device)])
            mask = partial_mask(N, J, ids.device)
            h = forward_hidden(m.backbone, X, positions, attn_mask=mask, layers=layers)
            logits_p = m.backbone.embed_out(h[:, N + 1:])
            pos_p = k * torch.arange(1, J, device=ids.device) + r
            out[:, pos_p] = chunked_nll(logits_p, ids[:, pos_p])
    return out


@torch.no_grad()
def check_walker(m: LoadedModel, ids: torch.Tensor, layers) -> float:
    """Max |logit| difference between the explicit layer walker and the model's own body."""
    with autocast(str(ids.device)):
        emb = m.backbone.embed_in(ids)
        G = m.pool(emb, ids)[:, :-1]
        ref = m.backbone.embed_out(m.backbone.body(G)).float()
        pos = torch.arange(G.size(1), device=ids.device)
        alt = m.backbone.embed_out(forward_hidden(m.backbone, G, pos, layers=layers)).float()
    return (ref - alt).abs().max().item()


def summarize(nll: torch.Tensor, sel: torch.Tensor, k: int, positions: torch.Tensor) -> dict:
    out = {"all": nll[sel].mean().item(), "n": int(sel.sum())}
    for r in range(k):
        s = sel & ((positions % k) == r).view(1, -1)
        out[f"r{r}"] = nll[s].mean().item() if s.any() else None
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", action="append", required=True,
                   metavar=("CONFIG", "CKPT"), help="CONFIG CKPT [LABEL] (k>1 runs)")
    p.add_argument("--baseline", nargs=2, default=None, metavar=("CONFIG", "CKPT"),
                   help="Optional k=1 model scored on the same positions.")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True, help="JSONL output")
    p.add_argument("--modes", default="renorm,repeat_last,norm_matched")
    p.add_argument("--max_seqs", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default=default_device())
    p.add_argument("--download_dir", default="/tmp/ta_ckpts")
    args = p.parse_args()

    modes = [s.strip() for s in args.modes.split(",") if s.strip()]
    base = None
    if args.baseline:
        base = load_model(args.baseline[0], args.baseline[1], device=args.device,
                          download_dir=args.download_dir)

    for entry in args.model:
        cfg_name, ck = entry[0], entry[1]
        label = entry[2] if len(entry) > 2 else cfg_name
        t0 = time.time()
        m = load_model(cfg_name, ck, device=args.device, download_dir=args.download_dir)
        if m.k == 1:
            print(f"[skip] {label} is k=1", flush=True)
            continue
        layers = layer_modules(m.backbone)
        seq_len = m.cfg.context_len
        positions = torch.arange(seq_len, device=args.device)

        sums: dict[str, list[torch.Tensor]] = {"ensemble": [], "baseline": []}
        for mode in modes:
            sums[f"single_{mode}"] = []
        checked = False
        for ids in eval_batches(args.data_dir, seq_len, args.batch_size, args.max_seqs):
            ids = ids.to(args.device)
            if not checked:
                err = check_walker(m, ids[:1], layers)
                print(f"  walker vs body max |dlogit| = {err:.3e}", flush=True)
                checked = True
            sums["ensemble"].append(per_position_nll(m, ids).cpu())
            for mode in modes:
                sums[f"single_{mode}"].append(single_cache_nll(m, ids, mode, layers).cpu())
            if base is not None:
                sums["baseline"].append(per_position_nll(base, ids).cpu())

        cat = {kk: torch.cat(v) for kk, v in sums.items() if v}
        common = torch.ones_like(cat["ensemble"], dtype=torch.bool)
        for v in cat.values():
            common &= ~torch.isnan(v)
        pos_cpu = positions.cpu()
        row = {"label": label, "config": cfg_name, "ckpt": str(ck), "k": m.k,
               "pooling": m.method_cfg.name, "n_seqs": int(cat["ensemble"].size(0)),
               "walker_max_abs_logit_diff": err}
        for kk, v in cat.items():
            row[kk] = summarize(v, common, m.k, pos_cpu)
        row["elapsed_s"] = time.time() - t0
        append_jsonl(args.out, row)

        print(f"\n{label} (k={m.k}, {row['n_seqs']} seqs, {row['ensemble']['n']:,} common targets)")
        cols = ["all"] + [f"r{r}" for r in range(m.k)]
        print(f"  {'method':24s} " + " ".join(f"{c:>8s}" for c in cols))
        for kk in cat:
            vals = [row[kk][c] for c in cols]
            print(f"  {kk:24s} " + " ".join(f"{v:8.4f}" if v is not None else f"{'-':>8s}"
                                             for v in vals))
        del m
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
