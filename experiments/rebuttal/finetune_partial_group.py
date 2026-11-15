"""
Short continued training that teaches a k>1 model to predict from a tentative
partial group, so that single-cache decoding (T/k KV entries) is in
distribution.

Loss per step = standard averaged loss (random offset, as in pretraining)
              + lam * partial-group loss for one random r in 1..k-1
The partial-group term uses the same masked parallel pass as
eval_partial_group.py. The result is saved as <out_dir>/checkpoints/final.pt
with a run_meta.json, so every other script can load it as a normal run of the
same config. Afterwards, run eval_partial_group.py on it and rescore.py to check
that the ordinary offset-ensemble loss did not degrade.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.rebuttal.common import (
    autocast, default_device, eq2_flops_per_raw_token, forward_hidden, layer_modules,
    load_model, pool_partial, train_batches,
)
from experiments.rebuttal.eval_partial_group import partial_mask


def partial_loss(m, ids: torch.Tensor, r: int, mode: str, layers) -> torch.Tensor:
    B, T = ids.shape
    k = m.k
    N = T // k
    emb = m.backbone.embed_in(ids)
    G = m.pool(emb[:, : N * k], ids[:, : N * k])
    J = min((T - 1 - r) // k + 1, N)
    windows = emb[:, : J * k].reshape(B, J, k, -1)[:, :, :r]
    ref_norm = G.detach().float().norm(dim=-1).mean() if mode == "norm_matched" else None
    P = pool_partial(m, windows, r, mode=mode, ref_norm=ref_norm).to(G.dtype)
    X = torch.cat([G, P], dim=1)
    positions = torch.cat([torch.arange(N, device=ids.device), torch.arange(J, device=ids.device)])
    h = forward_hidden(m.backbone, X, positions, attn_mask=partial_mask(N, J, ids.device),
                       layers=layers)
    logits = m.backbone.embed_out(h[:, N + 1:])
    targets = ids[:, k * torch.arange(1, J, device=ids.device) + r]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--mode", default="renorm", choices=["renorm", "repeat_last", "norm_matched"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_steps", type=int, default=100)
    p.add_argument("--device", default=default_device())
    p.add_argument("--download_dir", default="/tmp/ta_ckpts")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if (out_dir / "checkpoints" / "final.pt").exists():
        print(f"{out_dir} already has final.pt; nothing to do", flush=True)
        return

    m = load_model(args.config, args.ckpt, device=args.device, download_dir=args.download_dir)
    if m.k == 1:
        raise SystemExit("partial-group fine-tuning needs a k>1 model")
    lm = m.lm
    lm.train()
    layers = layer_modules(m.backbone)
    seq_len = m.cfg.context_len

    decay = [q for q in lm.parameters() if q.requires_grad and q.dim() >= 2]
    no_decay = [q for q in lm.parameters() if q.requires_grad and q.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    def lr_at(s: int) -> float:
        if s < args.warmup:
            return (s + 1) / args.warmup
        prog = (s - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    data = train_batches(args.data_dir, seq_len, args.batch_size, seed=1000 + args.seed)
    log_path = out_dir / "finetune_log.csv"
    log = open(log_path, "w")
    log.write("step,loss_std,loss_partial\n")
    t0 = time.time()
    ema_s = ema_p = None
    for step in range(1, args.steps + 1):
        ids = next(data).to(args.device)
        r = random.randint(1, m.k - 1)
        with autocast(args.device):
            loss_std, _ = lm(ids)
            if args.lam > 0:
                loss_p = partial_loss(m, ids, r, args.mode, layers)
                loss = loss_std + args.lam * loss_p
            else:
                loss_p = torch.full_like(loss_std, float("nan"))
                loss = loss_std
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
        opt.step()
        sched.step()
        ls, lp = loss_std.item(), loss_p.item()
        ema_s = ls if ema_s is None else 0.98 * ema_s + 0.02 * ls
        ema_p = lp if ema_p is None else 0.98 * ema_p + 0.02 * lp
        if step % args.log_steps == 0 or step == args.steps:
            log.write(f"{step},{ema_s:.5f},{ema_p:.5f}\n")
            log.flush()
            print(f"step {step}/{args.steps}  std {ema_s:.4f}  partial {ema_p:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s", flush=True)
    log.close()

    tokens = args.steps * args.batch_size * seq_len
    base_meta = m.meta.get("run_meta") or {}
    meta = {
        **base_meta,
        "run_name": out_dir.name,
        "base_config": args.config,
        "finetune_of": str(args.ckpt),
        "finetune": {"objective": "standard + partial-group" if args.lam > 0 else "standard",
                     "steps": args.steps, "warmup": args.warmup,
                     "batch_size": args.batch_size, "lr": args.lr, "lam": args.lam,
                     "mode": args.mode, "tokens": tokens,
                     "approx_flops_eq2": tokens * eq2_flops_per_raw_token(m.cfg, seq_len, m.k) * 3},
        "raw_seq_len": seq_len,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    torch.save({"model": lm.state_dict(), "step": args.steps,
                "tokens_seen": (m.meta.get("tokens_seen") or 0) + tokens},
               out_dir / "checkpoints" / "final.pt")
    print(f"saved {out_dir / 'checkpoints' / 'final.pt'}", flush=True)


if __name__ == "__main__":
    main()
