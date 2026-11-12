"""
Shared helpers for the rebuttal experiments.

- Checkpoint resolution (local .pt, local run dir, or hf:<repo>[:<file>]) and a
  loader that rebuilds the exact pooling rule a run was trained with. It strips
  the `_orig_mod.` prefix that torch.compile adds and handles two-stage
  (PLT/TST) runs, whose final model is an ordinary k=1 model.
- Offset-ensemble scoring with the real pooling (per-position NLL).
- An explicit per-layer forward over the OLM transformer that accepts RoPE
  positions, an attention mask, and a static KV cache. Used for the
  partial-group evaluation and the decoding benchmark.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chinchilla.model_configs import ModelConfig, get_config
from experiments.shared.averaged_lm import MethodConfig, build_method_config
from experiments.shared.olm_model import OLMAveragedLanguageModel, OLMTransformerBody
from utils.averaging_methods import compute_weights

VOCAB_FALLBACK = 50_277
EVAL_DTYPE = np.uint16
GRID_FAMILIES = ("uniform", "weighted", "learnable", "learnable_positional")


# ---------------------------------------------------------------------------
# Devices / numerics
# ---------------------------------------------------------------------------

def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def autocast(device: str):
    dt = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.autocast(device_type=dt, dtype=torch.bfloat16, enabled=(dt == "cuda"))


def chunked_nll(logits: torch.Tensor, labels: torch.Tensor, chunk_rows: int = 2048) -> torch.Tensor:
    """Per-token NLL in fp32 without materialising an fp32 copy of all logits."""
    shape = labels.shape
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    out = torch.empty(flat_labels.shape, dtype=torch.float32, device=logits.device)
    for i in range(0, flat_labels.numel(), chunk_rows):
        sl = slice(i, min(i + chunk_rows, flat_labels.numel()))
        out[sl] = F.cross_entropy(flat_logits[sl].float(), flat_labels[sl], reduction="none")
    return out.view(shape)


# ---------------------------------------------------------------------------
# Eval data
# ---------------------------------------------------------------------------

def eval_seq_count(data_dir: str | Path, seq_len: int) -> int:
    data = np.memmap(Path(data_dir) / "eval.bin", dtype=EVAL_DTYPE, mode="r")
    return (len(data) - 1) // seq_len


def eval_batches(
    data_dir: str | Path,
    seq_len: int,
    batch_size: int,
    max_seqs: Optional[int] = None,
) -> Iterator[torch.Tensor]:
    """Sequential, non-overlapping eval sequences from eval.bin (the split every run used)."""
    path = Path(data_dir) / "eval.bin"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; pass --data_dir with eval.bin")
    data = np.memmap(path, dtype=EVAL_DTYPE, mode="r")
    n_seqs = (len(data) - 1) // seq_len
    if max_seqs is not None:
        n_seqs = min(n_seqs, max_seqs)
    for start in range(0, n_seqs, batch_size):
        idx = range(start, min(start + batch_size, n_seqs))
        yield torch.from_numpy(
            np.stack([data[i * seq_len:(i + 1) * seq_len].astype(np.int64) for i in idx])
        )


def train_batches(
    data_dir: str | Path,
    seq_len: int,
    batch_size: int,
    seed: int = 0,
) -> Iterator[torch.Tensor]:
    """Random training sequences from train.bin (for short fine-tunes)."""
    data = np.memmap(Path(data_dir) / "train.bin", dtype=EVAL_DTYPE, mode="r")
    n_seqs = (len(data) - 1) // seq_len
    rng = np.random.default_rng(seed)
    while True:
        idx = rng.integers(0, n_seqs, size=batch_size)
        yield torch.from_numpy(
            np.stack([data[i * seq_len:(i + 1) * seq_len].astype(np.int64) for i in idx])
        )


# ---------------------------------------------------------------------------
# Checkpoint resolution and loading
# ---------------------------------------------------------------------------

def resolve_checkpoint(spec: str, download_dir: str | Path = "/tmp/ta_ckpts") -> Path:
    """
    spec forms:
      /path/to/file.pt
      /path/to/run_dir                 -> run_dir/checkpoints/final.pt
      hf:<owner>/<repo>[:<file>]       -> downloaded; default file checkpoints/final.pt
    """
    if spec.startswith("hf:"):
        from huggingface_hub import hf_hub_download

        body = spec[3:]
        repo, _, filename = body.partition(":")
        filename = filename or "checkpoints/final.pt"
        local = Path(download_dir) / repo.replace("/", "__")
        path = Path(hf_hub_download(repo_id=repo, filename=filename, local_dir=str(local)))
        if path.stat().st_size < 10_000:
            raise RuntimeError(f"{spec}: file is a broken LFS pointer")
        return path
    p = Path(spec)
    if p.is_dir():
        p = p / "checkpoints" / "final.pt"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _strip(sd: dict, prefix: str) -> dict:
    if not any(k.startswith(prefix) for k in sd):
        return sd
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}


def _vocab_and_tied(bb_sd: dict) -> tuple[int, bool]:
    win = bb_sd.get("embed_in.embedding.weight")
    wout = bb_sd.get("embed_out.blocks.1.weight")
    ref = win if win is not None else wout
    vocab = int(ref.shape[0]) if ref is not None else VOCAB_FALLBACK
    tied = win is None or wout is None or torch.equal(win, wout)
    return vocab, tied


@dataclass
class LoadedModel:
    name: str
    cfg: ModelConfig
    k: int
    backbone: OLMTransformerBody
    lm: Optional[nn.Module]
    method_cfg: Optional[MethodConfig]
    ckpt_path: Optional[Path]
    meta: dict = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def pool(self, emb: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Apply the run's own pooling rule: [B, N*k, D] -> [B, N, D]."""
        if self.k == 1:
            return emb
        return self.method_cfg.avg_fn(emb, ids)[0]


def read_run_meta(ckpt_path: Path) -> dict:
    for cand in (ckpt_path.parent.parent / "run_meta.json", ckpt_path.parent / "run_meta.json"):
        if cand.exists():
            try:
                return json.loads(cand.read_text())
            except Exception:
                return {}
    return {}


def load_model(
    config_name: str,
    ckpt: Optional[str | Path],
    device: str = "cpu",
    context_len: Optional[int] = None,
    download_dir: str | Path = "/tmp/ta_ckpts",
    verbose: bool = True,
) -> LoadedModel:
    """
    Rebuild and load a trained model. ckpt=None returns a randomly initialised
    model (used by the throughput benchmarks).

    Two-stage runs (run_meta.json has two_stage_k) are loaded as k=1 models:
    their final weights are an ordinary next-token model.
    """
    cfg = get_config(config_name)
    ckpt_path = resolve_checkpoint(str(ckpt), download_dir) if ckpt is not None else None
    run_meta = read_run_meta(ckpt_path) if ckpt_path is not None else {}

    sd, meta = None, {}
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        meta = {kk: state[kk] for kk in ("step", "tokens_seen", "cumulative_flops", "flops_eq2")
                if isinstance(state, dict) and kk in state}
        sd = _strip(sd, "_orig_mod.")
        sd = _strip(sd, "module.")

    two_stage = bool(run_meta.get("two_stage_k"))
    k = 1 if two_stage else cfg.averaging_k

    bb_sd = _strip(sd, "backbone.") if sd is not None else {}
    if sd is not None:
        bb_sd = {kk: v for kk, v in bb_sd.items() if not kk.startswith("learnable_averager.")}
    vocab, tied = _vocab_and_tied(bb_sd) if sd is not None else (VOCAB_FALLBACK, cfg.tie_embeddings)

    backbone = OLMTransformerBody(
        vocab_size=vocab, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_layers=cfg.n_layers, context_length=context_len or cfg.context_len,
    )
    if tied:
        backbone.tie_embedding_weights()

    lm, method_cfg = None, None
    if k > 1:
        method_cfg = build_method_config(cfg.method_name or f"uniform_k{k}")
        lm = OLMAveragedLanguageModel(backbone, method_cfg)
        method_cfg = lm.cfg

    if sd is not None:
        target = lm if lm is not None else backbone
        load_sd = sd if lm is not None else bb_sd
        if lm is not None and not any(kk.startswith("backbone.") for kk in load_sd):
            load_sd = {("backbone." + kk if not kk.startswith("learnable_averager.") else kk): v
                       for kk, v in load_sd.items()}
        missing, unexpected = target.load_state_dict(load_sd, strict=False)
        missing = [m for m in missing if "embed_out.blocks.1.weight" not in m]
        if missing:
            raise RuntimeError(f"missing keys loading {ckpt_path}: {missing[:8]}")
        if unexpected and verbose:
            print(f"  [warn] ignored unexpected keys: {list(unexpected)[:6]}", flush=True)

    backbone.to(device).eval()
    if lm is not None:
        lm.to(device).eval()
    if verbose:
        src = ckpt_path if ckpt_path is not None else "random init"
        print(f"[load] {config_name} k={k} pooling={method_cfg.name if method_cfg else 'none'} "
              f"tied={tied} vocab={vocab} two_stage={two_stage} <- {src}", flush=True)
    return LoadedModel(
        name=config_name, cfg=cfg, k=k, backbone=backbone, lm=lm, method_cfg=method_cfg,
        ckpt_path=ckpt_path, meta={**meta, "run_meta": run_meta},
    )


# ---------------------------------------------------------------------------
# Offset-ensemble scoring (true pooling)
# ---------------------------------------------------------------------------

@torch.no_grad()
def offset_logits(
    m: LoadedModel, ids: torch.Tensor, offset: int, pad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One pass of the averaged model on ids[:, offset:].
    Returns (logits [B, J, V], target_positions [J]) where target_positions index
    into ids: logits[:, j] predicts ids[:, target_positions[j]].

    pad=True appends k EOS tokens so the last complete group of real tokens is
    not the (dropped) final group; the padding only ever sits after the scored
    prefix, so causal attention keeps it out of every kept prediction.
    """
    k = m.k
    if m.method_cfg.method_family not in GRID_FAMILIES:
        raise NotImplementedError(
            f"offset scoring assumes non-overlapping k-groups; {m.method_cfg.name} "
            f"({m.method_cfg.method_family}) has a different label layout")
    T = ids.size(1)
    ids_o = ids[:, offset:]
    if pad:
        ids_o = torch.cat([ids_o, torch.zeros_like(ids_o[:, :k])], dim=1)
    hidden = m.backbone.embed_in(ids_o)
    pooled = m.pool(hidden, ids_o)
    logits = m.backbone.embed_out(m.backbone.body(pooled[:, :-1]))
    targets = offset + k * torch.arange(1, pooled.size(1), device=ids.device)
    keep = targets < T
    return logits[:, : int(keep.sum())], targets[keep]


@torch.no_grad()
def per_position_nll(m: LoadedModel, ids: torch.Tensor, pad: bool = False) -> torch.Tensor:
    """
    [B, T] NLL of each token under the model's standard scoring: next-token for
    k=1, offset ensemble for k>1. Uncovered positions are NaN. pad=False matches
    the eval_loss_all_pos column train.py logs.
    """
    B, T = ids.shape
    out = torch.full((B, T), float("nan"), device=ids.device)
    with autocast(str(ids.device)):
        if m.k == 1:
            logits = m.backbone(ids)[:, :-1]
            out[:, 1:] = chunked_nll(logits, ids[:, 1:])
            return out
        for o in range(m.k):
            logits, pos = offset_logits(m, ids, o, pad=pad)
            if pos.numel():
                out[:, pos] = chunked_nll(logits, ids[:, pos])
    return out


@torch.no_grad()
def full_position_logprobs(m: LoadedModel, ids: torch.Tensor) -> torch.Tensor:
    """
    [B, T-1, V] log-probs where row t predicts ids[:, t+1], filled by the
    offset ensemble for k>1 (with end padding). Rows not covered by any pass
    are -inf. Same contract as experiments/benchmarks/eval_downstream.get_logprobs.
    """
    B, T = ids.shape
    with autocast(str(ids.device)):
        if m.k == 1:
            return F.log_softmax(m.backbone(ids)[:, :-1].float(), dim=-1)
        V = m.backbone.vocab_size
        out = torch.full((B, T - 1, V), float("-inf"), device=ids.device)
        for o in range(m.k):
            if T - o < m.k:
                continue
            logits, pos = offset_logits(m, ids, o, pad=True)
            if pos.numel():
                out[:, pos - 1] = F.log_softmax(logits.float(), dim=-1)
    return out


# ---------------------------------------------------------------------------
# Explicit per-layer forward (positions, masks, KV cache)
# ---------------------------------------------------------------------------

def layer_modules(backbone: OLMTransformerBody) -> list[tuple[nn.Module, nn.Module, nn.Module, nn.Module]]:
    """[(ln_attn, attn, ln_ffn, ffn)] per layer, following OLM's TransformerBlock layout."""
    layers = []
    for layer in backbone.body.stack:
        inner = layer.blocks[0].blocks
        ln1, attn = inner[0].block.blocks
        ln2, ffn = inner[1].block.blocks
        layers.append((ln1, attn, ln2, ffn))
    return layers


def _qkv(attn: nn.Module, x: torch.Tensor, positions: torch.Tensor):
    B, N, _ = x.shape
    H, hd = attn.num_heads, attn.head_dim
    q = attn.q_proj(x).view(B, N, H, hd)
    k = attn.k_proj(x).view(B, N, H, hd)
    v = attn.v_proj(x).view(B, N, H, hd)
    pos = positions.view(1, N).expand(B, N)
    q = attn.rope(q, pos).transpose(1, 2)
    k = attn.rope(k, pos).transpose(1, 2)
    return q, k, v.transpose(1, 2)


def _attn_out(attn: nn.Module, o: torch.Tensor) -> torch.Tensor:
    B, H, N, hd = o.shape
    return attn.out_proj(o.transpose(1, 2).reshape(B, N, H * hd))


def forward_hidden(
    backbone: OLMTransformerBody,
    x: torch.Tensor,
    positions: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    layers=None,
) -> torch.Tensor:
    """
    Transformer body on input vectors x [B, N, D] with explicit RoPE positions [N]
    and a boolean attn_mask [N, N] (True = may attend). attn_mask=None is causal.
    Returns final hidden states before the output head.
    """
    layers = layers or layer_modules(backbone)
    for ln1, attn, ln2, ffn in layers:
        q, k, v = _qkv(attn, ln1(x), positions)
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None)
        x = x + _attn_out(attn, o)
        x = x + ffn(ln2(x))
    return x


class StaticKVCache:
    """Preallocated per-layer K/V of shape [L, B, H, max_len, hd]; `length` = committed slots."""

    def __init__(self, n_layers: int, batch: int, n_heads: int, max_len: int, head_dim: int,
                 device, dtype):
        self.k = torch.zeros(n_layers, batch, n_heads, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.length = 0
        self.max_len = max_len

    @property
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    def used_bytes(self) -> int:
        per_slot = self.k[:, :, :, :1].numel() * self.k.element_size() * 2
        return per_slot * self.length


@torch.no_grad()
def cached_forward(
    backbone: OLMTransformerBody,
    x: torch.Tensor,
    cache: StaticKVCache,
    commit: bool = True,
    layers=None,
) -> torch.Tensor:
    """
    Run x [B, n, D] at cache slots cache.length .. cache.length+n-1 (RoPE position
    = slot index). With commit=False the slots are written but cache.length is not
    advanced, so the next call overwrites them (tentative partial group).
    n > 1 is only supported when the cache is empty (prefill).
    """
    layers = layers or layer_modules(backbone)
    start, n = cache.length, x.size(1)
    if n > 1 and start != 0:
        raise ValueError("multi-position cached_forward only supported for prefill")
    if start + n > cache.max_len:
        raise ValueError("KV cache overflow")
    positions = torch.arange(start, start + n, device=x.device)
    for li, (ln1, attn, ln2, ffn) in enumerate(layers):
        q, k, v = _qkv(attn, ln1(x), positions)
        cache.k[li, :, :, start:start + n] = k
        cache.v[li, :, :, start:start + n] = v
        K = cache.k[li, :, :, :start + n]
        V = cache.v[li, :, :, :start + n]
        o = F.scaled_dot_product_attention(q, K, V, is_causal=(n > 1))
        x = x + _attn_out(attn, o)
        x = x + ffn(ln2(x))
    if commit:
        cache.length = start + n
    return x


# ---------------------------------------------------------------------------
# Partial-group pooling
# ---------------------------------------------------------------------------

def _fixed_position_weights(m: LoadedModel) -> Optional[torch.Tensor]:
    name = m.method_cfg.name if m.method_cfg else ""
    if name.startswith("weighted_"):
        parts = name.split("_")
        return torch.tensor(compute_weights(int(parts[-1][1:]), "_".join(parts[1:-1])),
                            dtype=torch.float32)
    return None


def pool_partial(
    m: LoadedModel,
    emb_windows: torch.Tensor,
    r: int,
    mode: str = "renorm",
    ref_norm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pool the first r tokens of each window. emb_windows: [B, J, r, D].

    renorm       : the run's pooling rule restricted to the r present slots and
                   renormalised (mean for uniform; w[:r]/sum for fixed weights;
                   softmax over present scores for learned pooling).
    repeat_last  : pad each window to k by repeating the last present token and
                   apply the full k-slot rule.
    norm_matched : renorm, then rescale each vector to ref_norm (mean norm of
                   complete groups) to undo the larger norm of short averages.
    """
    B, J, _, D = emb_windows.shape
    k = m.k
    if mode == "repeat_last":
        pad = emb_windows[:, :, -1:].expand(B, J, k - r, D)
        full = torch.cat([emb_windows, pad], dim=2).reshape(B, J * k, D)
        dummy = torch.zeros(B, J * k, dtype=torch.long, device=emb_windows.device)
        return m.pool(full, dummy)

    module = m.method_cfg.learnable_module if m.method_cfg else None
    fixed = _fixed_position_weights(m)
    if module is not None:
        scores = module.scorer(emb_windows).squeeze(-1)
        if hasattr(module, "position_logits"):
            scores = scores + module.position_logits[:r].view(1, 1, r)
        w = torch.softmax(scores.float(), dim=-1).to(emb_windows.dtype)
        pooled = (emb_windows * w.unsqueeze(-1)).sum(dim=2)
    elif fixed is not None:
        w = fixed[:r] / fixed[:r].sum()
        pooled = (emb_windows * w.to(emb_windows).view(1, 1, r, 1)).sum(dim=2)
    else:
        pooled = emb_windows.mean(dim=2)

    if mode == "norm_matched":
        if ref_norm is None:
            raise ValueError("norm_matched needs ref_norm")
        pooled = pooled * (ref_norm / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    elif mode != "renorm":
        raise ValueError(f"unknown partial pooling mode {mode}")
    return pooled


# ---------------------------------------------------------------------------
# FLOPs (Eq. 2 of the paper)
# ---------------------------------------------------------------------------

def eq2_flops_per_raw_token(cfg: ModelConfig, seq_len: int, k: int, vocab: int = 50_304) -> float:
    L = seq_len // k
    d = cfg.d_model
    return 3.0 / k * (cfg.n_layers * (23 * d * d + 4 * L * d) + 2 * d * vocab)


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def append_jsonl(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=float) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows
