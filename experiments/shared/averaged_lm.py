"""
Core wrapper and method registry for LLM token-averaging experiments.

AveragedLanguageModel inserts an averaging step between the Pythia embedding
lookup (gpt_neox.embed_in) and the transformer layers, using inputs_embeds to
bypass the model's internal embedding call.  The same class drives all three
experiments (zero-shot, finetune, from-scratch).

Method registry
---------------
build_method_config(name)  →  MethodConfig dataclass
get_all_config_names()     →  list[str]
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make project root importable when script is run directly
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.averaging_methods import (
    apply_dynamic_averaging,
    apply_overlapping_averaging,
    apply_weighted_averaging,
    compute_weights,
    LearnableAverager,
)


# ---------------------------------------------------------------------------
# MethodConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class MethodConfig:
    """Describes one averaging configuration used inside AveragedLanguageModel."""
    name: str                        # registry key, e.g. "uniform_k2"
    method_family: str               # "uniform" | "dynamic" | "overlapping" | "weighted" | "learnable" | "baseline"
    nominal_k: int                   # representative window size (for labelling / CSV)
    compression_ratio: float         # fraction of tokens removed  (0 = no compression)
    # called as avg_fn(hidden [B,T,D]) → (averaged [B,T',D], label_indices [T'-1])
    # label_indices are 1-D int positions into the ORIGINAL sequence (per-sequence, not per-batch)
    avg_fn: Callable                 # filled by build_method_config
    # optional: LearnableAverager instance (None for all other methods)
    learnable_module: Optional[LearnableAverager] = None


# ---------------------------------------------------------------------------
# Label construction helpers
# ---------------------------------------------------------------------------

def _uniform_labels(input_ids: torch.Tensor, k: int) -> torch.Tensor:
    """
    Non-overlapping windows of fixed size k.
    Label at compressed position j = input_ids[(j+1)*k]  (first token of next window).
    Returns LongTensor [B, T//k - 1].
    """
    return input_ids[:, k::k]


def _overlapping_labels(input_ids: torch.Tensor, w: int, s: int) -> torch.Tensor:
    """
    Overlapping average-pool with window w and stride s.
    Label at output position j = input_ids[j*s + w]  (first token beyond current window).
    Returns LongTensor [B, T'].
    """
    T = input_ids.size(1)
    T_out = (T - w) // s + 1
    # positions: w, w+s, w+2s, ...  (T_out - 1 labels because last output position has no next)
    positions = [j * s + w for j in range(T_out - 1) if j * s + w < T]
    idx = torch.tensor(positions, dtype=torch.long, device=input_ids.device)
    return input_ids[:, idx]


def _dynamic_labels(input_ids: torch.Tensor, groups: list) -> torch.Tensor:
    """
    Variable-length groups from apply_dynamic_averaging.
    Label at group j = input_ids[groups[j+1][0]]  (first token of next group).
    Returns LongTensor [1, n_groups - 1]  (always batch=1 for dynamic).
    """
    positions = [groups[j + 1][0] for j in range(len(groups) - 1)]
    idx = torch.tensor(positions, dtype=torch.long, device=input_ids.device)
    return input_ids[:, idx]


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def build_method_config(name: str, learnable_checkpoint_dir: Optional[str] = None) -> MethodConfig:
    """
    Build a MethodConfig for the given registry name.

    Args:
        name: one of the keys in METHOD_REGISTRY_KEYS
        learnable_checkpoint_dir: path containing averager_k{k}.pt files
            (required only for "learnable_*" configs in zero-shot mode)
    """
    # ---- baseline (k=1, no averaging) ----
    if name == "baseline_k1":
        def _avg_fn(hidden, input_ids):
            # No compression: labels = standard shifted tokens
            labels = input_ids[:, 1:]                              # [B, T-1]
            return hidden, labels
        return MethodConfig(
            name=name, method_family="baseline",
            nominal_k=1, compression_ratio=0.0,
            avg_fn=_avg_fn,
        )

    # ---- uniform ----
    if name.startswith("uniform_k"):
        k = int(name.split("_k")[1])
        def _avg_fn(hidden, input_ids, _k=k):
            B, T, D = hidden.shape
            n = T // _k
            avg = hidden[:, :n * _k].reshape(B, n, _k, D).mean(dim=2)  # [B, n, D]
            labels = _uniform_labels(input_ids, _k)                      # [B, n-1]
            return avg, labels
        return MethodConfig(
            name=name, method_family="uniform",
            nominal_k=k, compression_ratio=1.0 - 1.0 / k,
            avg_fn=_avg_fn,
        )

    # ---- dynamic alternating ----
    if name == "dynamic_alt23":
        def _avg_fn(hidden, input_ids):
            # Process each sequence individually (variable output length)
            results, label_list = [], []
            for i in range(hidden.size(0)):
                avg_i, groups = apply_dynamic_averaging(
                    hidden[i].unsqueeze(0), strategy="alternating", pattern=[2, 3]
                )
                lbl_i = _dynamic_labels(input_ids[i].unsqueeze(0), groups)
                results.append(avg_i)
                label_list.append(lbl_i)
            # Pad to the same length so we can stack
            min_len = min(r.size(1) for r in results)
            avg_out = torch.cat([r[:, :min_len] for r in results], dim=0)
            lbl_out = torch.cat([l[:, :min_len - 1] for l in label_list], dim=0)
            return avg_out, lbl_out
        return MethodConfig(
            name=name, method_family="dynamic",
            nominal_k=2, compression_ratio=1.0 - 2.0 / 5.0,  # avg group≈2.5
            avg_fn=_avg_fn,
        )

    # ---- dynamic random [2,4] ----
    if name == "dynamic_rnd24":
        def _avg_fn(hidden, input_ids):
            results, label_list = [], []
            for i in range(hidden.size(0)):
                avg_i, groups = apply_dynamic_averaging(
                    hidden[i].unsqueeze(0), strategy="random", k_min=2, k_max=4
                )
                lbl_i = _dynamic_labels(input_ids[i].unsqueeze(0), groups)
                results.append(avg_i)
                label_list.append(lbl_i)
            min_len = min(r.size(1) for r in results)
            avg_out = torch.cat([r[:, :min_len] for r in results], dim=0)
            lbl_out = torch.cat([l[:, :min_len - 1] for l in label_list], dim=0)
            return avg_out, lbl_out
        return MethodConfig(
            name=name, method_family="dynamic",
            nominal_k=3, compression_ratio=1.0 - 1.0 / 3.0,
            avg_fn=_avg_fn,
        )

    # ---- dynamic random [2,8] ----
    if name == "dynamic_rnd28":
        def _avg_fn(hidden, input_ids):
            results, label_list = [], []
            for i in range(hidden.size(0)):
                avg_i, groups = apply_dynamic_averaging(
                    hidden[i].unsqueeze(0), strategy="random", k_min=2, k_max=8
                )
                lbl_i = _dynamic_labels(input_ids[i].unsqueeze(0), groups)
                results.append(avg_i)
                label_list.append(lbl_i)
            min_len = min(r.size(1) for r in results)
            avg_out = torch.cat([r[:, :min_len] for r in results], dim=0)
            lbl_out = torch.cat([l[:, :min_len - 1] for l in label_list], dim=0)
            return avg_out, lbl_out
        return MethodConfig(
            name=name, method_family="dynamic",
            nominal_k=5, compression_ratio=1.0 - 1.0 / 5.0,
            avg_fn=_avg_fn,
        )

    # ---- overlapping (window w, stride s) ----
    if name.startswith("overlap_w"):
        parts = name.split("_")          # ["overlap", "w2", "s2"]
        w = int(parts[1][1:])
        s = int(parts[2][1:])
        def _avg_fn(hidden, input_ids, _w=w, _s=s):
            avg = apply_overlapping_averaging(hidden, window_size=_w, stride=_s)
            labels = _overlapping_labels(input_ids, _w, _s)
            # Align: avg has T' rows, labels has T'-1 rows
            min_len = min(avg.size(1) - 1, labels.size(1))
            return avg, labels[:, :min_len]
        compression = 1.0 - s / w
        return MethodConfig(
            name=name, method_family="overlapping",
            nominal_k=w, compression_ratio=compression,
            avg_fn=_avg_fn,
        )

    # ---- weighted (scheme, k) ----
    if name.startswith("weighted_"):
        parts = name.split("_")          # ["weighted", "exp", "k2"] or ["weighted", "uniform", "k2"]
        scheme_parts = parts[1:-1]       # everything between "weighted_" and "_k{N}"
        scheme = "_".join(scheme_parts)
        k = int(parts[-1][1:])
        weights = compute_weights(k, scheme)
        def _avg_fn(hidden, input_ids, _k=k, _w=weights):
            avg = apply_weighted_averaging(hidden, k=_k, weights=_w)
            labels = _uniform_labels(input_ids, _k)
            return avg, labels
        return MethodConfig(
            name=name, method_family="weighted",
            nominal_k=k, compression_ratio=1.0 - 1.0 / k,
            avg_fn=_avg_fn,
        )

    # ---- learnable ----
    if name.startswith("learnable_k"):
        k = int(name.split("_k")[1])
        # Hidden dim for Pythia family — will be patched by AveragedLanguageModel
        # after the base model is loaded (we don't know hidden_dim at registry time).
        averager = LearnableAverager(hidden_dim=512, k=k)  # placeholder dim

        if learnable_checkpoint_dir is not None:
            ckpt = os.path.join(learnable_checkpoint_dir, f"averager_k{k}.pt")
            if os.path.exists(ckpt):
                averager.load_state_dict(torch.load(ckpt, map_location="cpu"))

        def _avg_fn(hidden, input_ids, _averager=averager, _k=k):
            avg, _ = _averager(hidden)
            labels = _uniform_labels(input_ids, _k)
            return avg, labels

        return MethodConfig(
            name=name, method_family="learnable",
            nominal_k=k, compression_ratio=1.0 - 1.0 / k,
            avg_fn=_avg_fn,
            learnable_module=averager,
        )

    raise ValueError(
        f"Unknown method config name: '{name}'. "
        f"Valid names: {get_all_config_names()}"
    )


def get_all_config_names() -> list:
    return [
        "baseline_k1",
        "uniform_k2", "uniform_k4", "uniform_k8",
        "dynamic_alt23", "dynamic_rnd24", "dynamic_rnd28",
        "overlap_w2s1", "overlap_w2s2", "overlap_w4s2", "overlap_w4s4",
        "weighted_uniform_k2", "weighted_uniform_k4", "weighted_uniform_k8",
        "weighted_linear_k2", "weighted_linear_k4", "weighted_linear_k8",
        "weighted_exponential_k2", "weighted_exponential_k4", "weighted_exponential_k8",
        "weighted_gaussian_k2", "weighted_gaussian_k4", "weighted_gaussian_k8",
        "weighted_triangular_k2", "weighted_triangular_k4", "weighted_triangular_k8",
        "learnable_k2", "learnable_k4", "learnable_k8",
    ]


# ---------------------------------------------------------------------------
# AveragedLanguageModel
# ---------------------------------------------------------------------------

class AveragedLanguageModel(nn.Module):
    """
    Wraps a pretrained Pythia (GPT-NeoX) AutoModelForCausalLM and inserts a
    chosen averaging step between the embedding lookup and the transformer.

    The base model's gpt_neox.embed_in is called manually; the resulting
    (averaged) hidden states are fed back via inputs_embeds, bypassing the
    internal embedding call inside gpt_neox.forward().

    Works for all three experiment modes:
      - zero-shot : base model weights frozen, no LearnableAverager training
      - finetune  : all weights (including optional LearnableAverager) updated
      - from-scratch : base model freshly initialised, same interface
    """

    def __init__(self, base_model: nn.Module, method_config: MethodConfig):
        super().__init__()
        self.model = base_model
        self.cfg = method_config

        # Patch LearnableAverager hidden_dim to match the loaded model
        if method_config.learnable_module is not None:
            hidden_dim = base_model.gpt_neox.embed_in.embedding_dim
            if method_config.learnable_module.hidden_dim != hidden_dim:
                k = method_config.learnable_module.k
                method_config.learnable_module = LearnableAverager(hidden_dim, k)
                # Re-bind avg_fn closure to the new module
                _new_averager = method_config.learnable_module
                _k = k
                def _new_avg_fn(hidden, input_ids, _a=_new_averager, __k=_k):
                    avg, _ = _a(hidden)
                    return avg, _uniform_labels(input_ids, __k)
                method_config.avg_fn = _new_avg_fn

            # Register as a sub-module so optimizer can find its parameters
            self.learnable_averager = method_config.learnable_module

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:      [B, T]
            attention_mask: [B, T]  or None

        Returns:
            loss:   scalar cross-entropy loss
            logits: [B, T'-1, vocab_size]
        """
        # 1. Manual embedding lookup
        hidden = self.model.gpt_neox.embed_in(input_ids)          # [B, T, D]

        # 2. Apply averaging → [B, T', D] and construct label indices [B, T'-1]
        hidden_avg, labels = self.cfg.avg_fn(hidden, input_ids)    # labels: [B, T'-1]

        T_prime = hidden_avg.size(1)

        # 3. Compress attention mask to [B, T'-1] (drop last to align with labels)
        if attention_mask is not None:
            attn = _compress_attention_mask(attention_mask, T_prime - 1, hidden.device)
        else:
            attn = None

        # 4. Run through transformer with inputs_embeds (skips internal embed_in)
        out = self.model.gpt_neox(
            inputs_embeds=hidden_avg[:, :-1],    # [B, T'-1, D]
            attention_mask=attn,
        )

        # 5. LM head
        logits = self.model.embed_out(out.last_hidden_state)       # [B, T'-1, vocab]

        # 6. Align label length with logits length (may differ by ±1 due to truncation)
        lbl_len = min(logits.size(1), labels.size(1))
        logits_aligned = logits[:, :lbl_len]
        labels_aligned = labels[:, :lbl_len]

        # 7. Cross-entropy loss
        loss = F.cross_entropy(
            logits_aligned.reshape(-1, logits_aligned.size(-1)),
            labels_aligned.reshape(-1),
            ignore_index=-100,
        )
        return loss, logits_aligned

    def parameters_to_train(self):
        """Returns all trainable parameters (base model + optional LearnableAverager)."""
        return self.parameters()


def _compress_attention_mask(
    mask: torch.Tensor,
    target_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Reduce attention mask from [B, T] to [B, target_len] by taking every
    stride-th element.  For now we use simple stride-based subsampling; the
    exact mapping is approximate but sufficient for padding detection.
    """
    B, T = mask.shape
    if T <= target_len:
        # Pad with ones (non-padding) if somehow shorter
        pad = torch.ones(B, target_len - T, dtype=mask.dtype, device=device)
        return torch.cat([mask, pad], dim=1)
    stride = T // target_len
    # Take every stride-th element, then truncate/pad to exact target_len
    compressed = mask[:, ::stride]
    if compressed.size(1) > target_len:
        compressed = compressed[:, :target_len]
    elif compressed.size(1) < target_len:
        pad = torch.ones(B, target_len - compressed.size(1), dtype=mask.dtype, device=device)
        compressed = torch.cat([compressed, pad], dim=1)
    return compressed
