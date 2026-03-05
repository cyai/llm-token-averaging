"""
OLM-based model architecture for token-averaging from-scratch experiments.

Uses OLM (OpenLanguageModel) Block + LM building blocks to construct the
model architecture, then exposes the embedding and transformer components
separately so that token averaging can be inserted between them.

OLM's LM block is a sequential Block whose direct children are (in order):
  [0]   embedding block  — maps token IDs → dense hidden states
  [1:-1] transformer body — one or more Repeat/Block combinators
  [-1]  output head       — maps hidden states → vocabulary logits

OLMAveragedLanguageModel wraps OLMTransformerBody with the same
forward interface as AveragedLanguageModel so that all shared evaluation
utilities (compute_perplexity, save_results, etc.) work unchanged.

Architecture defaults (≈ Pythia-70m scale):
  vocab_size    : from tokenizer
  d_model       : 512
  n_heads       : 8
  n_layers      : 6
  context_length: 512
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.averaging_methods import LearnableAverager
from experiments.shared.averaged_lm import MethodConfig, _uniform_labels


# ---------------------------------------------------------------------------
# OLMTransformerBody
# ---------------------------------------------------------------------------

class OLMTransformerBody(nn.Module):
    """
    Wraps OLM's LM block and splits it into three named sub-modules:

      embed_in   — the OLM embedding block (token IDs → hidden states)
      body       — the OLM transformer stack (hidden → hidden)
      embed_out  — the OLM output head      (hidden → logits)

    OLM's Block stores its components as sequential children.  We extract
    them positionally: first child = embedding, last child = head,
    everything in between = transformer body.

    If OLM wraps its components in a single ModuleList child (rather than
    registering them as individual direct children), we unwrap one level
    automatically.

    Args:
        vocab_size     : vocabulary size (must match the tokeniser)
        d_model        : hidden dimension
        n_heads        : number of attention heads
        n_layers       : number of transformer layers
        context_length : maximum sequence length (used for positional encoding)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        context_length: int,
    ) -> None:
        super().__init__()

        from olm.nn.blocks import LM as OLMLM  # imported here to keep OLM optional

        olm_lm = OLMLM(vocab_size, d_model, n_heads, n_layers, context_length)

        children = list(olm_lm.children())

        # If OLM stores everything inside a single ModuleList, unwrap one level
        if len(children) == 1 and isinstance(children[0], nn.ModuleList):
            children = list(children[0])

        if len(children) < 3:
            raise RuntimeError(
                f"OLM LM yielded {len(children)} direct child module(s) after "
                f"unwrapping. Expected at least 3 (embedding, transformer body, "
                f"output head).  Inspect list(OLMLM(...).children()) to debug "
                f"and adjust the splitting logic in OLMTransformerBody."
            )

        # First child: embedding block
        self.embed_in: nn.Module = children[0]

        # Middle children: transformer body (may be a single Repeat or several Blocks)
        if len(children) == 3:
            self.body: nn.Module = children[1]
        else:
            self.body = nn.Sequential(*children[1:-1])

        # Last child: output head (logit projection)
        self.embed_out: nn.Module = children[-1]

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.context_length = context_length

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Standard LM forward without averaging. Returns logits [B, T, vocab]."""
        hidden = self.embed_in(input_ids)
        hidden = self.body(hidden)
        return self.embed_out(hidden)


# ---------------------------------------------------------------------------
# OLMAveragedLanguageModel
# ---------------------------------------------------------------------------

class OLMAveragedLanguageModel(nn.Module):
    """
    OLM-based counterpart to AveragedLanguageModel.

    Inserts a token-averaging step between OLM's embedding block and its
    transformer body, so the model trains end-to-end on compressed token
    representations from the very first step.

    Forward interface (identical to AveragedLanguageModel):
        loss, logits = model(input_ids, attention_mask=None)

    Parameters
    ----------
    backbone      : OLMTransformerBody instance
    method_config : MethodConfig from experiments.shared.averaged_lm
    """

    def __init__(
        self,
        backbone: OLMTransformerBody,
        method_config: MethodConfig,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.cfg = method_config

        # Patch LearnableAverager hidden_dim to match the OLM model's d_model
        if method_config.learnable_module is not None:
            hidden_dim = backbone.d_model
            if method_config.learnable_module.hidden_dim != hidden_dim:
                k = method_config.learnable_module.k
                method_config.learnable_module = LearnableAverager(hidden_dim, k)
                # Re-bind the avg_fn closure to the new, correctly-sized module
                _new_averager = method_config.learnable_module
                _k = k

                def _new_avg_fn(
                    hidden: torch.Tensor,
                    input_ids: torch.Tensor,
                    _a: LearnableAverager = _new_averager,
                    __k: int = _k,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    avg, _ = _a(hidden)
                    return avg, _uniform_labels(input_ids, __k)

                method_config.avg_fn = _new_avg_fn

            # Register as a sub-module so the optimiser picks up its parameters
            self.learnable_averager = method_config.learnable_module

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # kept for API compatibility
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        input_ids      : [B, T] — integer token IDs
        attention_mask : [B, T] — optional; accepted for API compatibility with
                         the shared eval utilities but not forwarded to OLM's
                         transformer (OLM's Block takes a single tensor input).

        Returns
        -------
        loss   : scalar cross-entropy
        logits : [B, T'-1, vocab_size]
        """
        # 1. OLM embedding lookup → [B, T, D]
        hidden = self.backbone.embed_in(input_ids)

        # 2. Apply token averaging → [B, T', D] + labels [B, T'-1]
        hidden_avg, labels = self.cfg.avg_fn(hidden, input_ids)

        # 3. OLM transformer body (single-tensor forward) → [B, T'-1, D]
        #    We drop the last compressed token (no label for it) before
        #    the transformer, matching the AveragedLanguageModel convention.
        hidden_out = self.backbone.body(hidden_avg[:, :-1])

        # 4. OLM output head → [B, T'-1, vocab]
        logits = self.backbone.embed_out(hidden_out)

        # 5. Align label length with logit length (may differ by ±1)
        lbl_len = min(logits.size(1), labels.size(1))
        logits_aligned = logits[:, :lbl_len]
        labels_aligned = labels[:, :lbl_len]

        # 6. Cross-entropy loss
        loss = F.cross_entropy(
            logits_aligned.reshape(-1, logits_aligned.size(-1)),
            labels_aligned.reshape(-1),
            ignore_index=-100,
        )
        return loss, logits_aligned

    def parameters_to_train(self):
        """All trainable parameters — backbone + optional LearnableAverager."""
        return self.parameters()
