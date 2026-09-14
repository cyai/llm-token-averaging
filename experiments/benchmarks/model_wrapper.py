"""
Custom lm-evaluation-harness model wrapper for OLM token-averaging models.

Supports both k=1 (standard) and k>1 (averaged) models loaded from
Hugging Face repos under the FAIRC org.

For k>1 models, uses offset-ensemble inference: runs k forward passes
(one per offset) and combines log-probabilities so that every token
position gets a prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.chinchilla.model_configs import get_config, ModelConfig
from experiments.shared.olm_model import OLMTransformerBody
from experiments.shared.averaged_lm import build_method_config


TOKENIZER_NAME = "EleutherAI/pythia-70m"


def _choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_backbone_from_checkpoint(
    cfg: ModelConfig, ckpt_path: str, device: str
) -> OLMTransformerBody:
    """Build backbone and load weights from a checkpoint file."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    vocab_size = len(tokenizer)

    backbone = OLMTransformerBody(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        context_length=cfg.context_len,
    )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state

    # Averaged models prefix keys with "backbone."
    if any(key.startswith("backbone.") for key in sd):
        sd = {k[len("backbone."):]: v for k, v in sd.items()
              if k.startswith("backbone.")}

    # Strip "learnable_averager." keys (pooling module weights not needed for inference)
    sd = {k: v for k, v in sd.items() if not k.startswith("learnable_averager.")}

    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys: {missing}")
    if unexpected:
        print(f"  [warn] ignored unexpected keys: {unexpected}", flush=True)

    backbone.to(device).eval()
    return backbone


@register_model("token_averaging")
class TokenAveragingLM(LM):
    """
    lm-evaluation-harness model class for OLM token-averaging checkpoints.

    Instantiate via:
        lm_eval --model token_averaging \
                --model_args model_name=avg_50m_k2,ckpt_path=/path/to/final.pt
    """

    def __init__(
        self,
        model_name: str,
        ckpt_path: str,
        device: Optional[str] = None,
        batch_size: int = 1,
        max_length: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = get_config(model_name)
        self.k = self.cfg.averaging_k
        self._device = device or _choose_device()
        self._batch_size = int(batch_size)

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = load_backbone_from_checkpoint(
            self.cfg, ckpt_path, self._device
        )

        self._max_length = max_length or self.cfg.context_len
        print(
            f"[TokenAveragingLM] model={model_name}, k={self.k}, "
            f"device={self._device}, max_length={self._max_length}",
            flush=True,
        )

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, left_truncate_len: int = None, add_special_tokens: bool = False) -> List[int]:
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _logits_k1(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Standard forward for k=1. Returns logits [B, T, V]."""
        with torch.no_grad():
            return self.backbone(input_ids)

    def _logprobs_averaged(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Offset-ensemble for k>1.

        Runs k offsets and assembles a full [B, T, V] log-probability tensor.
        Pads with k EOS tokens to ensure the last real position is covered.
        """
        B, T = input_ids.shape
        V = self.backbone.vocab_size
        k = self.k

        logprobs = torch.full((B, T, V), float("-inf"), device=input_ids.device)

        # Pad to ensure full coverage of all real positions
        pad = torch.full((B, k), self.eot_token_id, device=input_ids.device, dtype=input_ids.dtype)
        input_padded = torch.cat([input_ids, pad], dim=1)

        for offset in range(k):
            ids_o = input_padded[:, offset:]
            n_windows = ids_o.size(1) // k
            if n_windows < 2:
                continue
            ids_o = ids_o[:, :n_windows * k]

            hidden = self.backbone.embed_in(ids_o)
            _, _, D = hidden.shape
            avg = hidden.reshape(B, n_windows, k, D).mean(dim=2)

            out = self.backbone.body(avg[:, :-1])
            logits = self.backbone.embed_out(out)  # [B, n_windows-1, V]

            lp = F.log_softmax(logits.float(), dim=-1)

            for j in range(lp.size(1)):
                pos = offset + (j + 1) * k
                if pos < T:
                    logprobs[:, pos, :] = lp[:, j, :]

        return logprobs

    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Return log-probabilities for each position.
        Shape: [B, T, V] where position t predicts token at position t+1.
        """
        input_ids = input_ids.to(self._device)
        with torch.no_grad():
            if self.k == 1:
                logits = self._logits_k1(input_ids)
                return F.log_softmax(logits.float(), dim=-1)
            else:
                return self._logprobs_averaged(input_ids)

    def loglikelihood(self, requests: list) -> list:
        """
        Compute log-likelihood of continuation given context.
        Each request is (context, continuation).
        Returns list of (loglikelihood, is_greedy) tuples.
        """
        results = []
        for request in requests:
            ctx, cont = request.args
            ctx_enc = self.tok_encode(ctx)
            cont_enc = self.tok_encode(cont)
            full_enc = ctx_enc + cont_enc

            # Truncate from the left if too long
            if len(full_enc) > self.max_length:
                full_enc = full_enc[-self.max_length:]
                # Recalculate where continuation starts
                cont_len = len(cont_enc)
            else:
                cont_len = len(cont_enc)

            input_ids = torch.tensor([full_enc], dtype=torch.long)
            logprobs = self._model_call(input_ids)  # [1, T, V]

            # Log-probs for continuation tokens
            # Position t's logprob predicts token at t+1, but our logprobs
            # tensor has the prediction for position t at index t
            # For k=1: logprobs[0, t, :] gives distribution over token at position t+1
            # We want the log-prob of each continuation token
            cont_start = len(full_enc) - cont_len
            total_logprob = 0.0
            is_greedy = True

            for i in range(cont_len):
                pred_pos = cont_start + i - 1  # position whose output predicts token at cont_start+i
                if pred_pos < 0:
                    continue
                token_id = full_enc[cont_start + i]
                lp = logprobs[0, pred_pos, :]
                if lp.max() == float("-inf"):
                    # Position not covered (can happen for early positions with k>1)
                    continue
                token_lp = lp[token_id].item()
                total_logprob += token_lp
                if lp.argmax().item() != token_id:
                    is_greedy = False

            results.append((total_logprob, is_greedy))

        return results

    def loglikelihood_rolling(self, requests: list) -> list:
        """Rolling perplexity over a document."""
        results = []
        for request in requests:
            (string,) = request.args
            encoding = self.tok_encode(string)

            total_logprob = 0.0
            n_tokens = 0

            # Process in chunks of max_length
            stride = self.max_length
            for start in range(0, max(1, len(encoding) - 1), stride):
                end = min(start + self.max_length, len(encoding))
                chunk = encoding[start:end]
                if len(chunk) < 2:
                    continue

                input_ids = torch.tensor([chunk], dtype=torch.long)
                logprobs = self._model_call(input_ids)

                # Accumulate log-probs for positions after the first
                # (first position in each chunk except the very first one
                # would have been predicted by the previous chunk)
                eval_start = 0 if start == 0 else 0
                for t in range(eval_start, len(chunk) - 1):
                    lp = logprobs[0, t, :]
                    if lp.max() == float("-inf"):
                        continue
                    token_lp = lp[chunk[t + 1]].item()
                    total_logprob += token_lp
                    n_tokens += 1

            results.append((total_logprob,))

        return results

    def generate_until(self, requests: list) -> list:
        """
        Generate text until stop string. Limited support for small models.
        """
        results = []
        for request in requests:
            ctx = request.args[0]
            gen_kwargs = request.args[1] if len(request.args) > 1 else {}
            until = gen_kwargs.get("until", [self.tokenizer.eos_token])
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            ctx_enc = self.tok_encode(ctx)
            if len(ctx_enc) > self.max_length - max_gen:
                ctx_enc = ctx_enc[-(self.max_length - max_gen):]

            generated = list(ctx_enc)
            for _ in range(max_gen):
                if len(generated) > self.max_length:
                    input_chunk = generated[-self.max_length:]
                else:
                    input_chunk = generated

                input_ids = torch.tensor([input_chunk], dtype=torch.long)
                logprobs = self._model_call(input_ids)

                # Sample from last valid position
                last_lp = logprobs[0, -1, :]
                if last_lp.max() == float("-inf"):
                    # k>1: last position may not be covered; back off
                    for back in range(2, min(self.k + 1, len(input_chunk))):
                        last_lp = logprobs[0, -back, :]
                        if last_lp.max() != float("-inf"):
                            break

                next_token = last_lp.argmax().item()
                generated.append(next_token)

                # Check stop conditions
                gen_text = self.tok_decode(generated[len(ctx_enc):])
                if any(s in gen_text for s in until):
                    break

            gen_text = self.tok_decode(generated[len(ctx_enc):])
            for s in until:
                if s in gen_text:
                    gen_text = gen_text[:gen_text.index(s)]
            results.append(gen_text)

        return results
