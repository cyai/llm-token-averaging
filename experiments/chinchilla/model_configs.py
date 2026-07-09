"""
Model configuration registry for the Chinchilla FLOPs comparison experiment.

Three models are compared:
  model1_50m   —  51M params, d=512/h=8/l=8,   context=1024, no averaging   (baseline)
  model2_200m  — 202M params, d=1024/h=16/l=12, context=2048, no averaging  (larger + 2× context)
  avg_50m_k2   —  51M params, d=512/h=8/l=8,   context=1024, k=2 averaging  (effective 2× context)

Training budget: 4B tokens each.

Parameter estimates
-------------------
  N ≈ vocab_size × d_model  +  n_layers × 12 × d_model²

  model1_50m  : 50257 × 512  +  8 × 12 × 512²   = 25.7M +  25.2M ≈  51M
  model2_200m : 50257 × 1024 + 12 × 12 × 1024²  = 51.5M + 150.9M ≈ 202M
  avg_50m_k2  : same backbone as model1_50m ≈ 51M

FLOPs estimates on 8× A6000 (~155 TFLOPS BF16 × 8, ~50% MFU → 620 TFLOPS eff.)
----------------------------------------------------------------------------------
  model1_50m   : C = 6 × 51M  × 4B  = 1.22 × 10¹⁸ FLOPs  ≈  0.5 h
  avg_50m_k2   : C = 6 × 51M  × 2B  = 0.61 × 10¹⁸ FLOPs  ≈  0.3 h  (k=2 halves transformer tokens)
  model2_200m  : C = 6 × 202M × 4B  = 4.85 × 10¹⁸ FLOPs  ≈  2.2 h
  Total sequential                                          ≈  3.0 h
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# ModelConfig dataclass
# ---------------------------------------------------------------------------

TARGET_TOKENS = 4_000_000_000  # 4B tokens for every model


@dataclass
class ModelConfig:
    """Complete specification for one model in the Chinchilla comparison."""

    # Identifier used in directory names, CSV columns, and plot labels
    name: str

    # OLMTransformerBody architecture hyperparameters.
    # model1_50m / avg_50m_k2 share d=512/h=8/l=8/ctx=1024 (~51M params).
    # model2_200m uses d=1024/h=16/l=12/ctx=2048 (~202M params + 2× context).
    d_model: int
    n_heads: int
    n_layers: int
    context_len: int = 1024

    # averaging_k = 1  → standard LM (OLMTransformerBody used directly)
    # averaging_k = 2  → OLMAveragedLanguageModel(uniform_k2) wrapper
    # For non-uniform schemes set averaging_k to the effective k (for FLOPs/budget
    # calculations) and set method_name to the exact build_method_config key.
    averaging_k: int = 1

    # Optional override for the averaging method name passed to build_method_config.
    # When None (default), train.py uses f"uniform_k{averaging_k}".
    # Example: "mixed_k2k4" for the mixed 2×/4× averaging model.
    method_name: Optional[str] = None

    # Fraction of training steps that use multi-token prediction (0 = disabled).
    # E.g. 0.3 means first 30% of steps predict all k next tokens, then
    # the remaining 70% fall back to standard single-token prediction.
    multi_token_phase_ratio: float = 0.0

    # Tie input embedding and output LM head weights (reduces params by vocab×d_model)
    tie_embeddings: bool = True

    # Enable gradient checkpointing to fit large models in 24 GB VRAM
    grad_checkpoint: bool = False

    # Plot colour (hex) and display label
    color: str = "#58a6ff"
    label: str = ""

    # Training budget
    target_tokens: int = TARGET_TOKENS

    # Learning rate (OLM best practice: scale roughly as 1/sqrt(d_model))
    lr: float = 3e-4

    # Warmup steps
    warmup_steps: int = 2_000

    def __post_init__(self):
        if not self.label:
            self.label = self.name

    @property
    def n_params_approx(self) -> int:
        """Parameter count: tied shares one embedding matrix, untied has two."""
        vocab = 50_257  # Pythia GPT-NeoX BPE
        embed_factor = 1 if self.tie_embeddings else 2
        return (
            embed_factor * vocab * self.d_model + self.n_layers * 12 * self.d_model**2
        )

    @property
    def flops_per_token(self) -> float:
        """
        Approximate FLOPs consumed per *original* token processed.

        Standard model  : 6 × N  (forward + backward rule-of-thumb)
        Averaging k=2   : 6 × N / 2  (transformer sees half the tokens)
        """
        n = self.n_params_approx
        if self.averaging_k == 1:
            return 6.0 * n
        else:
            return 6.0 * n / self.averaging_k

    @property
    def total_flops(self) -> float:
        """Total FLOPs to train on target_tokens."""
        return self.flops_per_token * self.target_tokens


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "model1_8m": ModelConfig(
        name="model1_8m",
        d_model=128,
        n_heads=4,
        n_layers=6,
        context_len=512,  # n
        averaging_k=1,
        grad_checkpoint=False,
        color="#58a6ff",
        label="~8M standard (n=512)",
        lr=4e-4,
        warmup_steps=500,
        target_tokens=300_000_000,
    ),
    "avg_8m_k2": ModelConfig(
        name="avg_8m_k2",
        d_model=128,
        n_heads=4,
        n_layers=6,
        context_len=512,  # compressed length = n
        averaging_k=2,  # effective raw context = 1024 = 2n
        grad_checkpoint=False,
        color="#3fb950",
        label="~8M + 2× averaging",
        lr=4e-4,
        warmup_steps=500,
        target_tokens=600_000_000,
    ),
    "model2_8m_ctx2n": ModelConfig(
        name="model2_8m_ctx2n",
        d_model=128,
        n_heads=4,
        n_layers=6,
        context_len=1024,  # true 2n context
        averaging_k=1,
        grad_checkpoint=False,
        color="#f78166",
        label="~8M standard (2n=1024)",
        lr=4e-4,
        warmup_steps=500,
        target_tokens=300_000_000,
    ),
    "avg_8m_k4": ModelConfig(
        name="avg_8m_k4",
        d_model=128,
        n_heads=4,
        n_layers=6,
        context_len=512,  # compressed length = n
        averaging_k=4,  # effective raw context = 1024 = 2n
        grad_checkpoint=False,
        color="#3fb950",
        label="~8M + 4× averaging (k=4)",
        lr=4e-4,
        warmup_steps=500,
        target_tokens=800_000_000,
    ),
    "model2_8m_ctx4n": ModelConfig(
        name="model2_8m_ctx4n",
        d_model=128,
        n_heads=4,
        n_layers=6,
        context_len=2048,  # true 4n context
        averaging_k=1,
        grad_checkpoint=False,
        color="#f78166",
        label="~8M standard (k=1, 4n=2048)",
        lr=4e-4,
        warmup_steps=500,
        target_tokens=400_000_000,
    ),
    "model1_50m_v2": ModelConfig(
        name="model1_50m_v2",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,  # n
        averaging_k=1,
        grad_checkpoint=False,
        color="#4e9de0",
        label="~50M standard (n=1024) (v2)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=20_000_000_000,  # 20B ceiling; early-stop at target eval loss
    ),
    "avg_50m_k2": ModelConfig(
        name="avg_50m_k2",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,  # compressed length = n
        averaging_k=2,  # effective raw context = 2048 = 2n
        grad_checkpoint=False,
        color="#3fb950",
        label="~50M + 2× averaging",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=2_000_000_000,
    ),
    "model1_50m": ModelConfig(
        name="model1_50m",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,  # n
        averaging_k=1,
        grad_checkpoint=False,
        color="#4e9de0",
        label="~50M standard (n=1024)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=1_000_000_000,
    ),
    # ------------------------------------------------------------------
    # Tied-embedding variants  (embed_in and LM head share weights)
    # Same arch as 50M models but ~51M actual params instead of ~76M.
    # ------------------------------------------------------------------
    "model1_50m_tied": ModelConfig(
        name="model1_50m_tied",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,
        averaging_k=1,
        tie_embeddings=True,
        grad_checkpoint=False,
        color="#7c3aed",  # violet
        label="~51M standard tied (n=1024)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=1_000_000_000,
    ),
    "avg_50m_k4_tied": ModelConfig(
        name="avg_50m_k4_tied",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,
        averaging_k=4,
        tie_embeddings=True,
        grad_checkpoint=False,
        color="#a855f7",  # purple
        label="~51M k=4 tied averaging",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=4_072_000_000,
    ),
    # ------------------------------------------------------------------
    # Iso-FLOPs variants: train k=2 and k=4 to match k=1's total FLOPs.
    # k=1 uses 6.71e16 FLOPs for 1B tokens.
    # k=2 uses fewer FLOPs/step (transformer sees L/2 positions) → needs
    #   1B × (flops_per_step_k1 / flops_per_step_k2) = 1B × 2.287 = 2.287B tokens.
    # k=4 uses even fewer → 1B × 4.923 = 4.923B tokens.
    # ------------------------------------------------------------------
    "avg_50m_k2_isoflop": ModelConfig(
        name="avg_50m_k2_isoflop",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,
        averaging_k=2,
        tie_embeddings=True,
        grad_checkpoint=False,
        color="#3fb950",
        label="~51M k=2 tied (iso-FLOPs)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=2_500_000_000,
    ),
    "avg_50m_k4_isoflop": ModelConfig(
        name="avg_50m_k4_isoflop",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,
        averaging_k=4,
        tie_embeddings=True,
        grad_checkpoint=False,
        color="#f1c40f",
        label="~51M k=4 tied (iso-FLOPs)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=5_500_000_000,
    ),
    "model1_50m_tied_2nctx": ModelConfig(
        name="model1_50m_tied_2ctx",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=2048,
        averaging_k=1,
        tie_embeddings=True,
        grad_checkpoint=False,
        color="#4e9de0",
        label="~51M standard tied (2n=2048)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=1_000_000_000,
    ),
    # ~152M standard baseline  (d=1024, h=16, l=8, ctx=1024)
    # N = 50257×1024 + 8×12×1024² = 51.5M + 100.7M ≈ 152M
    # Chinchilla-optimal: D* = 20N ≈ 3B tokens
    "model1_150m": ModelConfig(
        name="model1_150m",
        d_model=1024,
        n_heads=16,
        n_layers=8,
        context_len=1024,
        averaging_k=1,
        grad_checkpoint=False,
        color="#c084fc",  # light purple
        label="~150M standard (n=1024)",
        lr=1.5e-4,  # ~2e-4 × sqrt(512/1024)
        warmup_steps=2000,
        target_tokens=3_000_000_000,
    ),
    # ------------------------------------------------------------------
    # ~125M models  (d=768, h=12, l=12, head_dim=64)
    # N = 50257×768 + 12×12×768² = 38.6M + 84.9M ≈ 123.5M (tied embeddings)
    #   model1_125m  : k=1 standard,    target 2.5B tokens
    #   avg_125m_k2  : k=2 averaging,   target 5B tokens (transformer sees half)
    # lr ≈ 2e-4 × sqrt(512/768) ≈ 1.6e-4
    # ------------------------------------------------------------------
    # d_model=512,
    #     n_heads=8,
    #     n_layers=8,
    "model1_125m": ModelConfig(
        name="model1_125m",
        d_model=768,
        n_heads=12, 
        n_layers=12,
        context_len=1024,  # n
        averaging_k=1,
        tie_embeddings=True,
        grad_checkpoint=False,
        color="#4e9de0",  # blue
        label="~125M standard (n=1024)",
        lr=1.6e-4,
        warmup_steps=2000,
        target_tokens=2_500_000_000,  # 2.5B
    ),
    "avg_125m_k2": ModelConfig(
        name="avg_125m_k2",
        d_model=768,
        n_heads=12,
        n_layers=12,
        context_len=1024,  # compressed length = n
        averaging_k=2,  # effective raw context = 2048 = 2n
        tie_embeddings=True,
        grad_checkpoint=False,
        color="#3fb950",  # green
        label="~125M + 2× averaging",
        lr=1.6e-4,
        warmup_steps=2000,
        target_tokens=5_000_000_000,  # 5B
    ),
    # ------------------------------------------------------------------
    # ~250M models  (d=1024, h=16, l=16, head_dim=64)
    # N = 50257×1024 + 16×12×1024² = 51.5M + 201.3M ≈ 253M (tied embeddings)
    #   model1_250m  : k=1 standard,    target 5B raw tokens
    #   avg_250m_k2  : k=2 averaging,   target 10B raw tokens (transformer sees 5B)
    # lr ≈ 2e-4 × sqrt(512/1024) ≈ 1.4e-4
    # ------------------------------------------------------------------
    "model1_250m": ModelConfig(
        name="model1_250m",
        d_model=1024,
        n_heads=16,
        n_layers=16,
        context_len=1024,  # n
        averaging_k=1,
        tie_embeddings=True,
        grad_checkpoint=True,
        color="#4e9de0",  # blue
        label="~250M standard (n=1024)",
        lr=1.4e-4,
        warmup_steps=2000,
        target_tokens=5_000_000_000,  # 5B
    ),
    "avg_250m_k2": ModelConfig(
        name="avg_250m_k2",
        d_model=1024,
        n_heads=16,
        n_layers=16,
        context_len=1024,  # compressed length = n
        averaging_k=2,  # effective raw context = 2048 = 2n
        tie_embeddings=True,
        grad_checkpoint=True,
        color="#3fb950",  # green
        label="~250M + 2× averaging",
        lr=1.4e-4,
        warmup_steps=2000,
        target_tokens=10_000_000_000,  # 10B
    ),
    "model2_50m_ctx2n_v2": ModelConfig(
        name="model2_50m_ctx2n_v2",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=2048,  # true 2n context
        averaging_k=1,
        grad_checkpoint=True,  # 2048 ctx may be tight on VRAM
        color="#f0a500",
        label="~50M standard (2n=2048) (v2)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=1_000_000_000,
    ),
    "model2_50m_ctx2n": ModelConfig(
        name="model2_50m_ctx2n",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=2048,  # true 2n context
        averaging_k=1,
        grad_checkpoint=True,  # 2048 ctx may be tight on VRAM
        color="#f0a500",
        label="~50M standard (2n=2048)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=1_000_000_000,
    ),
    "avg_50m_k2_v2": ModelConfig(
        name="avg_50m_k2_v2",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,  # compressed length = n
        averaging_k=2,  # effective raw context = 2048 = 2n
        grad_checkpoint=False,
        color="#3fb950",
        label="~50M + 2× averaging (v2)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=2_000_000_000,
    ),
    "avg_50m_k4": ModelConfig(
        name="avg_50m_k4",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,  # compressed length = n
        averaging_k=4,  # effective raw context = 4096 = 4n
        grad_checkpoint=False,
        color="#f1c40f",  # yellow
        label="~50M + 4× averaging (k=4)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=4_072_000_000,  # = 4 × 20N
    ),
    "avg_50m_k2_ctx512": ModelConfig(
        name="avg_50m_k2_ctx512",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=512,  # compressed length = n/2
        averaging_k=2,  # effective raw context = 1024 = n (same as baseline)
        grad_checkpoint=False,
        color="#e67e22",  # orange
        label="~50M + 2× averaging (ctx=512)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=2_036_000_000,  # = 2 × 20N
    ),
    # Mixed model: first 512 compressed positions use k=2 (1024 raw tokens),
    # last 512 compressed positions use k=4 (2048 raw tokens).
    # Effective context = 3072 original tokens per sequence, k_eff = 3.
    # Train with --seq_len 3072.
    "avg_50m_mixed_k2k4": ModelConfig(
        name="avg_50m_mixed_k2k4",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,  # total compressed positions (512 k=2 + 512 k=4)
        averaging_k=3,  # effective k for FLOPs / budget maths (k_eff = 3072/1024)
        method_name="mixed_k2k4",  # routes to build_method_config("mixed_k2k4")
        grad_checkpoint=False,
        color="#9b59b6",  # purple
        label="~50M mixed k=2/4 averaging",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=3_054_000_000,  # ≈ 3 × 20N  (k_eff = 3)
    ),
    "avg_50m_k8": ModelConfig(
        name="avg_50m_k8",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,  # compressed length = n
        averaging_k=8,  # effective raw context = 8192 = 8n
        grad_checkpoint=False,
        color="#00c8c8",  # cyan
        label="~50M + 8× averaging (k=8)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=8_144_000_000,  # = 8 × 20N
    ),
    "avg_50m_k16": ModelConfig(
        name="avg_50m_k16",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,  # compressed length = n
        averaging_k=16,  # effective raw context = 16384 = 16n
        grad_checkpoint=False,
        color="#00c8c8",  # cyan
        label="~50M + 16× averaging (k=16)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=16_288_000_000,  # = 16 × 20N
    ),
    "avg_50m_k32": ModelConfig(
        name="avg_50m_k32",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,  # compressed length = n
        averaging_k=32,  # effective raw context = 32768 = 32n
        grad_checkpoint=False,
        color="#00c8c8",  # cyan
        label="~50M + 32× averaging (k=32)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=32_576_000_000,  # = 32 × 20N
    ),
    "avg_50m_k64": ModelConfig(
        name="avg_50m_k64",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,  # compressed length = n
        averaging_k=64,  # effective raw context = 65536 = 64n
        grad_checkpoint=False,
        color="#00c8c8",  # cyan
        label="~50M + 64× averaging (k=64)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=65_152_000_000,  # = 64 × 20N
    ),
    "avg_50m_k128": ModelConfig(
        name="avg_50m_k128",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,  # compressed length = n
        averaging_k=128,  # effective raw context = 131072 = 128n
        grad_checkpoint=False,
        color="#00c8c8",  # cyan
        label="~50M + 128× averaging (k=128)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=130_304_000_000,  # = 128 × 20N
    ),
    "avg_50m_k2_wide": ModelConfig(
        name="avg_50m_k2_wide",
        d_model=864,  # solved: matches model2_50m_ctx2n FLOPs (+0.57%)
        n_heads=8,  # head_dim = 107
        n_layers=8,
        context_len=1024,  # compressed length
        averaging_k=2,  # effective raw context = 2048 = 2n
        grad_checkpoint=False,
        color="#e040fb",  # purple
        label="~113M k=2 FLOPs-matched (d=856)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=2_000_000_000,  # same token budget as model2_50m_ctx2n
    ),
    # ------------------------------------------------------------------
    # Phased multi-token prediction experiments
    # Phase 1 (first 30% steps): predict all k next tokens per position
    # Phase 2 (remaining 70%):   standard single-token prediction
    # Token budgets match corresponding non-phased averaged models.
    # ------------------------------------------------------------------
    "avg_50m_k2_phased": ModelConfig(
        name="avg_50m_k2_phased",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,
        averaging_k=2,
        multi_token_phase_ratio=0.5,
        grad_checkpoint=False,
        color="#2ecc71",  # emerald
        label="~50M k=2 phased (30% multi-tok)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=2_000_000_000,
    ),
    "avg_50m_k4_phased": ModelConfig(
        name="avg_50m_k4_phased",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,
        averaging_k=4,
        multi_token_phase_ratio=0.5,
        grad_checkpoint=False,
        color="#e74c3c",  # crimson
        label="~50M k=4 phased (50% multi-tok)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=4_072_000_000,
    ),
    "avg_50m_k4_phased_30": ModelConfig(
        name="avg_50m_k4_phased",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,
        averaging_k=4,
        multi_token_phase_ratio=0.3,
        grad_checkpoint=False,
        color="#e74c3c",  # crimson
        label="~50M k=4 phased (30% multi-tok)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=4_072_000_000,
    ),
    "avg_50m_k8_phased": ModelConfig(
        name="avg_50m_k8_phased",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,
        averaging_k=8,
        multi_token_phase_ratio=0.5,
        grad_checkpoint=False,
        color="#1abc9c",  # teal
        label="~50M k=8 phased (30% multi-tok)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=8_144_000_000,
    ),
    # ==================================================================
    # POOLING ABLATIONS  (all Config A: seq_len 1024, transformer L = 1024/k)
    #
    # Each run changes exactly ONE thing vs avg_50m_k2 (uniform mean pooling)
    # and keeps everything else identical: same architecture, same raw
    # context (1024), same batch/LR/schedule, same 2B-token budget so the
    # endpoint is iso-FLOPs with the clean k=1 baseline (loss_log_1x_ctx).
    #
    # Train exactly like avg_50m_k2, e.g.:
    #   torchrun --standalone --nproc_per_node=8 experiments/chinchilla/train.py \
    #       --model avg_50m_k2_learnable --batch_size 2 --seq_len 1024
    # ==================================================================
    # (1) Learned pooling: a small trainable module decides how to combine
    #     the k embeddings, instead of fixed equal weights (0.5/0.5).
    #     Question: is plain mean already good enough?
    "avg_50m_k2_learnable": ModelConfig(
        name="avg_50m_k2_learnable",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,
        averaging_k=2,
        method_name="learnable_k2",
        grad_checkpoint=False,
        color="#e67e22",  # orange
        label="~50M k=2 learnable pooling",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=2_000_000_000,
    ),
    # (2) Same question at k=4, where compression is harsher and a learned
    #     combiner has more room to help (or fail).
    "avg_50m_k4_learnable": ModelConfig(
        name="avg_50m_k4_learnable",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,
        averaging_k=4,
        method_name="learnable_k4",
        grad_checkpoint=False,
        color="#d35400",  # dark orange
        label="~50M k=4 learnable pooling",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=4_072_000_000,
    ),
    # (3) Fixed but UNEQUAL weights: exponential weighting gives the later
    #     token in each window more weight than the earlier one.
    #     Question: does emphasising the most recent token matter, or is
    #     any reasonable fixed weighting equivalent?  (Robustness check.)
    "avg_50m_k2_wexp": ModelConfig(
        name="avg_50m_k2_wexp",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,
        averaging_k=2,
        method_name="weighted_exponential_k2",
        grad_checkpoint=False,
        color="#8e44ad",  # purple
        label="~50M k=2 exponential weights",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=2_000_000_000,
    ),
    "avg_50m_k4_wexp": ModelConfig(
        name="avg_50m_k4_wexp",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,
        averaging_k=4,
        method_name="weighted_exponential_k4",
        grad_checkpoint=False,
        color="#8e44ad",  # purple
        label="~50M k=4 exponential weights",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=4_072_000_000,
    ),
    # (4) Overlapping windows: each output position averages 4 tokens but
    #     windows slide by 2 → same 2x compression as k=2, but neighbouring
    #     windows share tokens, so no token sits on a hard boundary.
    #     Question: do hard window cuts (token 2 vs token 3) cost us anything?
    "avg_50m_k2_ov4s2": ModelConfig(
        name="avg_50m_k2_ov4s2",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,
        averaging_k=2,  # effective compression = stride 2 → 2x
        method_name="overlap_w4_s2",
        grad_checkpoint=False,
        color="#16a085",  # teal
        label="~50M k=2 overlap (w=4, s=2)",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=2_000_000_000,
    ),
    # (5) Word-boundary windows: windows break at word starts instead of
    #     every fixed 2 tokens, so no window blends two different words
    #     together.  Groups are >=2 tokens (force-close at 4) and end at
    #     the next word start, so mean compression is slightly above 2x.
    #     Question: is word-blending the main source of the averaged model's
    #     per-token handicap?  Highest-upside ablation of the set.
    "avg_50m_k2_word": ModelConfig(
        name="avg_50m_k2_word",
        d_model=512,
        n_heads=8,
        n_layers=8,
        context_len=1024,
        averaging_k=2,  # nominal; iso-FLOPs analysis must use realized ratio
        method_name="word_k2",
        grad_checkpoint=False,
        color="#c0392b",  # red
        label="~50M k=2 word-boundary windows",
        lr=2e-4,
        warmup_steps=2000,
        target_tokens=2_000_000_000,
    ),
}

# Ordered list for sequential training (smallest to largest)
TRAINING_ORDER = ["model1_50m", "avg_50m_k2", "model2_200m"]


def get_config(name: str) -> ModelConfig:
    """Return a ModelConfig by name, raising KeyError with a helpful message."""
    if name not in MODEL_CONFIGS:
        valid = list(MODEL_CONFIGS.keys())
        raise KeyError(f"Unknown model config {name!r}. Valid choices: {valid}")
    return MODEL_CONFIGS[name]


def print_summary() -> None:
    """Print a summary table of all three model configs."""
    header = (
        f"{'Model':<20} {'d_model':>8} {'heads':>6} {'layers':>7} "
        f"{'~Params':>10} {'FLOPs/tok':>12} {'Total FLOPs':>14} {'avg_k':>6}"
    )
    print(header)
    print("-" * len(header))
    for cfg in MODEL_CONFIGS.values():
        n = cfg.n_params_approx
        print(
            f"{cfg.name:<20} {cfg.d_model:>8} {cfg.n_heads:>6} {cfg.n_layers:>7} "
            f"{n/1e6:>9.1f}M {cfg.flops_per_token:>12.3e} "
            f"{cfg.total_flops:>14.3e} {cfg.averaging_k:>6}"
        )


if __name__ == "__main__":
    print_summary()
