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
from typing import Dict

# ---------------------------------------------------------------------------
# ModelConfig dataclass
# ---------------------------------------------------------------------------

TARGET_TOKENS = 4_000_000_000   # 4B tokens for every model


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
    averaging_k: int = 1

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
        """Rough parameter count (embedding + transformer layers)."""
        vocab = 50_257  # Pythia GPT-NeoX BPE
        return vocab * self.d_model + self.n_layers * 12 * self.d_model**2

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
        context_len=512,          # n
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
        context_len=512,          # compressed length = n
        averaging_k=2,            # effective raw context = 1024 = 2n
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
        context_len=1024,         # true 2n context
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
        context_len=512,          # compressed length = n
        averaging_k=4,            # effective raw context = 1024 = 2n
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
        context_len=2048,         # true 4n context
        averaging_k=1,
        grad_checkpoint=False,
        color="#f78166",
        label="~8M standard (k=1, 4n=2048)",
        lr=4e-4,
        warmup_steps=500,
        target_tokens=400_000_000,
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
