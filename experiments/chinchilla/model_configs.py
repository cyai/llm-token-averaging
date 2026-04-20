"""
Model configuration registry for the Chinchilla FLOPs comparison experiment.

Three models are compared:
  model1_125m  — 124M params, context=1024, no averaging       (baseline)
  model2_500m  — 504M params, context=2048, no averaging       (bigger model + 2× context)
  avg_125m_k2  — 124M params, context=1024, k=2 averaging     (effective 2× context via compression)

The key research question: can a small model with token averaging match the
quality of a 4× larger model trained with a genuine 2× context window, at a
fraction of the FLOPs?

Parameter estimates
-------------------
  N ≈ vocab_size × d_model  +  n_layers × 12 × d_model²

  model1_125m : 50257 × 768  + 12 × 12 × 768²   = 38.6M +  85.0M ≈ 124M
  model2_500m : 50257 × 1024 + 36 × 12 × 1024²  = 51.5M + 453.0M ≈ 504M
  avg_125m_k2 : same backbone as model1_125m ≈ 124M

FLOPs estimates on 8× A6000 (~155 TFLOPS BF16 × 8, ~50% MFU → 620 TFLOPS eff.)
----------------------------------------------------------------------------------
  model1_125m  : C = 6 × 124M × 10B  = 7.44 × 10¹⁸ FLOPs  ≈   3.3 h
  avg_125m_k2  : C = 6 × 124M × 5B   = 3.72 × 10¹⁸ FLOPs  ≈   1.7 h  (k=2 halves transformer tokens)
  model2_500m  : C = 6 × 504M × 10B  = 3.02 × 10¹⁹ FLOPs  ≈  13.5 h
  Total sequential                                           ≈  18.5 h
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

# ---------------------------------------------------------------------------
# ModelConfig dataclass
# ---------------------------------------------------------------------------

TARGET_TOKENS = 10_000_000_000  # 10B tokens for every model


@dataclass
class ModelConfig:
    """Complete specification for one model in the Chinchilla comparison."""

    # Identifier used in directory names, CSV columns, and plot labels
    name: str

    # OLMTransformerBody architecture hyperparameters.
    # model1_125m / avg_125m_k2 share d=768/h=12/l=12/ctx=1024.
    # model2_500m uses d=1024/h=16/l=36/ctx=2048 for ~504M params + 2× context.
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
    "model1_125m": ModelConfig(
        name="model1_125m",
        d_model=768,
        n_heads=12,
        n_layers=12,
        context_len=1024,
        averaging_k=1,
        grad_checkpoint=False,
        color="#58a6ff",  # blue
        label="125M standard",
        lr=3e-4,
        warmup_steps=2_000,
    ),
    "model2_500m": ModelConfig(
        name="model2_500m",
        d_model=1024,
        n_heads=16,
        n_layers=36,
        context_len=2048,       # 2× context vs model1_125m
        averaging_k=1,
        grad_checkpoint=True,   # 504M params + seq=2048 requires checkpointing on A6000
        color="#f78166",        # coral
        label="500M 2× context",
        lr=1e-4,                # lower lr for larger model
        warmup_steps=2_000,
    ),
    "avg_125m_k2": ModelConfig(
        name="avg_125m_k2",
        d_model=768,
        n_heads=12,
        n_layers=12,
        context_len=1024,
        averaging_k=2,
        grad_checkpoint=False,
        color="#3fb950",  # green
        label="125M + 2× averaging",
        lr=3e-4,
        warmup_steps=2_000,
    ),
}

# Ordered list for sequential training (smallest to largest to warm up the run)
TRAINING_ORDER = ["model1_125m", "avg_125m_k2", "model2_500m"]


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
