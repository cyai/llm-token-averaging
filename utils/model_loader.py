"""
Model loading utilities for Pythia-410M.
"""

import torch
import logging
from transformers import AutoModel, AutoTokenizer
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)


class EmbeddingExtractorModel:
    """Wrapper for model that extracts embeddings from all layers."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_outputs = {}
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to capture layer outputs."""

        # Hook for embedding layer
        def embedding_hook(module, input, output):
            self.layer_outputs["embedding"] = output.detach()

        # Register embedding hook
        embedding_layer = self.model.get_input_embeddings()
        hook = embedding_layer.register_forward_hook(embedding_hook)
        self.hooks.append(hook)

        # Hook for transformer layers
        if hasattr(self.model, "gpt_neox"):
            layers = self.model.gpt_neox.layers
        elif hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "h"):
                layers = self.model.transformer.h
            elif hasattr(self.model.transformer, "layers"):
                layers = self.model.transformer.layers
        else:
            layers = []
            logger.warning("Could not find transformer layers in model")

        for idx, layer in enumerate(layers):

            def make_hook(layer_idx):
                def hook(module, input, output):
                    # output is typically a tuple, get the hidden states
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    self.layer_outputs[f"layer_{layer_idx}"] = hidden_states.detach()

                return hook

            hook = layer.register_forward_hook(make_hook(idx))
            self.hooks.append(hook)

        logger.info(f"Registered {len(self.hooks)} hooks for layer extraction")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_outputs = {}

    def extract(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from all layers.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dictionary mapping layer names to embeddings [batch_size, seq_len, hidden_dim]
        """
        self.layer_outputs = {}

        with torch.no_grad():
            if attention_mask is not None:
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                _ = self.model(input_ids=input_ids)

        return self.layer_outputs.copy()


def load_pythia_model(
    model_name: str, device: str = "cuda"
) -> Tuple[EmbeddingExtractorModel, AutoTokenizer]:
    """
    Load Pythia model and tokenizer with embedding extraction capabilities.

    Args:
        model_name: Name of the model to load (e.g., "EleutherAI/pythia-410m")
        device: Device to load model on ("cuda" or "cpu")

    Returns:
        Tuple of (EmbeddingExtractorModel, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )

    model.eval()
    model.to(device)

    # Wrap model with embedding extractor
    extractor_model = EmbeddingExtractorModel(model, tokenizer)
    extractor_model.register_hooks()

    logger.info(f"Model loaded successfully on {device}")
    logger.info(f"Model has {len(extractor_model.hooks)} layers")

    return extractor_model, tokenizer
