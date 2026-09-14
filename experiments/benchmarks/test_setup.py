#!/usr/bin/env python3
"""
Quick smoke test: verify the model wrapper loads correctly and can
produce logits for a short input. Run this before the full benchmark.

Usage:
    python experiments/benchmarks/test_setup.py --model model1_50m --ckpt /path/to/final.pt
    python experiments/benchmarks/test_setup.py --model model1_50m  # downloads from HF
"""

import argparse
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Smoke test for benchmark model wrapper")
    parser.add_argument("--model", type=str, required=True, help="Model config name")
    parser.add_argument("--ckpt", type=str, default=None, help="Local checkpoint path")
    parser.add_argument("--tag", type=str, default="", help="Results tag for HF download")
    args = parser.parse_args()

    # Determine device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # Download checkpoint if needed
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        print("Downloading checkpoint from HF...")
        sys.path.insert(0, str(Path(__file__).parent))
        from run_benchmarks import download_checkpoint, NAMESPACE, repo_id
        cache_dir = _ROOT / "experiments" / "benchmarks" / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = download_checkpoint(args.model, args.tag, cache_dir)

    print(f"Checkpoint: {ckpt_path}")

    # Load model
    from experiments.chinchilla.model_configs import get_config
    from model_wrapper import load_backbone_from_checkpoint

    cfg = get_config(args.model)
    print(f"Config: d={cfg.d_model}, h={cfg.n_heads}, l={cfg.n_layers}, k={cfg.averaging_k}")

    t0 = time.time()
    backbone = load_backbone_from_checkpoint(cfg, str(ckpt_path), device)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"Parameters: {n_params/1e6:.1f}M")

    # Test forward pass
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", use_fast=True)
    test_text = "The quick brown fox jumps over the lazy dog. In machine learning, we"
    tokens = tokenizer.encode(test_text)
    input_ids = torch.tensor([tokens], device=device)
    print(f"\nTest input: {len(tokens)} tokens")
    print(f'  "{test_text[:60]}..."')

    with torch.no_grad():
        t0 = time.time()
        if cfg.averaging_k == 1:
            logits = backbone(input_ids)
            print(f"  k=1 forward: logits shape = {logits.shape}")
        else:
            k = cfg.averaging_k
            hidden = backbone.embed_in(input_ids)
            B, T, D = hidden.shape
            n = T // k
            hidden_trunc = hidden[:, :n*k, :]
            avg = hidden_trunc.reshape(B, n, k, D).mean(dim=2)
            out = backbone.body(avg[:, :-1])
            logits = backbone.embed_out(out)
            print(f"  k={k} forward: input {T} tokens -> {n} windows -> logits {logits.shape}")

        elapsed = time.time() - t0
        print(f"  Forward pass: {elapsed*1000:.1f}ms")

    # Check next-token prediction
    if cfg.averaging_k == 1:
        probs = torch.softmax(logits[0, -1].float(), dim=-1)
    else:
        probs = torch.softmax(logits[0, -1].float(), dim=-1)

    top5 = probs.topk(5)
    print(f"\nTop-5 predictions for next token:")
    for i in range(5):
        tok_id = top5.indices[i].item()
        prob = top5.values[i].item()
        tok_str = tokenizer.decode([tok_id])
        print(f"  {i+1}. '{tok_str}' (p={prob:.4f})")

    print("\n[OK] Smoke test passed!")


if __name__ == "__main__":
    main()
