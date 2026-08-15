#!/usr/bin/env python3
"""
Standalone downstream benchmark evaluation for token-averaging models.

This script evaluates models on standard benchmarks WITHOUT requiring
lm-evaluation-harness's model registration system. It implements the
scoring logic directly, making it more portable and debuggable.

Supported benchmarks:
  - LAMBADA (last-word prediction accuracy + perplexity)
  - HellaSwag (4-way commonsense NLI)
  - PIQA (2-way physical intuition)
  - ARC-Easy (4-way science QA)
  - WinoGrande (coreference resolution)

Usage:
    # Single model from HF
    python experiments/benchmarks/eval_downstream.py \
        --model model1_50m --tasks lambada,hellaswag

    # From local checkpoint
    python experiments/benchmarks/eval_downstream.py \
        --model avg_50m_k2 --ckpt /path/to/final.pt

    # All 50M models
    python experiments/benchmarks/eval_downstream.py \
        --models model1_50m avg_50m_k2 avg_50m_k4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.chinchilla.model_configs import get_config, ModelConfig
from experiments.shared.olm_model import OLMTransformerBody

TOKENIZER_NAME = "EleutherAI/pythia-70m"
NAMESPACE = "FAIRC"
REPO_PREFIX = "token-averaging"


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_backbone(cfg: ModelConfig, ckpt_path: str, device: str) -> OLMTransformerBody:
    """Build backbone and load weights."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    backbone = OLMTransformerBody(
        vocab_size=len(tokenizer),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        context_length=cfg.context_len,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state

    if any(key.startswith("backbone.") for key in sd):
        sd = {k[len("backbone."):]: v for k, v in sd.items()
              if k.startswith("backbone.")}
    sd = {k: v for k, v in sd.items() if not k.startswith("learnable_averager.")}

    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys: {missing}")
    if unexpected:
        print(f"  [warn] ignored {len(unexpected)} unexpected keys", flush=True)

    backbone.to(device).eval()

    # Tie embeddings if the config says so (for accurate param count display)
    if cfg.tie_embeddings:
        backbone.tie_embedding_weights()

    return backbone


def download_checkpoint(model_name: str, tag: str, cache_dir: Path) -> Path:
    """Download final.pt from HF."""
    from huggingface_hub import hf_hub_download

    parts = [REPO_PREFIX]
    if tag:
        parts.append(tag)
    parts.append(model_name)
    rid = f"{NAMESPACE}/{'-'.join(parts)}"

    local = cache_dir / (f"{tag}_{model_name}" if tag else model_name)
    local.mkdir(parents=True, exist_ok=True)

    ckpt_file = local / "final.pt"
    if ckpt_file.exists() and ckpt_file.stat().st_size > 1000:
        return ckpt_file

    print(f"  Downloading from {rid}...", flush=True)
    path = hf_hub_download(repo_id=rid, filename="checkpoints/final.pt")
    result = Path(path)

    # Verify it's not just an LFS pointer
    if result.stat().st_size < 1000:
        raise RuntimeError(
            f"Downloaded file is only {result.stat().st_size} bytes — "
            f"likely a dangling LFS pointer in {rid}.\n"
            f"  Fix: re-upload from training machine with:\n"
            f"    python experiments/chinchilla/upload_to_hf.py "
            f"--public --recreate --only {model_name}"
        )
    return result


# ===========================================================================
# Log-probability computation
# ===========================================================================

@torch.no_grad()
def get_logprobs_k1(backbone: OLMTransformerBody, input_ids: torch.Tensor) -> torch.Tensor:
    """
    k=1 model: returns log-probs [B, T-1, V] where position t predicts token t+1.
    """
    logits = backbone(input_ids)  # [B, T, V]
    return F.log_softmax(logits[:, :-1].float(), dim=-1)


@torch.no_grad()
def get_logprobs_averaged(
    backbone: OLMTransformerBody, input_ids: torch.Tensor, k: int
) -> torch.Tensor:
    """
    k>1 model: offset-ensemble. Returns log-probs [B, T-1, V].
    Position t contains the log-prob distribution over token at position t+1.
    Positions not covered by any offset have -inf.

    To ensure the last token IS covered, we pad the input with k-1 EOS tokens.
    The padding only appears in the last (dropped) window so it never enters
    the transformer body and doesn't affect predictions for real tokens.
    """
    B, T = input_ids.shape
    V = backbone.vocab_size
    logprobs = torch.full((B, T - 1, V), float("-inf"),
                          device=input_ids.device, dtype=torch.float32)

    # Pad with EOS to ensure every real position is covered by some offset
    eos_id = 0  # GPT-NeoX EOS token ID
    pad = torch.full((B, k), eos_id, device=input_ids.device, dtype=input_ids.dtype)
    input_padded = torch.cat([input_ids, pad], dim=1)

    for offset in range(k):
        ids_o = input_padded[:, offset:]
        n_windows = ids_o.size(1) // k
        if n_windows < 2:
            continue
        ids_o = ids_o[:, :n_windows * k]

        hidden = backbone.embed_in(ids_o)
        D = hidden.size(-1)
        avg = hidden.reshape(B, n_windows, k, D).mean(dim=2)
        out = backbone.body(avg[:, :-1])
        logits = backbone.embed_out(out)  # [B, n_windows-1, V]
        lp = F.log_softmax(logits.float(), dim=-1)

        for j in range(lp.size(1)):
            target_pos = offset + (j + 1) * k
            idx = target_pos - 1
            if 0 <= idx < T - 1:
                logprobs[:, idx, :] = lp[:, j, :]

    return logprobs


def get_logprobs(backbone: OLMTransformerBody, input_ids: torch.Tensor, k: int):
    """Dispatch to k=1 or k>1 log-prob computation."""
    if k == 1:
        return get_logprobs_k1(backbone, input_ids)
    else:
        return get_logprobs_averaged(backbone, input_ids, k)


def score_choices(
    backbone: OLMTransformerBody, k: int, context_ids: list[int],
    choices_ids: list[list[int]], max_len: int, device: str,
) -> list[float]:
    """
    Score multiple continuations given a shared context.
    Returns list of average log-probs (one per choice).
    """
    scores = []
    for choice_ids in choices_ids:
        full = context_ids + choice_ids
        if len(full) > max_len:
            full = full[-max_len:]
            choice_len = len(choice_ids)
        else:
            choice_len = len(choice_ids)

        input_ids = torch.tensor([full], device=device, dtype=torch.long)
        logprobs = get_logprobs(backbone, input_ids, k)  # [1, T-1, V]

        cont_start = len(full) - choice_len
        total_lp = 0.0
        n_valid = 0
        for i in range(choice_len):
            pos_idx = cont_start + i - 1  # logprobs index
            if pos_idx < 0 or pos_idx >= logprobs.size(1):
                continue
            lp = logprobs[0, pos_idx, :]
            if lp.max() == float("-inf"):
                continue
            token_id = full[cont_start + i]
            total_lp += lp[token_id].item()
            n_valid += 1

        avg_lp = total_lp / max(n_valid, 1)
        scores.append(avg_lp)
    return scores


# ===========================================================================
# Benchmark implementations
# ===========================================================================

def eval_lambada(backbone, k, tokenizer, device, max_len, limit=None):
    """LAMBADA: predict the last word of a passage."""
    print("  Loading LAMBADA...", flush=True)
    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total_nll = 0.0
    total_tokens = 0
    total = len(ds)

    for i, example in enumerate(ds):
        text = example["text"]
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue

        # The target is the last token
        context_tokens = tokens[:-1]
        target_token = tokens[-1]

        if len(tokens) > max_len:
            tokens = tokens[-max_len:]
            context_tokens = tokens[:-1]
            target_token = tokens[-1]

        input_ids = torch.tensor([tokens], device=device, dtype=torch.long)
        logprobs = get_logprobs(backbone, input_ids, k)

        # Last position prediction
        last_idx = logprobs.size(1) - 1
        lp = logprobs[0, last_idx, :]

        if lp.max() != float("-inf"):
            pred = lp.argmax().item()
            if pred == target_token:
                correct += 1
            total_nll += -lp[target_token].item()
            total_tokens += 1

        if (i + 1) % 500 == 0:
            acc_so_far = correct / max(total_tokens, 1) * 100
            print(f"    [{i+1}/{total}] acc={acc_so_far:.1f}%", flush=True)

    acc = correct / max(total_tokens, 1)
    ppl = np.exp(total_nll / max(total_tokens, 1))
    return {"acc": acc, "ppl": ppl, "n": total_tokens}


def eval_hellaswag(backbone, k, tokenizer, device, max_len, limit=None):
    """HellaSwag: 4-way commonsense completion."""
    print("  Loading HellaSwag...", flush=True)
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    for i, example in enumerate(ds):
        ctx = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        ctx_ids = tokenizer.encode(ctx)
        choices_ids = [tokenizer.encode(e) for e in endings]

        scores = score_choices(backbone, k, ctx_ids, choices_ids, max_len, device)
        pred = np.argmax(scores)
        if pred == label:
            correct += 1
        total += 1

        if (i + 1) % 500 == 0:
            print(f"    [{i+1}/{len(ds)}] acc={correct/total*100:.1f}%", flush=True)

    return {"acc": correct / max(total, 1), "n": total}


def _load_piqa_validation():
    """Load PIQA validation set from the original source (bypasses broken HF script)."""
    import urllib.request, tempfile, os
    base = "https://yonatanbisk.com/piqa/data"
    cache = Path(tempfile.gettempdir()) / "piqa_cache"
    cache.mkdir(exist_ok=True)
    rows = []
    for fname in ("valid.jsonl", "valid-labels.lst"):
        dst = cache / fname
        if not dst.exists():
            urllib.request.urlretrieve(f"{base}/{fname}", dst)
    with open(cache / "valid.jsonl") as f:
        import json as _json
        items = [_json.loads(l) for l in f]
    with open(cache / "valid-labels.lst") as f:
        labels = [int(l.strip()) for l in f]
    for item, label in zip(items, labels):
        rows.append({"goal": item["goal"], "sol1": item["sol1"], "sol2": item["sol2"], "label": label})
    return Dataset.from_list(rows)


def eval_piqa(backbone, k, tokenizer, device, max_len, limit=None):
    """PIQA: 2-way physical intuition QA."""
    print("  Loading PIQA...", flush=True)
    ds = _load_piqa_validation()
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    for i, example in enumerate(ds):
        goal = example["goal"]
        choices = [example["sol1"], example["sol2"]]
        label = int(example["label"])

        ctx_ids = tokenizer.encode(goal)
        choices_ids = [tokenizer.encode(c) for c in choices]

        scores = score_choices(backbone, k, ctx_ids, choices_ids, max_len, device)
        pred = np.argmax(scores)
        if pred == label:
            correct += 1
        total += 1

        if (i + 1) % 500 == 0:
            print(f"    [{i+1}/{len(ds)}] acc={correct/total*100:.1f}%", flush=True)

    return {"acc": correct / max(total, 1), "n": total}


def eval_arc_easy(backbone, k, tokenizer, device, max_len, limit=None):
    """ARC-Easy: 4-way science QA."""
    print("  Loading ARC-Easy...", flush=True)
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    for i, example in enumerate(ds):
        question = example["question"]
        choices_text = example["choices"]["text"]
        label_key = example["answerKey"]
        # Map label (A/B/C/D or 1/2/3/4) to index
        labels_list = example["choices"]["label"]
        try:
            label = labels_list.index(label_key)
        except ValueError:
            continue

        ctx_ids = tokenizer.encode(f"Question: {question}\nAnswer:")
        choices_ids = [tokenizer.encode(f" {c}") for c in choices_text]

        scores = score_choices(backbone, k, ctx_ids, choices_ids, max_len, device)
        pred = np.argmax(scores)
        if pred == label:
            correct += 1
        total += 1

        if (i + 1) % 500 == 0:
            print(f"    [{i+1}/{len(ds)}] acc={correct/total*100:.1f}%", flush=True)

    return {"acc": correct / max(total, 1), "n": total}


def eval_winogrande(backbone, k, tokenizer, device, max_len, limit=None):
    """WinoGrande: coreference resolution (2-way)."""
    print("  Loading WinoGrande...", flush=True)
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    for i, example in enumerate(ds):
        sentence = example["sentence"]
        option1 = example["option1"]
        option2 = example["option2"]
        label = int(example["answer"]) - 1  # 1-indexed -> 0-indexed

        # Fill in the blank
        sent1 = sentence.replace("_", option1)
        sent2 = sentence.replace("_", option2)

        ids1 = tokenizer.encode(sent1)
        ids2 = tokenizer.encode(sent2)

        # Score full sentences as rolling log-prob
        scores = []
        for ids in [ids1, ids2]:
            if len(ids) > max_len:
                ids = ids[-max_len:]
            input_ids = torch.tensor([ids], device=device, dtype=torch.long)
            logprobs = get_logprobs(backbone, input_ids, k)
            total_lp = 0.0
            n_valid = 0
            for t in range(logprobs.size(1)):
                lp = logprobs[0, t, :]
                if lp.max() == float("-inf"):
                    continue
                total_lp += lp[ids[t + 1]].item()
                n_valid += 1
            scores.append(total_lp / max(n_valid, 1))

        pred = np.argmax(scores)
        if pred == label:
            correct += 1
        total += 1

        if (i + 1) % 500 == 0:
            print(f"    [{i+1}/{len(ds)}] acc={correct/total*100:.1f}%", flush=True)

    return {"acc": correct / max(total, 1), "n": total}


TASK_FNS = {
    "lambada": eval_lambada,
    "hellaswag": eval_hellaswag,
    "piqa": eval_piqa,
    "arc_easy": eval_arc_easy,
    "winogrande": eval_winogrande,
}


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Standalone benchmark evaluation")
    parser.add_argument("--model", type=str, default=None, help="Single model name")
    parser.add_argument("--models", nargs="+", default=None, help="Multiple model names")
    parser.add_argument("--tag", type=str, default="", help="HF tag (e.g. 'matched_mean')")
    parser.add_argument("--ckpt", type=str, default=None, help="Local checkpoint path")
    parser.add_argument(
        "--tasks", type=str, default="lambada,hellaswag,piqa,arc_easy,winogrande",
        help="Comma-separated tasks"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1, help="(reserved for future)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max examples per task (for quick testing)")
    parser.add_argument("--output_dir", type=str,
                        default=str(_ROOT / "experiments" / "benchmarks" / "results"))
    parser.add_argument("--cache_dir", type=str,
                        default=str(_ROOT / "experiments" / "benchmarks" / ".cache"))
    args = parser.parse_args()

    device = args.device or choose_device()
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tasks = [t.strip() for t in args.tasks.split(",")]
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_names = args.models or ([args.model] if args.model else None)
    if not model_names:
        parser.error("Specify --model or --models")

    all_results = []

    for model_name in model_names:
        cfg = get_config(model_name)
        k = cfg.averaging_k
        max_len = cfg.context_len

        print(f"\n{'='*60}", flush=True)
        print(f"  Model: {model_name} (k={k}, {cfg.d_model}d, {cfg.n_layers}L)", flush=True)
        print(f"{'='*60}", flush=True)

        # Load checkpoint
        if args.ckpt and len(model_names) == 1:
            ckpt_path = Path(args.ckpt)
        else:
            ckpt_path = download_checkpoint(model_name, args.tag, cache_dir)

        backbone = load_backbone(cfg, str(ckpt_path), device)
        n_params = sum(p.numel() for p in backbone.parameters())
        print(f"  Params: {n_params/1e6:.1f}M", flush=True)

        # Run benchmarks
        model_results = {"model": model_name, "k": k, "tag": args.tag,
                         "params_M": n_params / 1e6, "tasks": {}}

        for task in tasks:
            if task not in TASK_FNS:
                print(f"  [skip] Unknown task: {task}", flush=True)
                continue

            print(f"\n  --- {task} ---", flush=True)
            t0 = time.time()
            result = TASK_FNS[task](backbone, k, tokenizer, device, max_len, args.limit)
            elapsed = time.time() - t0

            result["elapsed_s"] = elapsed
            model_results["tasks"][task] = result
            acc = result.get("acc", 0)
            extra = f", ppl={result['ppl']:.1f}" if "ppl" in result else ""
            print(f"  {task}: acc={acc*100:.2f}%{extra} ({elapsed:.0f}s)", flush=True)

        all_results.append(model_results)

        # Save individual result
        suffix = f"{args.tag}_{model_name}" if args.tag else model_name
        out_path = output_dir / f"{suffix}.json"
        with open(out_path, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"\n  Saved: {out_path}", flush=True)

        # Free memory
        del backbone
        if device != "cpu":
            torch.cuda.empty_cache() if device == "cuda" else None

    # Print summary table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    task_short = [t[:12] for t in tasks]
    header = f"{'Model':<30} {'k':>2} " + " ".join(f"{t:>12}" for t in task_short)
    print(header)
    print("-" * len(header))
    for res in all_results:
        name = res["model"]
        k = res["k"]
        scores = []
        for task in tasks:
            tr = res["tasks"].get(task, {})
            acc = tr.get("acc")
            if acc is not None:
                scores.append(f"{acc*100:>11.2f}%")
            else:
                scores.append(f"{'N/A':>12}")
        print(f"{name:<30} {k:>2} " + " ".join(scores))
    print(f"{'='*80}")

    # Save combined
    combined_path = output_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results: {combined_path}")


if __name__ == "__main__":
    main()
