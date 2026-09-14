#!/usr/bin/env python3
"""
Run downstream benchmarks on token-averaging models.

Downloads checkpoints from Hugging Face (FAIRC org) and evaluates them
using lm-evaluation-harness on standard NLU benchmarks.

Designed for MacBook (MPS or CPU). Processes one model at a time.

Usage:
    # Single model
    python experiments/benchmarks/run_benchmarks.py --model model1_50m

    # List all available models
    python experiments/benchmarks/run_benchmarks.py --list

    # Custom benchmarks
    python experiments/benchmarks/run_benchmarks.py --model avg_50m_k2 \
        --tasks lambada_openai,hellaswag,piqa

    # All models sequentially
    python experiments/benchmarks/run_benchmarks.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# HF repo naming convention from upload_to_hf.py
REPO_PREFIX = "token-averaging"
NAMESPACE = "FAIRC"

# Models to evaluate (grouped by experiment type)
MODELS_MAIN = {
    # 50M scale - main comparison
    "model1_50m": {"tag": "", "k": 1, "group": "50M main"},
    "avg_50m_k2": {"tag": "", "k": 2, "group": "50M main"},
    "avg_50m_k4": {"tag": "", "k": 4, "group": "50M main"},
    # 250M scale
    "model1_250m": {"tag": "", "k": 1, "group": "250M"},
    "avg_250m_k2": {"tag": "", "k": 2, "group": "250M"},
    # 500M scale
    "model1_500m": {"tag": "", "k": 1, "group": "500M"},
    "avg_500m_k2": {"tag": "", "k": 2, "group": "500M"},
}

# Ablation models (50M k=2 pooling variants)
MODELS_ABLATION_K2 = {
    "avg_50m_k2_learnable": {"tag": "", "k": 2, "group": "k=2 ablation"},
    "avg_50m_k2_wexp": {"tag": "", "k": 2, "group": "k=2 ablation"},
    "avg_50m_k2_learnable_pos": {"tag": "", "k": 2, "group": "k=2 ablation"},
}

# Ablation models (50M k=4 pooling variants)
MODELS_ABLATION_K4 = {
    "avg_50m_k4_learnable": {"tag": "", "k": 4, "group": "k=4 ablation"},
    "avg_50m_k4_wexp": {"tag": "", "k": 4, "group": "k=4 ablation"},
    "avg_50m_k4_learnable_pos": {"tag": "", "k": 4, "group": "k=4 ablation"},
}

# Ablation models (50M k=8 pooling variants)
MODELS_ABLATION_K8 = {
    "avg_50m_k8_learnable": {"tag": "", "k": 8, "group": "k=8 ablation"},
    "avg_50m_k8_wexp": {"tag": "", "k": 8, "group": "k=8 ablation"},
    "avg_50m_k8_learnable_pos": {"tag": "", "k": 8, "group": "k=8 ablation"},
}

# Matched-mean controls
MODELS_MATCHED_MEAN = {
    "avg_50m_k2": {"tag": "matched_mean", "k": 2, "group": "matched-mean"},
    "avg_50m_k4": {"tag": "matched_mean", "k": 4, "group": "matched-mean"},
    "avg_50m_k8": {"tag": "matched_mean", "k": 8, "group": "matched-mean"},
}

ALL_MODELS = {}
ALL_MODELS.update(MODELS_MAIN)
ALL_MODELS.update(MODELS_ABLATION_K2)
ALL_MODELS.update(MODELS_ABLATION_K4)
ALL_MODELS.update(MODELS_ABLATION_K8)
ALL_MODELS.update(MODELS_MATCHED_MEAN)

# Default benchmarks suitable for small LMs (50M-500M params)
DEFAULT_TASKS = [
    "lambada_openai",   # last-word prediction (tests language modeling quality)
    "hellaswag",        # commonsense NLI (4-way multiple choice)
    "piqa",             # physical intuition QA (2-way)
    "arc_easy",         # science QA (4-way)
    "winogrande",       # coreference resolution
]


def repo_id(tag: str, run_name: str) -> str:
    """Construct HF repo ID from tag and run name."""
    parts = [REPO_PREFIX]
    if tag:
        parts.append(tag)
    parts.append(run_name)
    return f"{NAMESPACE}/{'-'.join(parts)}"


def download_checkpoint(model_name: str, tag: str, cache_dir: Path) -> Path:
    """Download final.pt from HF repo and return local path."""
    from huggingface_hub import hf_hub_download

    rid = repo_id(tag, model_name)
    local_path = cache_dir / f"{tag}_{model_name}" if tag else cache_dir / model_name
    local_path.mkdir(parents=True, exist_ok=True)

    ckpt_file = local_path / "final.pt"
    if ckpt_file.exists() and ckpt_file.stat().st_size > 1000:
        print(f"  [cache] {ckpt_file} already downloaded", flush=True)
        return ckpt_file

    print(f"  Downloading from {rid} ...", flush=True)
    try:
        downloaded = hf_hub_download(repo_id=rid, filename="checkpoints/final.pt")
        result = Path(downloaded)
        # Verify it's not a dangling LFS pointer
        if result.stat().st_size < 1000:
            raise RuntimeError(
                f"Downloaded file is only {result.stat().st_size} bytes — "
                f"likely a dangling LFS pointer in {rid}.\n"
                f"  Fix: re-upload from training machine with:\n"
                f"    python experiments/chinchilla/upload_to_hf.py "
                f"--public --recreate --only {model_name}"
            )
        return result
    except Exception as e:
        print(f"  [ERROR] Failed to download {rid}: {e}", flush=True)
        raise


def run_eval_for_model(
    model_name: str,
    tag: str,
    ckpt_path: Path,
    tasks: list[str],
    output_dir: Path,
    batch_size: int,
    device: str,
) -> dict:
    """Run lm-eval benchmarks for a single model and return results."""
    import lm_eval

    # Import our custom model class (registers it with lm_eval)
    sys.path.insert(0, str(Path(__file__).parent))
    import model_wrapper  # noqa: F401 - registers @register_model

    suffix = f"{tag}_{model_name}" if tag else model_name
    results_path = output_dir / f"{suffix}.json"

    if results_path.exists():
        print(f"  [skip] Results already exist: {results_path}", flush=True)
        with open(results_path) as f:
            return json.load(f)

    print(f"\n{'='*60}", flush=True)
    print(f"  Evaluating: {model_name} (tag={tag or 'canonical'})", flush=True)
    print(f"  Checkpoint: {ckpt_path}", flush=True)
    print(f"  Tasks: {', '.join(tasks)}", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()

    model_args = f"model_name={model_name},ckpt_path={ckpt_path},device={device},batch_size={batch_size}"

    results = lm_eval.simple_evaluate(
        model="token_averaging",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        device=device,
    )

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f} min", flush=True)

    # Add metadata
    results["model_name"] = model_name
    results["tag"] = tag
    results["k"] = ALL_MODELS.get(model_name, {}).get("k", 1)
    results["elapsed_seconds"] = elapsed

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved to {results_path}", flush=True)

    return results


def print_summary(all_results: list[dict], tasks: list[str]) -> None:
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Header
    task_cols = [t.replace("_openai", "").replace("_", " ")[:12] for t in tasks]
    header = f"{'Model':<35} {'k':>3} " + " ".join(f"{t:>12}" for t in task_cols)
    print(header)
    print("-" * len(header))

    for res in all_results:
        name = res.get("model_name", "?")
        tag = res.get("tag", "")
        k = res.get("k", 1)
        display = f"{tag+'/' if tag else ''}{name}"

        scores = []
        task_results = res.get("results", {})
        for task in tasks:
            task_res = task_results.get(task, {})
            # lm-eval stores accuracy under various keys
            acc = task_res.get("acc,none", task_res.get("acc_norm,none",
                   task_res.get("acc", task_res.get("acc_norm", None))))
            if acc is not None:
                scores.append(f"{acc*100:>11.2f}%")
            else:
                # Try perplexity for lambada
                ppl = task_res.get("perplexity,none", task_res.get("perplexity", None))
                if ppl is not None:
                    scores.append(f"{ppl:>11.2f}p")
                else:
                    scores.append(f"{'N/A':>12}")

        print(f"{display:<35} {k:>3} " + " ".join(scores))

    print("=" * 80)


def list_available_models():
    """Print all models that can be evaluated."""
    print("\nAvailable models for benchmarking:")
    print("-" * 70)
    print(f"{'Name':<35} {'k':>3} {'Tag':<15} {'Group':<20}")
    print("-" * 70)

    seen = set()
    for name, info in ALL_MODELS.items():
        key = (name, info["tag"])
        if key in seen:
            continue
        seen.add(key)
        print(f"{name:<35} {info['k']:>3} {info['tag'] or 'canonical':<15} {info['group']:<20}")
    print(f"\nTotal: {len(seen)} model(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Run downstream benchmarks on token-averaging models"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model config name to evaluate (e.g. 'model1_50m', 'avg_50m_k2')"
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Results tag (e.g. 'matched_mean'). Empty = canonical results."
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Local checkpoint path (skips HF download)"
    )
    parser.add_argument(
        "--tasks", type=str, default=",".join(DEFAULT_TASKS),
        help=f"Comma-separated task list. Default: {','.join(DEFAULT_TASKS)}"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(_ROOT / "experiments" / "benchmarks" / "results"),
        help="Directory to save results JSON files"
    )
    parser.add_argument(
        "--cache_dir", type=str,
        default=str(_ROOT / "experiments" / "benchmarks" / ".cache"),
        help="Directory to cache downloaded checkpoints"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for evaluation (reduce if OOM)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'mps', 'cpu', 'cuda'. Auto-detected if not set."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate ALL registered models sequentially"
    )
    parser.add_argument(
        "--group", type=str, default=None,
        help="Evaluate only models in this group (e.g. '50M main', 'k=2 ablation')"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available models and exit"
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip models whose checkpoints are not already cached"
    )

    args = parser.parse_args()

    if args.list:
        list_available_models()
        return

    # Determine device
    if args.device:
        device = args.device
    else:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"Using device: {device}", flush=True)

    tasks = [t.strip() for t in args.tasks.split(",")]
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to evaluate
    models_to_run = []

    if args.model:
        tag = args.tag
        if args.model in ALL_MODELS and not tag:
            tag = ALL_MODELS[args.model].get("tag", "")
        models_to_run.append((args.model, tag))
    elif args.all or args.group:
        seen = set()
        for name, info in ALL_MODELS.items():
            if args.group and info["group"] != args.group:
                continue
            key = (name, info["tag"])
            if key not in seen:
                seen.add(key)
                models_to_run.append((name, info["tag"]))
    else:
        parser.error("Specify --model, --all, or --group")

    print(f"\nModels to evaluate: {len(models_to_run)}")
    for name, tag in models_to_run:
        print(f"  - {tag+'/' if tag else ''}{name}")

    # Run evaluations
    all_results = []
    for model_name, tag in models_to_run:
        try:
            if args.ckpt and len(models_to_run) == 1:
                ckpt_path = Path(args.ckpt)
            else:
                ckpt_path = download_checkpoint(model_name, tag, cache_dir)

            results = run_eval_for_model(
                model_name=model_name,
                tag=tag,
                ckpt_path=ckpt_path,
                tasks=tasks,
                output_dir=output_dir,
                batch_size=args.batch_size,
                device=device,
            )
            all_results.append(results)

        except Exception as e:
            print(f"\n  [ERROR] {model_name}: {e}", flush=True)
            if args.skip_download and "Failed to download" in str(e):
                continue
            elif not args.all:
                raise

    if all_results:
        print_summary(all_results, tasks)

        # Save combined summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nFull results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
