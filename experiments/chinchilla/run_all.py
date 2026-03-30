"""
Orchestrator for the Chinchilla FLOPs comparison experiment.

Trains all three models sequentially (or a selected subset), generates
intermediate visualizations after each model completes, and supports
resuming interrupted training runs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  8× A6000 recommended launch commands
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 0 — pre-download tokenizer ONCE before torchrun (prevents 8-process
          lock-file race on first launch):

    python -c "from transformers import AutoTokenizer; \
               AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')"

Step 1 — fix NumPy ABI mismatch (NumPy 2.x installed, PyTorch 1.12 needs 1.x):

    pip install "numpy<2"

Step 2 — full sequential run (all 3 models, ~19 h total):

    nohup torchrun --standalone --nproc_per_node=8 \
        experiments/chinchilla/run_all.py \
        --batch_size 16 --seq_len 1024 \
        > chinchilla_train.log 2>&1 &

    # tail -f chinchilla_train.log     ← monitor progress

Single model:

    torchrun --standalone --nproc_per_node=8 \
        experiments/chinchilla/run_all.py \
        --models model2_500m --batch_size 8

Resume interrupted run:

    torchrun --standalone --nproc_per_node=8 \
        experiments/chinchilla/run_all.py \
        --models model2_500m --batch_size 8 --resume

Visualize existing results without training:

    python experiments/chinchilla/run_all.py --visualize_only

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Recommended per-GPU batch sizes for A6000 (48 GB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  model1_125m  / avg_125m_k2   --batch_size 16  (seq_len 1024)
  model2_500m                  --batch_size 8   (seq_len 1024)

  Global effective batch = batch_size × 8 GPUs
    125M  →  128 sequences / step  = 131,072 tokens / step
    500M  →   64 sequences / step  =  65,536 tokens / step

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Estimated training times on 8× A6000
  (~155 TFLOPS BF16 × 8 = 1240 TFLOPS peak;
   ~50 % MFU ≈ 620 TFLOPS effective)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  model1_125m  7.41×10¹⁸ FLOPs → ≈  3.3 h
  avg_125m_k2  3.71×10¹⁸ FLOPs → ≈  1.7 h
  model2_500m  3.03×10¹⁹ FLOPs → ≈ 13.6 h
  ─────────────────────────────────────────
  Total sequential              ≈ 18.6 h

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Logs & checkpoints
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  experiments/chinchilla/results/<model>/loss_log.csv
  experiments/chinchilla/results/<model>/checkpoints/step_XXXXXXXX.pt
  experiments/chinchilla/plots/loss_vs_flops.png
  experiments/chinchilla/plots/loss_vs_flops_interactive.html
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ── Set HF cache dir BEFORE any HuggingFace import ───────────────────────────
# datasets/transformers read HF_HOME & friends at *import time*, so the env
# vars must be present before "from transformers import ..." is executed.
# We do a quick scan of sys.argv here; proper argparse runs later.
def _early_set_cache_dir() -> None:
    for i, arg in enumerate(sys.argv):
        if arg == "--cache_dir" and i + 1 < len(sys.argv):
            cp = os.path.abspath(sys.argv[i + 1])
            os.environ.setdefault("HF_HOME",            cp)
            os.environ.setdefault("HF_DATASETS_CACHE",  os.path.join(cp, "datasets"))
            os.environ.setdefault("TRANSFORMERS_CACHE",  os.path.join(cp, "hub"))
            break

_early_set_cache_dir()
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.chinchilla.model_configs import TRAINING_ORDER, get_config
from experiments.chinchilla.train import train_model
from experiments.chinchilla.visualize import generate_plots


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full Chinchilla FLOPs comparison experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    p.add_argument(
        "--models",
        nargs="+",
        choices=["model1_125m", "model2_500m", "avg_125m_k2"],
        default=TRAINING_ORDER,
        help="Models to train, in the given order. Default: all three.",
    )

    # Training hyperparameters (passed through to train.py)
    p.add_argument("--batch_size",        type=int, default=16,
                   help="Per-GPU batch size. Recommended: 16 for 125M, 8 for 500M.")
    p.add_argument("--seq_len",           type=int, default=1024)
    p.add_argument("--device",            type=str, default="cuda",
                   help="Device for single-GPU mode (ignored by torchrun).")
    p.add_argument("--log_steps",         type=int, default=1_000,
                   help="Log to CSV + print every N optimizer steps.")
    p.add_argument("--checkpoint_steps",  type=int, default=50_000,
                   help="Save checkpoint every N steps.")
    p.add_argument("--eval_batches",      type=int, default=64)
    p.add_argument("--num_workers",       type=int, default=2,
                   help="DataLoader worker processes per rank.")
    p.add_argument("--tokenizer_name",    type=str,
                   default="EleutherAI/pythia-70m")

    # Flow control
    p.add_argument("--resume",            action="store_true",
                   help="Resume each model from its latest checkpoint.")
    p.add_argument("--visualize_only",    action="store_true",
                   help="Skip training; only regenerate plots from existing logs.")
    p.add_argument("--no_intermediate_plots", action="store_true",
                   help="Skip intermediate plot generation after each model.")

    # Paths
    p.add_argument("--results_dir", type=str,
                   default=str(_ROOT / "experiments" / "chinchilla" / "results"),
                   help="Root directory for training results.")
    p.add_argument("--plots_dir",   type=str,
                   default=str(_ROOT / "experiments" / "chinchilla" / "plots"),
                   help="Output directory for generated plots.")
    p.add_argument("--cache_dir",   type=str, default=None,
                   help="Local directory for HuggingFace model/dataset cache "
                        "(overrides ~/.cache/huggingface). "
                        "Example: --cache_dir ./hf_cache")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args        = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir   = Path(args.plots_dir)

    # Only rank 0 should drive the outer orchestration when run under torchrun.
    # The inner train_model() function handles per-rank DDP setup itself.
    local_rank  = int(os.environ.get("LOCAL_RANK",  0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))
    is_main     = local_rank == 0

    # -----------------------------------------------------------------------
    # Visualize-only mode
    # -----------------------------------------------------------------------
    if args.visualize_only:
        if is_main:
            print("=== Visualize-only mode ===", flush=True)
            generate_plots(results_dir, plots_dir)
        return

    # -----------------------------------------------------------------------
    # Training loop (each model trained to completion before the next starts)
    # -----------------------------------------------------------------------
    completed: list[str] = []
    failed:    list[str] = []
    total_start = time.time()

    for model_name in args.models:
        cfg = get_config(model_name)

        if is_main:
            print(
                f"\n{'='*70}\n"
                f"  Training: {model_name}  "
                f"({cfg.n_params_approx/1e6:.0f}M params, k={cfg.averaging_k})\n"
                f"  Global batch: {args.batch_size} × {world_size} GPU(s) "
                f"= {args.batch_size * world_size} sequences / step\n"
                f"{'='*70}",
                flush=True,
            )

        train_args = argparse.Namespace(
            model            = model_name,
            batch_size       = args.batch_size,
            seq_len          = args.seq_len,
            device           = args.device,
            log_steps        = args.log_steps,
            checkpoint_steps = args.checkpoint_steps,
            eval_batches     = args.eval_batches,
            num_workers      = args.num_workers,
            tokenizer_name   = args.tokenizer_name,
            resume           = args.resume,
            results_dir      = str(results_dir),
            cache_dir        = args.cache_dir,
        )

        model_start = time.time()
        try:
            model_results_dir = results_dir / model_name
            log_path = train_model(cfg, train_args, results_dir=model_results_dir)
            elapsed_h = (time.time() - model_start) / 3600
            if is_main:
                print(
                    f"\n[{model_name}] Finished in {elapsed_h:.1f}h. "
                    f"Log: {log_path}",
                    flush=True,
                )
            completed.append(model_name)
        except Exception as exc:
            elapsed_h = (time.time() - model_start) / 3600
            if is_main:
                print(
                    f"\n[{model_name}] FAILED after {elapsed_h:.1f}h: {exc}",
                    flush=True,
                )
            failed.append(model_name)
            continue   # keep going — train remaining models

        # --- intermediate visualization (rank 0, after each model) ----------
        if is_main and not args.no_intermediate_plots:
            print(f"\n[run_all] Generating intermediate plots …", flush=True)
            try:
                generate_plots(results_dir, plots_dir)
            except Exception as exc:
                print(f"[run_all] Plot generation failed: {exc}", flush=True)

    # -----------------------------------------------------------------------
    # Final summary + plots
    # -----------------------------------------------------------------------
    if is_main:
        total_h = (time.time() - total_start) / 3600
        print(
            f"\n{'='*70}\n"
            f"  Chinchilla FLOPs Comparison — run complete\n"
            f"  Total wall time : {total_h:.1f}h\n"
            f"  Completed       : {completed}\n"
            f"  Failed          : {failed}\n"
            f"{'='*70}",
            flush=True,
        )

        print("\n[run_all] Generating final plots …", flush=True)
        try:
            static_path, interactive_path = generate_plots(results_dir, plots_dir)
            if static_path:
                print(f"  Static plot      → {static_path}", flush=True)
            if interactive_path:
                print(f"  Interactive plot → {interactive_path}", flush=True)
        except Exception as exc:
            print(f"[run_all] Final plot generation failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
