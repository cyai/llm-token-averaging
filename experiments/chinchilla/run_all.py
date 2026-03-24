"""
Orchestrator for the Chinchilla FLOPs comparison experiment.

Trains all three models sequentially (or a selected subset), generates
intermediate visualizations after each model completes, and supports
resuming interrupted training runs.

Recommended usage on AWS g5.2xlarge (~13 days total sequential runtime):

    # Full sequential run — all 3 models
    nohup python experiments/chinchilla/run_all.py \
        --batch_size 8 --seq_len 1024 --device cuda \
        > chinchilla_train.log 2>&1 &

    # Single model
    python experiments/chinchilla/run_all.py --models model1_125m

    # Resume interrupted run
    python experiments/chinchilla/run_all.py --models model2_500m --resume

    # Visualize what has been collected so far (no training)
    python experiments/chinchilla/run_all.py --visualize_only

Typical batch sizes for a 24 GB A10G:
  model1_125m / avg_125m_k2 : --batch_size 8   (seq_len 1024)
  model2_500m               : --batch_size 4   (seq_len 1024, grad-ckpt enabled in config)

Logs
----
Progress is printed to stdout every `--log_steps` steps as a single line per
step — safe for `nohup` redirection and `tail -f` monitoring.  Each model's
loss log is written to:

    experiments/chinchilla/results/<model_name>/loss_log.csv

Checkpoints are saved every `--checkpoint_steps` steps to:

    experiments/chinchilla/results/<model_name>/checkpoints/step_XXXXXXXX.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.chinchilla.model_configs import TRAINING_ORDER, get_config
from experiments.chinchilla.train import train_model, parse_args as _train_parse_args
from experiments.chinchilla.visualize import generate_plots


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full Chinchilla FLOPs comparison experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection
    p.add_argument(
        "--models",
        nargs="+",
        choices=["model1_125m", "model2_500m", "avg_125m_k2"],
        default=TRAINING_ORDER,
        help="Which models to train (in the given order). "
             "Default: all three in the recommended order.",
    )

    # Training hyperparameters
    p.add_argument("--batch_size",        type=int, default=8)
    p.add_argument("--seq_len",           type=int, default=1024)
    p.add_argument("--device",            type=str, default="cuda")
    p.add_argument("--log_steps",         type=int, default=1_000,
                   help="Frequency (steps) for CSV logging and stdout progress.")
    p.add_argument("--checkpoint_steps",  type=int, default=50_000,
                   help="Frequency (steps) for saving checkpoints.")
    p.add_argument("--eval_batches",      type=int, default=64,
                   help="Number of eval batches per evaluation pass.")
    p.add_argument("--num_workers",       type=int, default=0,
                   help="DataLoader worker processes.")
    p.add_argument("--tokenizer_name",    type=str,
                   default="EleutherAI/pythia-70m",
                   help="HuggingFace tokenizer identifier.")

    # Flow control
    p.add_argument("--resume",           action="store_true",
                   help="Resume each model from its latest checkpoint.")
    p.add_argument("--visualize_only",   action="store_true",
                   help="Skip training; only regenerate plots from existing logs.")
    p.add_argument("--no_intermediate_plots", action="store_true",
                   help="Skip intermediate plot generation after each model.")

    # Paths
    p.add_argument("--results_dir", type=str,
                   default=str(_ROOT / "experiments" / "chinchilla" / "results"),
                   help="Root directory for training results (logs + checkpoints).")
    p.add_argument("--plots_dir",   type=str,
                   default=str(_ROOT / "experiments" / "chinchilla" / "plots"),
                   help="Output directory for generated plots.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir   = Path(args.plots_dir)

    # -----------------------------------------------------------------------
    # Visualize-only mode
    # -----------------------------------------------------------------------
    if args.visualize_only:
        print("=== Visualize-only mode ===", flush=True)
        generate_plots(results_dir, plots_dir)
        return

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    completed: list[str] = []
    failed: list[str] = []

    total_start = time.time()

    for model_name in args.models:
        cfg = get_config(model_name)
        print(
            f"\n{'='*70}\n"
            f"  Training: {model_name}  "
            f"({cfg.n_params_approx/1e6:.0f}M params, "
            f"k={cfg.averaging_k})\n"
            f"{'='*70}",
            flush=True,
        )

        # Build a Namespace that matches train.py's parse_args interface
        train_args = argparse.Namespace(
            model=model_name,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=args.device,
            log_steps=args.log_steps,
            checkpoint_steps=args.checkpoint_steps,
            eval_batches=args.eval_batches,
            num_workers=args.num_workers,
            tokenizer_name=args.tokenizer_name,
            resume=args.resume,
            results_dir=str(results_dir),
        )

        model_start = time.time()
        try:
            model_results_dir = results_dir / model_name
            log_path = train_model(cfg, train_args, results_dir=model_results_dir)
            elapsed_h = (time.time() - model_start) / 3600
            print(
                f"\n[{model_name}] Finished in {elapsed_h:.1f}h. "
                f"Log: {log_path}",
                flush=True,
            )
            completed.append(model_name)
        except Exception as exc:
            elapsed_h = (time.time() - model_start) / 3600
            print(
                f"\n[{model_name}] FAILED after {elapsed_h:.1f}h: {exc}",
                flush=True,
            )
            failed.append(model_name)
            # Continue to the next model rather than aborting the whole run
            continue

        # --- intermediate visualization after each model ---
        if not args.no_intermediate_plots:
            print(f"\n[run_all] Generating intermediate plots …", flush=True)
            try:
                generate_plots(results_dir, plots_dir)
            except Exception as exc:
                print(f"[run_all] Plot generation failed: {exc}", flush=True)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
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

    # Final visualization over all completed models
    print("\n[run_all] Generating final plots …", flush=True)
    try:
        static_path, interactive_path = generate_plots(results_dir, plots_dir)
        if static_path:
            print(f"  Static plot     : {static_path}", flush=True)
        if interactive_path:
            print(f"  Interactive plot: {interactive_path}", flush=True)
    except Exception as exc:
        print(f"[run_all] Final plot generation failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
