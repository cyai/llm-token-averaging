"""Equal-loss compute saving for each (baseline, k=2) pair.

Reads the target loss off the baseline's final logged point, then finds where
the averaged run's loss-versus-FLOPs curve first crosses that loss. The ratio
of the two FLOPs counts is the compute saving quoted in the paper.

FLOPs come from Eq. (flops) in the paper and are recomputed from tokens_seen;
the cumulative_flops column in the CSVs is ignored. Crossing point is found by
linear interpolation in (log FLOPs, loss) between the two logged evaluations
that bracket the target.

Usage:  python experiments/chinchilla/equal_loss_saving.py [--latex]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from model_configs import MODEL_CONFIGS  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
VOCAB_SIZE = 50304

# (scale, baseline run, averaged run). A run is (config name, results root).
# The 50M averaged arm is the matched-protocol rerun, which is the one quoted
# in the main table; results/avg_50m_k2 is the earlier unmatched run.
PAIRS = [
    ("50M", ("model1_50m", RESULTS), ("avg_50m_k2", HERE / "results_matched_mean")),
    ("125M", ("model1_125m", RESULTS), ("avg_125m_k2", RESULTS)),
    ("250M", ("model1_250m", RESULTS), ("avg_250m_k2", RESULTS)),
    ("500M", ("model1_500m", RESULTS), ("avg_500m_k2", RESULTS)),
]


def per_seq_flops(cfg, raw_seq_len: int) -> float:
    """Training FLOPs for one raw sequence, transformer body plus output head."""
    L = raw_seq_len // cfg.averaging_k
    d = cfg.d_model
    body = cfg.n_layers * (23 * d * d + 4 * L * d)
    head = 2 * d * VOCAB_SIZE
    return 3 * L * (body + head)


def load(run: tuple[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    """Return (cumulative FLOPs, eval loss) for the logged evaluation points."""
    name, root = run
    cfg = MODEL_CONFIGS[name]
    df = pd.read_csv(root / name / "loss_log.csv")
    df = df.dropna(subset=["eval_loss"]).drop_duplicates("step", keep="last")
    df = df.sort_values("step")
    flops = (df["tokens_seen"].values / cfg.context_len) * per_seq_flops(
        cfg, cfg.context_len
    )
    return flops, df["eval_loss"].values


def flops_at_loss(flops: np.ndarray, loss: np.ndarray, target: float) -> float | None:
    """FLOPs at which `loss` first reaches `target`, log-linearly interpolated."""
    hits = np.where(loss <= target)[0]
    if len(hits) == 0:
        return None
    i = hits[0]
    if i == 0:
        return float(flops[0])
    l_hi, l_lo = loss[i - 1], loss[i]
    if l_hi == l_lo:
        return float(flops[i])
    frac = (l_hi - target) / (l_hi - l_lo)
    log_c = np.log(flops[i - 1]) + frac * (np.log(flops[i]) - np.log(flops[i - 1]))
    return float(np.exp(log_c))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="emit a LaTeX tabular body")
    args = ap.parse_args()

    rows = []
    for scale, base_run, avg_run in PAIRS:
        cb, lb = load(base_run)
        ca, la = load(avg_run)
        target = lb[-1]
        c_base = cb[-1]
        c_avg = flops_at_loss(ca, la, target)
        rows.append((scale, target, c_base, c_avg, la[-1]))

    if args.latex:
        for scale, target, c_base, c_avg, _ in rows:
            if c_avg is None:
                print(f"{scale} & {target:.3f} & ${c_base:.2e}$ & never & -- \\\\")
            else:
                print(
                    f"{scale} & {target:.3f} & ${c_base:.2e}$ & ${c_avg:.2e}$ & "
                    f"${c_base / c_avg:.2f}\\times$ \\\\"
                )
        return

    print(f"{'scale':>6} {'target L':>9} {'C_base':>11} {'C_avg':>11} {'saving':>8}")
    print("-" * 50)
    for scale, target, c_base, c_avg, avg_final in rows:
        if c_avg is None:
            print(
                f"{scale:>6} {target:>9.4f} {c_base:>11.3e} {'never':>11} "
                f"{'--':>8}   (avg final {avg_final:.4f})"
            )
        else:
            print(
                f"{scale:>6} {target:>9.4f} {c_base:>11.3e} {c_avg:>11.3e} "
                f"{c_base / c_avg:>7.2f}x"
            )


if __name__ == "__main__":
    main()
