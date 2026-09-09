#!/bin/bash
set -e

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# ── 50M runs to close out the 50M section ──────────────────────────
#
# Same protocol for all four: 1 GPU, 16,384 raw tokens/update, 2k warmup,
# eval_batches 16, log every 100 steps.  batch × seq_len = 16,384 throughout,
# so batch 16 at seq 1024 and batch 8 at seq 2048.
#
# more-data pair (seq 1024):
#   k=1   1.00B tokens →  61,035 steps
#   k=2   2.00B tokens → 122,070 steps
#
# more-context pair (seq 2048, both arms see 2048 raw tokens):
#   k=1   real 2048-position attention window, 1.00B tokens →  61,035 steps
#   k=2   2048 raw tokens compressed into 1024 positions, 2.00B → 122,070 steps
#
# Why re-run instead of rescoring the old checkpoints:
#   - model1_50m/final.pt on the Hub is a 0-byte broken LFS pointer
#   - the averaged 2048-context checkpoint is gone (Hub copy is an untied
#     model from a different run), and its k=1 partner used the older
#     protocol, so both arms have to be redone together
#   - train.py now logs eval_loss_all_pos, so these runs produce the
#     cross-k-comparable loss directly in loss_log.csv — no rescoring needed
#
# --seed 1 means results land in <name>_seed1, so nothing existing is
# overwritten.  The two seq-2048 runs use their own --results_dir because
# they share the avg_50m_k2 config name with the seq-1024 run.
# ────────────────────────────────────────────────────────────────────

RESULTS=experiments/chinchilla/results_seed_replicates
RESULTS_CTX=experiments/chinchilla/results_ctx2048

log "Starting 50M k=1 (standard, 1.00B tokens) on 1 GPU..."
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model model1_50m --seed 1 \
  --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 \
  --results_dir $RESULTS \
  --resume
log "model1_50m done. Sleeping 2 min..."
sleep 120

log "Starting 50M k=2 (2x averaging, 2.00B tokens) on 1 GPU..."
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model avg_50m_k2 --seed 1 \
  --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 \
  --results_dir $RESULTS \
  --resume
log "avg_50m_k2 done. Sleeping 2 min..."
sleep 120

log "Starting 50M k=1 full attention @2048 (1.00B tokens) on 1 GPU..."
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model model1_50m_tied_2nctx --seed 1 \
  --batch_size 8 --seq_len 2048 --log_steps 100 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 \
  --results_dir $RESULTS_CTX \
  --resume
log "model1_50m_tied_2nctx done. Sleeping 2 min..."
sleep 120

log "Starting 50M k=2 more-context @2048 (2.00B raw tokens) on 1 GPU..."
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model avg_50m_k2 --seed 1 \
  --batch_size 8 --seq_len 2048 --log_steps 100 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 \
  --results_dir $RESULTS_CTX \
  --resume
log "avg_50m_k2 @2048 done. All four runs complete."
