#!/bin/bash
set -e

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# ── 1B Config A runs on 2× H100 SXM ────────────────────────────────────
#
# Architecture: d=1664, h=26, l=28, head_dim=64  (975,945,984 tied params)
#   Continues the ladder invariants: head_dim=64 everywhere, d/n_layers≈59
#   (500M sat at 58.2). d=1664 is divisible by 128 for tensor-core alignment.
# LR: 1.1e-4  (2e-4 × sqrt(512/1664))
#
# Global batch 16×2 = 32 seqs = 32,768 raw tokens/step, IDENTICAL to the
# 500M protocol. Keep it that way: changing tokens/step at 1B would add a
# fresh confound to the very scaling comparison these runs exist to make.
#
# k=1:  20B raw tokens, L=1024  →   610,351 steps  (D/N = 20.5)
# k=2:  40B raw tokens, L= 512  → 1,220,703 steps  (D/N = 41.0)
#
# grad_checkpoint=True in the config: required to hold batch 16/GPU at 1B on
# 80 GB. Costs ~33% compute. On 4+ GPUs use --batch_size 8 and flip
# grad_checkpoint=False in model_configs.py for the same global batch at
# better MFU.
#
# Checkpoints are ~12 GB each at 1B. --keep_last_checkpoints 3 caps the
# checkpoint dir at ~36 GB/run; saves are atomic, so a full disk can no
# longer leave a corrupted .pt that breaks --resume.
#
# ⚠  DATA: the k=2 arm needs 40B raw tokens. Pre-tokenize before launching:
#      python experiments/chinchilla/fineweb_loader.py \
#        --data_dir /data/fineweb --max_train_tokens 40000000000 --num_proc 16
#    That selects FineWeb sample-100BT (~56 shards) and needs ~80 GB for
#    train.bin plus room for the raw parquets.
#
# ⏱  Estimated ~6.6 days (k=1) + ~13.1 days (k=2) on 2× H100 at the 500M's
#    measured throughput scaled for size and checkpointing. Run under tmux.
# ────────────────────────────────────────────────────────────────────────

log "Starting 1B k=1 (standard, 20B tokens) on 2× H100 GPUs..."
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  experiments/chinchilla/train.py \
  --model model1_1b \
  --batch_size 16 --seq_len 1024 --log_steps 500 --eval_batches 16 \
  --num_workers 8 --data_dir /data/fineweb \
  --checkpoint_steps 50000 --keep_last_checkpoints 3 \
  --resume
log "model1_1b done. Sleeping 2 min..."
sleep 120

log "Starting 1B k=2 (2× averaging, 40B tokens) on 2× H100 GPUs..."
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  experiments/chinchilla/train.py \
  --model avg_1b_k2 \
  --batch_size 16 --seq_len 1024 --log_steps 500 --eval_batches 16 \
  --num_workers 8 --data_dir /data/fineweb \
  --checkpoint_steps 50000 --keep_last_checkpoints 3 \
  --resume
log "avg_1b_k2 done."
