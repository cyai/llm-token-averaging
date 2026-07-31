#!/bin/bash
set -e

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# ── 1B Config A runs on 8× A6000 (48 GB) ───────────────────────────────
#
# Architecture: d=1664, h=26, l=28, head_dim=64  (975,945,984 tied params)
#   Continues the ladder invariants: head_dim=64 everywhere, d/n_layers≈59
#   (500M sat at 58.2). d=1664 is divisible by 128 for tensor-core alignment.
# LR: 1.1e-4  (2e-4 × sqrt(512/1664))
#
# Global batch 4×8 = 32 seqs = 32,768 raw tokens/step, IDENTICAL to the
# 500M protocol (16×2 on H100s). Keep it that way: changing tokens/step at
# 1B would add a fresh confound to the scaling comparison these runs exist
# to make.
#
# k=1:  20B raw tokens, L=1024  →   610,351 steps  (D/N = 20.5)
# k=2:  40B raw tokens, L= 512  → 1,220,703 steps  (D/N = 41.0)
#
# grad_checkpoint=True in the config: required at batch 4/GPU on 48 GB
# A6000s for a ~1B model. Costs ~33% compute. Do not flip it off here
# without a smoke test — A6000 VRAM is tighter than the H100s these configs
# were first sized for.
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
# ⏱  Wall-time: 8× A6000 has less peak BF16 FLOPs than 2× H100 SXM and
#    much less memory bandwidth per GPU, so expect slower than the old
#    H100 estimate. Rough ballpark ~10–14 days (k=1) + ~20–28 days (k=2)
#    if tok/s lands near a healthy MFU; check the first log line and scale.
#    Run under tmux.
#
# num_workers=4 per rank (32 total): enough for pre-tokenized train.bin
# without spawning 64 DataLoader workers across 8 processes.
# ────────────────────────────────────────────────────────────────────────

export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}

log "Starting 1B k=1 (standard, 20B tokens) on 8× A6000 GPUs..."
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/chinchilla/train.py \
  --model model1_1b \
  --batch_size 4 --seq_len 1024 --log_steps 500 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb \
  --checkpoint_steps 50000 --keep_last_checkpoints 3 \
  --resume
log "model1_1b done. Sleeping 2 min..."
sleep 120

log "Starting 1B k=2 (2× averaging, 40B tokens) on 8× A6000 GPUs..."
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/chinchilla/train.py \
  --model avg_1b_k2 \
  --batch_size 4 --seq_len 1024 --log_steps 500 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb \
  --checkpoint_steps 50000 --keep_last_checkpoints 3 \
  --resume
log "avg_1b_k2 done."
