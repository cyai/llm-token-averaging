#!/bin/bash
set -e

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# ── 500M Config A runs on 2× H100 SXM ──────────────────────────────
#
# Architecture: d=1280, h=20, l=22, head_dim=64  (~497M tied params)
# LR: 1.2e-4 (scaled from 2e-4 × sqrt(512/1280))
# Grad checkpointing: ON  (saves VRAM, ~10 GB/GPU with batch 32)
#
# k=1:  10B tokens, seq_len 1024, L=1024
#       global batch = 32×2 = 64 seqs = 65,536 tokens/step
#       → 152,587 steps
#
# k=2:  20B raw tokens, seq_len 1024, L=512
#       global batch = 32×2 = 64 seqs = 65,536 raw tokens/step
#       → 305,175 steps
#
# grad_checkpoint=OFF (SDPA/FlashAttn gives O(T) memory, ckpt not needed)
#
# ⚠  k=2 needs 20B tokens — FineWeb sample-10BT only has ~10B.
#    Use a larger FineWeb slice or set --data_dir to a bigger shard.
# ────────────────────────────────────────────────────────────────────

log "Starting 500M k=1 (standard, 10B tokens) on 2× H100 GPUs..."
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  experiments/chinchilla/train.py \
  --model model1_500m \
  --batch_size 16 --seq_len 1024 --log_steps 500 --eval_batches 16 \
  --num_workers 8 --data_dir /data/fineweb --checkpoint_steps 50000 \
  --resume
log "model1_500m done. Sleeping 2 min..."
sleep 120

log "Starting 500M k=2 (2× averaging, 20B tokens) on 2× H100 GPUs..."
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  experiments/chinchilla/train.py \
  --model avg_500m_k2 \
  --batch_size 16 --seq_len 1024 --log_steps 500 --eval_batches 16 \
  --num_workers 8 --data_dir /data/fineweb --checkpoint_steps 50000 \
  --resume
log "avg_500m_k2 done."
