#!/bin/bash
set -e

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

log "Starting 250m k=1 (standard, 5B tokens) on 2 GPUs..."
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  experiments/chinchilla/train.py \
  --model model1_250m \
  --batch_size 16 --seq_len 1024 --log_steps 1000 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 
log "model1_250m done. Sleeping 2 min..."
sleep 120

log "Starting 250m k=2 (2x averaging, 10B tokens) on 2 GPUs..."
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  experiments/chinchilla/train.py \
  --model avg_250m_k2 \
  --batch_size 16 --seq_len 1024 --log_steps 1000 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 
log "avg_250m_k2 done."