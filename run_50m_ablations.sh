#!/bin/bash
set -e

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# Pooling ablations, all Config A (seq_len 1024, ~50M params, iso token budgets
# with avg_50m_k2 / avg_50m_k4). Runs sequentially, cheapest first.

log "Starting 50m k=2 learnable pooling (2B tokens) on 1 GPU..."
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model avg_50m_k2_learnable \
  --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 
log "avg_50m_k2_learnable done. Sleeping 2 min..."
sleep 120

log "Starting 50m k=2 word-boundary windows (2B tokens) on 1 GPU..."
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model avg_50m_k2_word \
  --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 
log "avg_50m_k2_word done. Sleeping 2 min..."
sleep 120

log "Starting 50m k=2 overlap w=4 s=2 (2B tokens) on 1 GPU..."
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model avg_50m_k2_ov4s2 \
  --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 
log "avg_50m_k2_ov4s2 done. Sleeping 2 min..."
sleep 120

log "Starting 50m k=2 exponential weights (2B tokens) on 1 GPU..."
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model avg_50m_k2_wexp \
  --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 
log "avg_50m_k2_wexp done. Sleeping 2 min..."
sleep 120

log "Starting 50m k=4 learnable pooling (4.07B tokens) on 1 GPU..."
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model avg_50m_k4_learnable \
  --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 
log "avg_50m_k4_learnable done. Sleeping 2 min..."
sleep 120

log "Starting 50m k=4 exponential weights (4.07B tokens) on 1 GPU..."
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model avg_50m_k4_wexp \
  --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 
log "avg_50m_k4_wexp done. All ablations complete."
