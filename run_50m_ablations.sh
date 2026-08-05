#!/bin/bash
set -e

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# Original pooling ablations, retained for reproducibility.

# log "Starting 50m k=2 learnable pooling (2B tokens) on 1 GPU..."
# python -m torch.distributed.run --standalone --nproc_per_node=1 \
#   experiments/chinchilla/train.py \
#   --model avg_50m_k2_learnable \
#   --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
#   --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000
# log "avg_50m_k2_learnable done. Sleeping 2 min..."
# sleep 120

# log "Starting 50m k=2 word-boundary windows (2B tokens) on 1 GPU..."
# python -m torch.distributed.run --standalone --nproc_per_node=1 \
#   experiments/chinchilla/train.py \
#   --model avg_50m_k2_word \
#   --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
#   --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000
# log "avg_50m_k2_word done. Sleeping 2 min..."
# sleep 120

# log "Starting 50m k=2 overlap w=4 s=2 (2B tokens) on 1 GPU..."
# python -m torch.distributed.run --standalone --nproc_per_node=1 \
#   experiments/chinchilla/train.py \
#   --model avg_50m_k2_ov4s2 \
#   --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
#   --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000
# log "avg_50m_k2_ov4s2 done. Sleeping 2 min..."
# sleep 120

# log "Starting 50m k=2 exponential weights (2B tokens) on 1 GPU..."
# python -m torch.distributed.run --standalone --nproc_per_node=1 \
#   experiments/chinchilla/train.py \
#   --model avg_50m_k2_wexp \
#   --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
#   --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000
# log "avg_50m_k2_wexp done. Sleeping 2 min..."
# sleep 120

# log "Starting 50m k=4 learnable pooling (4.07B tokens) on 1 GPU..."
# python -m torch.distributed.run --standalone --nproc_per_node=1 \
#   experiments/chinchilla/train.py \
#   --model avg_50m_k4_learnable \
#   --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
#   --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000
# log "avg_50m_k4_learnable done. Sleeping 2 min..."
# sleep 120

# log "Starting 50m k=4 exponential weights (4.07B tokens) on 1 GPU..."
# python -m torch.distributed.run --standalone --nproc_per_node=1 \
#   experiments/chinchilla/train.py \
#   --model avg_50m_k4_wexp \
#   --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
#   --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000
# log "avg_50m_k4_wexp done. Sleeping 2 min..."
# sleep 120

# Matched follow-up runs.
# All four runs below use the exact same runtime protocol as those experiments:
# one GPU, batch 16, raw seq_len 1024, 2k-step warmup, and 16 eval batches.

# log "Starting original avg_50m_k2 with matched ablation settings (2B tokens) on 1 GPU..."
# python -m torch.distributed.run --standalone --nproc_per_node=1 \
#   experiments/chinchilla/train.py \
#   --model avg_50m_k2 \
#   --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
#   --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 \
#   --results_dir experiments/chinchilla/results_matched_mean
# log "avg_50m_k2 matched rerun done. Sleeping 2 min..."
# sleep 120

# log "Starting 50m k=2 learnable content + position (2B tokens) on 1 GPU..."
# python -m torch.distributed.run --standalone --nproc_per_node=1 \
#   experiments/chinchilla/train.py \
#   --model avg_50m_k2_learnable_pos \
#   --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
#   --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 --resume
# log "avg_50m_k2_learnable_pos done. Sleeping 2 min..."
# sleep 120

# log "Starting original avg_50m_k4 with matched ablation settings (4.07B tokens) on 1 GPU..."
# python -m torch.distributed.run --standalone --nproc_per_node=1 \
#   experiments/chinchilla/train.py \
#   --model avg_50m_k4 \
#   --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
#   --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 \
#   --results_dir experiments/chinchilla/results_matched_mean --resume
# log "avg_50m_k4 matched rerun done. Sleeping 2 min..."
# sleep 120

# log "Starting 50m k=4 learnable content + position (4.07B tokens) on 1 GPU..."
# python -m torch.distributed.run --standalone --nproc_per_node=1 \
#   experiments/chinchilla/train.py \
#   --model avg_50m_k4_learnable_pos \
#   --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
#   --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 --resume
# log "avg_50m_k4_learnable_pos done. All matched follow-up runs complete."
# sleep 120

# Update-count control: k=1 baseline with HALF the batch (8 instead of 16),
# so 1B tokens take 122,071 optimizer steps — the same update count (and the
# same cosine-schedule length) as the matched avg_50m_k2 run, at identical
# FLOPs to the standard baseline.  If this lands near the baseline's 4.437,
# the "k=2 only wins because it takes 2x updates" objection is dead; if it
# lands near 4.363, the headline result is a step-count artifact.
# eval_batches 32 (x batch 8 = 256 eval seqs) matches the matched runs'
# eval set size (16 x 16).  Logs go to results_update_control/model1_50m/.

# log "Starting 50m k=1 update-count control (1B tokens, batch 8 -> 122k steps) on 1 GPU..."
# python -m torch.distributed.run --standalone --nproc_per_node=1 \
#   experiments/chinchilla/train.py \
#   --model model1_50m \
#   --batch_size 8 --seq_len 1024 --log_steps 200 --eval_batches 32 \
#   --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 \
#   --results_dir experiments/chinchilla/results_update_control
# log "model1_50m update-count control done."

# ==================================================================
# k=8 ABLATIONS  (Config A: seq_len 1024, transformer L = 128)
# Same matched protocol: 1 GPU, batch 16, seq_len 1024, eval_batches 16.
# Each run trains 8.14B raw tokens (= 8 × 20N for 51M tied params).
# ⚠  8.14B tokens — make sure /data/fineweb has enough shards.
# ==================================================================

log "Starting 50m k=8 mean (matched rerun, 8.14B tokens) on 1 GPU..."
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model avg_50m_k8 \
  --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 \
  --results_dir experiments/chinchilla/results_matched_mean --resume
log "avg_50m_k8 matched rerun done. Sleeping 2 min..."
sleep 120

log "Starting 50m k=8 learnable pooling (8.14B tokens) on 1 GPU..."
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model avg_50m_k8_learnable \
  --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 --resume
log "avg_50m_k8_learnable done. Sleeping 2 min..."
sleep 120

log "Starting 50m k=8 exponential weights (8.14B tokens) on 1 GPU..."
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model avg_50m_k8_wexp \
  --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 --resume
log "avg_50m_k8_wexp done. Sleeping 2 min..."
sleep 120

log "Starting 50m k=8 learnable content + position (8.14B tokens) on 1 GPU..."
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model avg_50m_k8_learnable_pos \
  --batch_size 16 --seq_len 1024 --log_steps 100 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 --resume
log "avg_50m_k8_learnable_pos done. All k=8 ablations complete."
