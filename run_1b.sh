#!/bin/bash
set -e

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# ── 1B Config A runs on 8× A6000 (48 GB each) ────────────────────────────
#
# Architecture: d=1664, h=26, l=28, head_dim=64  (975,945,984 tied params)
# LR: 1.1e-4  (2e-4 × sqrt(512/1664))
#
# Memory budget per GPU (48 GB):
#   Model params (bf16):    ~2.0 GB
#   Optimizer (fp32 + mom): ~8.0 GB
#   Gradients (bf16):       ~2.0 GB
#   Fixed overhead:        ~12.0 GB
#   Remaining for acts:    ~36.0 GB (with grad_checkpoint)
#
# batch_size=4 per GPU (conservative, ~30 GB activations with grad_ckpt)
# Global batch = 4 × 8 GPUs = 32 seqs/step = 32,768 raw tokens/step
# Same global batch as the 500M protocol — no gradient accumulation needed.
#
# k=1:  20B raw tokens, L=1024  →   610,351 steps
# k=2:  40B raw tokens, L= 512  → 1,220,703 steps
#
# ⏱  Expected throughput:
#    8× A6000 at ~40% MFU ≈ 8 × 310 TFLOPS × 0.40 = 992 effective TFLOPS
#    FLOPs/token ≈ 6 × 976M = 5.86 TFLOPS per token (fwd+bwd)
#    TPS ≈ 992 / 5.86e-3 ≈ 169k tok/s → ~1.4 days for k=1
#    (Real-world with comm overhead: ~100-130k tok/s → 1.8-2.3 days for k=1)
#
# ⚠  DATA: pre-tokenize 40B tokens before launching k=2:
#      python experiments/chinchilla/fineweb_loader.py \
#        --data_dir /data/fineweb --max_train_tokens 40000000000 --num_proc 16
# ────────────────────────────────────────────────────────────────────────

# ── NCCL tuning for 8× A6000 ──
# Disable P2P if GPUs are on different PCIe root complexes (common on multi-GPU
# servers without NVLink). Prevents hangs from failed P2P reads.
export NCCL_P2P_DISABLE=1

# Let NCCL auto-select the best algorithm for the topology
# (Tree can be slower than Ring on some PCIe topologies)
# export NCCL_ALGO=Tree

# Increase buffer size for large allreduce (1B model has big gradients)
export NCCL_BUFFSIZE=16777216  # 16 MB (default 4 MB)

# Timeout: 10 minutes (default 30 min is too long to detect real hangs)
export NCCL_TIMEOUT=600000


# ── CUDA tuning ──
# Allow TF32 for matmuls (A6000 Ampere supports it, gives ~2x over FP32)
export NVIDIA_TF32_OVERRIDE=1

# ── Data loading ──
export OMP_NUM_THREADS=1

# ────────────────────────────────────────────────────────────────────────

NPROC=8
BATCH=10
SEQ_LEN=1024
NUM_WORKERS=8
LOG_STEPS=500
EVAL_BATCHES=16
CKPT_STEPS=25000
KEEP_CKPTS=3
DATA_DIR=/data/fineweb

# Use venv's torch.distributed.run (system torchrun uses /usr/bin/python3 which lacks deps)
TORCHRUN="python3 -m torch.distributed.run"

log "=== 1B training on 8× A6000 ==="
log "Global batch: ${BATCH} × ${NPROC} = $((BATCH * NPROC)) seqs = $((BATCH * NPROC * SEQ_LEN)) tokens/step"
log "NCCL: P2P_DISABLE=1, ALGO=Tree, BUFFSIZE=16MB"

# ── k=1: 20B tokens ──────────────────────────────────────────────────────
log "Starting 1B k=1 (standard, 20B tokens)..."
$TORCHRUN --standalone --nproc_per_node=$NPROC \
  experiments/chinchilla/train.py \
  --model model1_1b \
  --batch_size $BATCH \
  --seq_len $SEQ_LEN \
  --log_steps $LOG_STEPS \
  --eval_batches $EVAL_BATCHES \
  --num_workers $NUM_WORKERS \
  --data_dir $DATA_DIR \
  --checkpoint_steps $CKPT_STEPS \
  --keep_last_checkpoints $KEEP_CKPTS \
  --resume

log "model1_1b done. Sleeping 2 min before k=2..."
sleep 120

# ── k=2: 40B tokens ──────────────────────────────────────────────────────
# log "Starting 1B k=2 (2× averaging, 40B tokens)..."
# $TORCHRUN --standalone --nproc_per_node=$NPROC \
#   experiments/chinchilla/train.py \
#   --model avg_1b_k2 \
#   --batch_size $BATCH \
#   --seq_len $SEQ_LEN \
#   --log_steps $LOG_STEPS \
#   --eval_batches $EVAL_BATCHES \
#   --num_workers $NUM_WORKERS \
#   --data_dir $DATA_DIR \
#   --checkpoint_steps $CKPT_STEPS \
#   --keep_last_checkpoints $KEEP_CKPTS \
#   --resume

# log "avg_1b_k2 done."
# log "=== All 1B runs complete ==="
