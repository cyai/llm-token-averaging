cat > run_sequence.sh << 'EOF'
#!/bin/bash
set -e

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

log "Starting avg_50m_k16..."
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/chinchilla/train.py \
  --model avg_50m_k16 \
  --batch_size 16 --seq_len 1024 --log_steps 50 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 \
  --resume \
log "avg_50m_k16 done. Sleeping 2 min..."
sleep 120

log "Starting avg_50m_k32..."
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/chinchilla/train.py \
  --model avg_50m_k32 \
  --batch_size 16 --seq_len 1024 --log_steps 50 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000
log "avg_50m_k32 done. Sleeping 2 min..."
sleep 120

log "Starting avg_50m_k64..."
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/chinchilla/train.py \
  --model avg_50m_k64 \
  --batch_size 16 --seq_len 3072 --log_steps 50 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000
log "avg_50m_k64 done. Sleeping 2 min..."
sleep 120

log "Starting avg_50m_k128..."
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/chinchilla/train.py \
  --model avg_50m_k128 \
  --batch_size 16 --seq_len 1024 --log_steps 50 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000
log "avg_50m_k128 done. Sleeping 2 min..."
sleep 120

EOF

chmod +x run_sequence.sh
nohup bash run_sequence.sh > experiments/chinchilla/logs/run_sequence.out 2>&1 &
echo "Sequencer running as PID $!"