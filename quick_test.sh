cat > run_sequence.sh << 'EOF'
#!/bin/bash
set -e

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

log "Starting 50m k=1..."
python -m torch.distributed.run --standalone --nproc_per_node=6 \
  experiments/chinchilla/train.py \
  --model model1_50m \
  --batch_size 16 --seq_len 1024 --log_steps 100 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000
log "  done. Sleeping 2 min..."
sleep 120

log "Starting 50m k=2 iso-FLOPs..."
python -m torch.distributed.run --standalone --nproc_per_node=6 \
  experiments/chinchilla/train.py \
  --model avg_50m_k2_isoflop \
  --batch_size 16 --seq_len 2048 --log_steps 100 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000
log "avg_50m_k2_isoflop done. Sleeping 2 min..."
sleep 120

log "Starting 50m k=4 iso-FLOPs..."
python -m torch.distributed.run --standalone --nproc_per_node=6 \
  experiments/chinchilla/train.py \
  --model avg_50m_k4_isoflop \
  --batch_size 16 --seq_len 4096 --log_steps 100 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000
log "avg_50m_k4_isoflop done. Sleeping 2 min..."


EOF

chmod +x run_sequence.sh
nohup bash run_sequence.sh > experiments/chinchilla/logs/run_sequence.out 2>&1 &
echo "Sequencer running as PID $!"




python -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/chinchilla/train.py \
  --model avg_50m_k8_phased \
  --batch_size 16 --seq_len 8192 --log_steps 50 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 \
  > experiments/chinchilla/logs/run_avg_50m_k8_phased.out 2>&1 &

# k=1 standard, tied
nohup python -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/chinchilla/train.py \
  --model model1_50m_tied \
  --batch_size 16 --seq_len 1024 --log_steps 50 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 \
  > experiments/chinchilla/logs/run_model1_50m_tied.out 2>&1 &

# k=4 averaging, tied
nohup python -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/chinchilla/train.py \
  --model avg_50m_k4_tied \
  --batch_size 16 --seq_len 1024 --log_steps 50 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 \
  > experiments/chinchilla/logs/run_avg_50m_k4_tied.out 2>&1 &

  # k=1 standard, tied, 2n context
nohup python -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/chinchilla/train.py \
  --model model1_50m_tied_2nctx \
  --batch_size 16 --seq_len 1024 --log_steps 50 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 \
  > experiments/chinchilla/logs/run_model1_50m_tied_2nctx.out 2>&1 &


  # k=2 iso-FLOPs
nohup python -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/chinchilla/train.py \
  --model avg_50m_k2_isoflop --batch_size 16 --seq_len 1024 \
  --log_steps 10 --num_workers 4 --data_dir /data/fineweb \
  --checkpoint_steps 50000 > experiments/chinchilla/logs/run_avg_50m_k2_isoflop.out 2>&1 &

# k=4 iso-FLOPs
nohup python -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/chinchilla/train.py \
  --model avg_50m_k4_isoflop --batch_size 16 --seq_len 1024 \
  --log_steps 10 --num_workers 4 --data_dir /data/fineweb \
  --checkpoint_steps 50000 > experiments/chinchilla/logs/run_avg_50m_k4_isoflop.out 2>&1 &






cat > run_125m.sh << 'EOF'
#!/bin/bash
set -e

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

log "Starting 125m k=1 (standard, 2.5B tokens)..."
python -m torch.distributed.run --standalone --nproc_per_node=6 \
  experiments/chinchilla/train.py \
  --model model1_125m \
  --batch_size 16 --seq_len 1024 --log_steps 1000 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000
log "model1_125m done. Sleeping 2 min..."
sleep 120

log "Starting 125m k=2 (2x averaging, 5B tokens)..."
python -m torch.distributed.run --standalone --nproc_per_node=6 \
  experiments/chinchilla/train.py \
  --model avg_125m_k2 \
  --batch_size 16 --seq_len 1024 --log_steps 1000 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000
log "avg_125m_k2 done."

EOF

chmod +x run_125m.sh
nohup bash run_125m.sh > experiments/chinchilla/logs/run_125m.out 2>&1 &
echo "Sequencer running as PID $!"


# 50M iso flops training by doubling context:
# k=1
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model model1_50m \
  --batch_size 16 --seq_len 1024 --log_steps 100 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 

# k=2 iso-FLOPs
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model avg_50m_k2 \
  --batch_size 16 --seq_len 2048 --log_steps 100 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 

# k=4 iso-FLOPs
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model avg_50m_k4 \
  --batch_size 16 --seq_len 4096 --log_steps 100 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 

# model2_50m_ctx2n
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  experiments/chinchilla/train.py \
  --model model2_50m_ctx2n \
  --batch_size 16 --seq_len 2048 --log_steps 100 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 

cat > run_250m.sh << 'EOF'
#!/bin/bash
set -e

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

log "Starting 250m k=1 (standard, 5B tokens) on 2 GPUs..."
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  experiments/chinchilla/train.py \
  --model model1_250m \
  --batch_size 16 --seq_len 1024 --log_steps 1000 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 --resume
log "model1_250m done. Sleeping 2 min..."
sleep 120

log "Starting 250m k=2 (2x averaging, 10B tokens) on 2 GPUs..."
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  experiments/chinchilla/train.py \
  --model avg_250m_k2 \
  --batch_size 16 --seq_len 1024 --log_steps 1000 --eval_batches 16 \
  --num_workers 4 --data_dir /data/fineweb --checkpoint_steps 50000 --resume
log "avg_250m_k2 done."

EOF

chmod +x run_250m.sh
nohup bash run_250m.sh > experiments/chinchilla/logs/run_250m.out 2>&1 &
echo "250M sequencer running as PID $!"

