#!/bin/bash
# ============================================================================
# Quick benchmark evaluation for all token-averaging models.
#
# Prerequisites:
#   cd /Users/vardh/token-averaging
#   source venv/bin/activate
#   pip install datasets huggingface_hub   (already installed)
#   huggingface-cli login
#
# Usage:
#   ./experiments/benchmarks/run_eval.sh           # all models
#   ./experiments/benchmarks/run_eval.sh --quick   # fast test (100 examples)
# ============================================================================

set -e
cd "$(dirname "$0")/../.."

# Activate venv
source venv/bin/activate

LIMIT_FLAG=""
if [[ "$1" == "--quick" ]]; then
    LIMIT_FLAG="--limit 100"
    echo ">>> Quick mode: 100 examples per task"
fi

TASKS="lambada,hellaswag,piqa,arc_easy,winogrande"

echo ""
echo "============================================"
echo "  Token Averaging Downstream Benchmarks"
echo "  Tasks: $TASKS"
echo "============================================"
echo ""

# ----------- 50M Main: k=1 vs k=2 vs k=4 -----------
# NOTE: model1_50m, avg_50m_k2, avg_50m_k4 have broken LFS pointers on HF.
# You need to re-upload from training machine:
#   python experiments/chinchilla/upload_to_hf.py --public --recreate --only model1_50m avg_50m_k2 avg_50m_k4
# For now, use matched-mean k=2 as a substitute for avg_50m_k2.
# echo ">>> 50M Main Models (available ones)"
# python3 experiments/benchmarks/eval_downstream.py \
#     --models model2_50m_ctx2n \
#     --tasks "$TASKS" $LIMIT_FLAG || echo "  [WARN] failed, continuing..."

# # Use the update-control model as k=1 baseline since model1_50m has broken LFS
# python3 experiments/benchmarks/eval_downstream.py \
#     --model model1_50m --tag "update_control" \
#     --tasks "$TASKS" $LIMIT_FLAG || echo "  [WARN] update_control-model1_50m failed, continuing..."

# # Use matched-mean as k=2/k=4/k=8 baselines
# python3 experiments/benchmarks/eval_downstream.py \
#     --model avg_50m_k2 --tag "matched_mean" \
#     --tasks "$TASKS" $LIMIT_FLAG || echo "  [WARN] matched_mean-avg_50m_k2 failed, continuing..."

# python3 experiments/benchmarks/eval_downstream.py \
#     --model avg_50m_k4 --tag "matched_mean" \
#     --tasks "$TASKS" $LIMIT_FLAG || echo "  [WARN] matched_mean-avg_50m_k4 failed, continuing..."

# python3 experiments/benchmarks/eval_downstream.py \
#     --model avg_50m_k8 --tag "matched_mean" \
#     --tasks "$TASKS" $LIMIT_FLAG || echo "  [WARN] matched_mean-avg_50m_k8 failed, continuing..."

# ----------- 50M k=2 Ablations (already done) -----------
# echo ""
# echo ">>> 50M k=2 Ablations"
# python3 experiments/benchmarks/eval_downstream.py \
#     --models avg_50m_k2_learnable avg_50m_k2_wexp avg_50m_k2_learnable_pos \
#     --tasks "$TASKS" $LIMIT_FLAG

# ----------- 50M k=4 Ablations (already done) -----------
# echo ""
# echo ">>> 50M k=4 Ablations"
# python3 experiments/benchmarks/eval_downstream.py \
#     --models avg_50m_k4_learnable avg_50m_k4_wexp avg_50m_k4_learnable_pos \
#     --tasks "$TASKS" $LIMIT_FLAG

# ----------- 50M k=8 Ablations (already done) -----------
# echo ""
# echo ">>> 50M k=8 Ablations"
# python3 experiments/benchmarks/eval_downstream.py \
#     --models avg_50m_k8_learnable avg_50m_k8_wexp avg_50m_k8_learnable_pos \
#     --tasks "$TASKS" $LIMIT_FLAG

# ----------- 125M -----------
# NOTE: 125M models not yet uploaded to HF. Uncomment when available.
# echo ""
# echo ">>> 125M Models"
# python3 experiments/benchmarks/eval_downstream.py \
#     --models model1_125m avg_125m_k2 \
#     --tasks "$TASKS" $LIMIT_FLAG

# ----------- 250M (start from avg_250m_k2) -----------
echo ""
echo ">>> 250M Models"
python3 experiments/benchmarks/eval_downstream.py \
    --models avg_250m_k2 \
    --tasks "$TASKS" $LIMIT_FLAG

# ----------- 500M (lower batch for memory) -----------
echo ""
echo ">>> 500M Models"
python3 experiments/benchmarks/eval_downstream.py \
    --models model1_500m avg_500m_k2 \
    --tasks "$TASKS" $LIMIT_FLAG

echo ""
echo "============================================"
echo "  All benchmarks complete!"
echo "  Results: experiments/benchmarks/results/"
echo "============================================"
