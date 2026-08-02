#!/bin/bash
# Upload all training checkpoints + loss logs to Hugging Face.
# Run on the training machine (not locally).
#
# Prerequisites:
#   pip install -U huggingface_hub
#   export HF_TOKEN=hf_...          # or: huggingface-cli login / hf auth login
#
# Examples:
#   ./upload_checkpoints_hf.sh --dry-run
#   ./upload_checkpoints_hf.sh
#   ./upload_checkpoints_hf.sh --only model1_500m avg_500m_k2 model1_1b avg_1b_k2
#   ./upload_checkpoints_hf.sh --include-old
#   ./upload_checkpoints_hf.sh --public
#   ./upload_checkpoints_hf.sh --namespace my-user   # override default FAIRC

set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set. Relying on cached 'hf auth login' credentials."
fi

python -m pip install -q -U "huggingface_hub>=0.26"

exec python experiments/chinchilla/upload_to_hf.py "$@"
