#!/usr/bin/env bash
# Run the rebuttal experiments through the GPU queue, then build the tables.
#
#   GPUS=0,1,2,3,4,5,6,7 DATA_DIR=/data/fineweb bash experiments/rebuttal/run.sh            # everything
#   GPUS=0,1 bash experiments/rebuttal/run.sh rescore_hub kv                                # some sets
#   bash experiments/rebuttal/run.sh --list                                                 # show plan
#
# Safe to stop (Ctrl-C) and restart: finished jobs are skipped and training resumes
# from its last checkpoint. Logs: experiments/rebuttal/results/logs/<job>.log
set -euo pipefail
cd "$(dirname "$0")/../.."

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
DATA_DIR=${DATA_DIR:-/data/fineweb}
PYTHON=${PYTHON:-python}
OUT=${RB_OUT:-experiments/rebuttal/results}

LIST=""
SETS=()
for a in "$@"; do
  if [[ "$a" == "--list" ]]; then LIST="--list"; else SETS+=("$a"); fi
done
[[ ${#SETS[@]} -eq 0 ]] && SETS=(all)

export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}

$PYTHON experiments/rebuttal/jobs.py --sets "${SETS[@]}" --gpus "$GPUS" \
  --data_dir "$DATA_DIR" --out "$OUT" --max_foreign_mb "${MAX_FOREIGN_MB:-12000}" $LIST
[[ -n "$LIST" ]] && exit 0

$PYTHON experiments/rebuttal/aggregate.py --out "$OUT/analysis" \
  --rescore "$OUT/rescore_local.jsonl" "$OUT/rescore_hub.jsonl"
$PYTHON experiments/rebuttal/summarize.py --results "$OUT"
