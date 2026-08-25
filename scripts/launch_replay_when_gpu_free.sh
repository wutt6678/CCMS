#!/usr/bin/env bash
# Wait for a GPU with enough free memory, then launch the CCMS replay.
# Usage: launch_replay_when_gpu_free.sh <min_free_gib> <mode>
#   mode: smoke (5 families) | full (all 20 families)
set -u
MIN_GIB="${1:-24}"
MODE="${2:-smoke}"
source /scratch/wutiantong/miniconda3/etc/profile.d/conda.sh
conda activate midp-qwen35
cd /scratch/wutiantong/CCMS || exit 1

echo "[wait] looking for a GPU with >= ${MIN_GIB} GiB free..."
while true; do
  GPU=""
  while read -r idx free_mib; do
    idx="${idx//[^0-9]/}"
    free_mib="${free_mib//[^0-9]/}"
    [ -z "$idx" ] && continue
    free_gib=$((free_mib / 1024))
    if [ "$free_gib" -ge "$MIN_GIB" ]; then GPU="$idx"; break; fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  if [ -n "$GPU" ]; then
    echo "[wait] GPU $GPU free; launching replay ($(date))"
    break
  fi
  sleep 1
done

DATESTAMP=$(date +%Y-%m-%d)
if [ "$MODE" = "smoke" ]; then
  RUN_ID="smoke-${DATESTAMP}-qwen35-9b"
  python -m causal_mllm.cli.replay \
    --input-dir outputs/families/scale_b_smoke \
    --max-families 5 --device "cuda:${GPU}" --run-id "$RUN_ID"
else
  RUN_ID="scale-b-${DATESTAMP}-qwen35-9b"
  python -m causal_mllm.cli.replay \
    --input-dir outputs/families/scale_b_smoke \
    --device "cuda:${GPU}" --run-id "$RUN_ID"
fi
echo "[done] exit=$? run_id=${RUN_ID}"
