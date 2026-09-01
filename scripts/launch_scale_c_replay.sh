#!/usr/bin/env bash
# Scale-C (Iteration 10) replay launcher — frozen protocol values only.
# Protocol: configs/experiments/scale_c_protocol.json (FROZEN).
#
# Usage: launch_scale_c_replay.sh <mode>
#   mode: preflight (first 10 panel families) | full (all 100 families)
#
# Output-cap policy (frozen): start at 1536. If truncation is material
# or differs by variant, raise the cap UNIFORMLY and rerun the ENTIRE
# panel — never individual conditions.
set -euo pipefail
MODE="${1:-preflight}"
MIN_GIB="${2:-24}"
MAX_NEW_TOKENS="${3:-1536}"

# Frozen by the Scale-C protocol — do not edit after seeing results.
REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
INPUT_DIR="outputs/scale_c/families_panel"
OUTPUT_ROOT="outputs/scale_c/replay_runs"

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
  sleep 30
done

DATESTAMP=$(date +%Y-%m-%d)
if [ "$MODE" = "preflight" ]; then
  RUN_ID="scale-c-preflight-${DATESTAMP}-t${MAX_NEW_TOKENS}-qwen35-9b"
  python -m causal_mllm.cli.replay \
    --input-dir "$INPUT_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --max-families 10 --max-new-tokens "$MAX_NEW_TOKENS" \
    --model-revision "$REVISION" \
    --device "cuda:${GPU}" --run-id "$RUN_ID"
elif [ "$MODE" = "full" ]; then
  # Run ID frozen by the protocol manifest.
  RUN_ID="scale-c-100-t1536-qwen35-9b"
  python -m causal_mllm.cli.replay \
    --input-dir "$INPUT_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --model-revision "$REVISION" \
    --device "cuda:${GPU}" --run-id "$RUN_ID"
else
  echo "unknown mode: $MODE (expected preflight|full)" >&2
  exit 2
fi
echo "[done] exit=0 run_id=${RUN_ID}"
