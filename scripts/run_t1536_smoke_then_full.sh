#!/usr/bin/env bash
# Iteration-9 panel at max_new_tokens=1536 (1024 still truncated 3/30
# smoke completions). LITERAL run ids: no date-rollover races between
# the smoke and the full panel. The full panel launches only after a
# complete, failure-free smoke.
set -u
cd "$(dirname "$0")/.." || exit 1

SMOKE_ID="smoke-2026-08-27-t1536-qwen35-9b"
FULL_ID="scale-b-2026-08-27-t1536-qwen35-9b"
source /scratch/wutiantong/miniconda3/etc/profile.d/conda.sh
conda activate midp-qwen35

wait_for_gpu() {
  local min_gib="$1"
  while true; do
    local gpu=""
    while read -r idx free_mib; do
      idx="${idx//[^0-9]/}"
      free_mib="${free_mib//[^0-9]/}"
      [ -z "$idx" ] && continue
      if [ $((free_mib / 1024)) -ge "$min_gib" ]; then gpu="$idx"; break; fi
    done < <(nvidia-smi --query-gpu=index,memory.free \
             --format=csv,noheader,nounits)
    if [ -n "$gpu" ]; then
      echo "[wait] GPU $gpu free ($(date))"
      echo "$gpu"
      return 0
    fi
    sleep 1
  done
}

echo "[chain] waiting for a GPU with >= 28 GiB free (1536-token cap)..."
GPU=$(wait_for_gpu 28 | tail -1)

echo "[chain] smoke: 5 families x 6 @ 1536 on cuda:${GPU}"
python -m causal_mllm.cli.replay \
  --input-dir outputs/families/scale_b_smoke \
  --max-families 5 --max-new-tokens 1536 \
  --device "cuda:${GPU}" --run-id "$SMOKE_ID" || exit 1

SMOKE_DIR="outputs/replay_runs/${SMOKE_ID}"
if [ "$(wc -l < "$SMOKE_DIR/replay_outputs.jsonl")" -ne 30 ] \
    || [ -s "$SMOKE_DIR/replay_failures.jsonl" ] \
    || ! grep -q '^    "n_truncated": 0,' "$SMOKE_DIR/replay_report.json"; then
  echo "[chain] smoke incomplete or truncated — NOT launching the full panel"
  exit 1
fi

echo "[chain] full panel: 20 families x 6 @ 1536 on cuda:${GPU}"
python -m causal_mllm.cli.replay \
  --input-dir outputs/families/scale_b_smoke \
  --max-new-tokens 1536 \
  --device "cuda:${GPU}" --run-id "$FULL_ID"
echo "[chain] done exit=$?"
