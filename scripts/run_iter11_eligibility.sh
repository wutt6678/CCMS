#!/usr/bin/env bash
# Iteration 11.5 — replay the pre-registered 12-family selection against each
# target, SEQUENTIALLY on cuda:3 (standing instruction: use gpu3).
#
# Sequential rather than parallel: the four checkpoints together exceed what
# cuda:3 holds while other work is resident, and one target's OOM must not be
# able to kill a sibling's 72-cell run. Each target's report is written and
# self-validated before the next starts, so a failure leaves the earlier
# targets' evidence intact and only its own key needs re-running.
#
# The gate inside iter11_run_eligibility.py refuses to generate unless the
# tree is clean, so this script commits nothing and must be launched from a
# committed state.
#
# Usage:
#   bash scripts/run_iter11_eligibility.sh                    # three targets
#   KEYS="phi4_mm" bash scripts/run_iter11_eligibility.sh     # just Phi-4
#   RESUME=1 bash scripts/run_iter11_eligibility.sh           # continue
set -uo pipefail

cd /scratch/wutiantong/CCMS
source /scratch/wutiantong/miniconda3/etc/profile.d/conda.sh
conda activate ccms-iter11

DEVICE="${DEVICE:-cuda:3}"
KEYS="${KEYS:-qwen35_2b qwen35_4b ministral3_3b}"
RESUME="${RESUME:-}"
POLL_SECONDS="${POLL_SECONDS:-120}"
# cuda:3 is shared with several long-running jobs, so "wait" has to mean
# longer than any one of them is likely to hold the card. Give up after three
# days rather than six hours: a poller that exits silently overnight looks
# exactly like a poller that is still waiting, and the difference is only
# discovered when the evidence is needed.
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-259200}"
GPU_INDEX="${DEVICE#cuda:}"

# MiB that must be FREE before a load is attempted, per target. Sized from
# the declared checkpoint parameter counts in BF16 plus headroom for the KV
# cache and the vision tower's activations at the frozen 1536 cap. A load
# that dies halfway is worse than waiting: it leaves no artifact and wastes
# every cell the run had already generated.
required_mib() {
  case "$1" in
    qwen35_2b)     echo 10000 ;;   #  2.27 B params
    ministral3_3b) echo 12000 ;;   #  3.40 B params
    qwen35_4b)     echo 13000 ;;   #  4.02 B params
    phi4_mm)       echo 17000 ;;   #  5.57 B params + bundled LoRA towers
    *)             echo 13000 ;;
  esac
}

free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    -i "$GPU_INDEX" | tr -d ' '
}

wait_for_headroom() {
  local need free waited=0
  need="$(required_mib "$1")"
  while :; do
    free="$(free_mib)"
    if [ -n "$free" ] && [ "$free" -ge "$need" ]; then
      echo "  ${DEVICE}: ${free} MiB free (>= ${need} required)"
      return 0
    fi
    echo "  waiting for ${DEVICE}: ${free:-?} MiB free, ${need} required"
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
    if [ "$waited" -ge 21600 ]; then
      echo "  gave up after ${waited}s waiting for ${need} MiB on ${DEVICE}"
      return 1
    fi
  done
}

echo "=== Iteration 11.5 eligibility ==="
echo "HEAD       $(git rev-parse HEAD)"
echo "device     ${DEVICE}"
echo "targets    ${KEYS}"
echo "tracked modifications (must be empty):"
git status --porcelain --untracked-files=no
echo "untracked, non-ignored files (must be empty):"
git status --porcelain --untracked-files=all | grep '^??' || true
echo "editable installs in this environment:"
python -m pip freeze | grep -E '^-e ' || echo "  (none)"

for key in $KEYS; do
  echo ""
  echo "################ ${key} ################"
  if ! wait_for_headroom "$key"; then
    echo "---- SKIPPED ${key}: insufficient GPU headroom ----"
    continue
  fi
  args=(--model-key "$key" --device "$DEVICE")
  if [ -n "$RESUME" ]; then
    args+=(--resume)
  fi
  PYTHONPATH=src python scripts/iter11_run_eligibility.py "${args[@]}"
  echo "---- exit: $? ----"
done

echo ""
echo "=== done ==="
