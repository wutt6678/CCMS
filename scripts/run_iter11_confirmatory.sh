#!/usr/bin/env bash
# Iteration 11.6 — confirmatory replay of the ENTIRE 100-family frozen panel
# against all four targets: 100 families x 6 variants x 4 targets = 2400
# generations.
#
# One target per GPU, in parallel. That is what the frozen protocol's
# scheduling note asks for ("one model on one GPU for all its variants where
# possible"), and the stage-scoped output exclusions make it safe: every
# confirmatory run excludes outputs/iteration_11/generations/, so no target's
# journal dirties the tree another target is certifying. The scheduling slot is
# excluded from the run fingerprint, so which GPU a target landed on does not
# change its evidence.
#
# Each worker waits for enough FREE memory on its own device before loading.
# The thresholds are calibrated from the declared safetensors shard_bytes in
# the 11.2 preflight artifacts plus the ~0.8 GiB of CUDA context and
# activations measured on the 11.5 runs, with ~25% margin:
#
#   target          shard_bytes  measured resident  required here
#   qwen35_2b       4.24 GiB     5182 MiB           7000 MiB
#   ministral3_3b   7.17 GiB     8023 MiB          10000 MiB
#   qwen35_4b       8.68 GiB     9357 MiB          11500 MiB
#   phi4_mm        10.60 GiB    ~11800 MiB         14000 MiB
#
# The wait does not make the load safe on a contended GPU: the check and the
# allocation are not atomic, and qwen35_4b's first 11.5 attempt died with
# torch.OutOfMemoryError after the check passed at 26935 MiB free because
# another job took ~19 GiB in between. What makes it safe is that an OOM at
# load time writes nothing at all, and the evidence-protection guard refuses to
# restart over a journal that does hold evidence -- so a failed attempt costs
# time, never correctness. Re-run this script to retry; completed targets are
# skipped by the guard and interrupted ones continue with RESUME=1.
#
# The tree must be clean when this starts: the confirmatory gate refuses to
# generate otherwise. This script commits nothing.
#
# Usage:
#   bash scripts/run_iter11_confirmatory.sh
#   ASSIGNMENTS="phi4_mm=cuda:1" bash scripts/run_iter11_confirmatory.sh
#   RESUME=1 bash scripts/run_iter11_confirmatory.sh
set -uo pipefail

cd /scratch/wutiantong/CCMS
source /scratch/wutiantong/miniconda3/etc/profile.d/conda.sh
conda activate ccms-iter11

PANEL="outputs/scale_c/families_panel"
LOCK="outputs/iteration_11/preflight/resolved_models.lock.yaml"
# The frozen uniform cap. ReplayConfig defaults to 256, so this MUST be passed
# explicitly or the confirmatory gate rejects the run for cap mismatch.
CAP="${CAP:-1536}"
ASSIGNMENTS="${ASSIGNMENTS:-qwen35_2b=cuda:3 ministral3_3b=cuda:0 qwen35_4b=cuda:2 phi4_mm=cuda:1}"
RESUME="${RESUME:-}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-259200}"
STAGGER_SECONDS="${STAGGER_SECONDS:-45}"
LOG_DIR="${LOG_DIR:-/scratch/wutiantong/logs_iter11_6}"

mkdir -p "$LOG_DIR"

required_mib() {
  case "$1" in
    qwen35_2b)     echo  7000 ;;
    ministral3_3b) echo 10000 ;;
    qwen35_4b)     echo 11500 ;;
    phi4_mm)       echo 14000 ;;
    *)             echo 12000 ;;
  esac
}

free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    -i "${1#cuda:}" | tr -d ' '
}

wait_for_headroom() {
  local device="$1" need free waited=0
  need="$(required_mib "$2")"
  while :; do
    free="$(free_mib "$device")"
    if [ -n "$free" ] && [ "$free" -ge "$need" ]; then
      echo "  ${device}: ${free} MiB free (>= ${need} required)"
      return 0
    fi
    echo "  waiting for ${device}: ${free:-?} MiB free, ${need} required"
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
    if [ "$waited" -ge "$MAX_WAIT_SECONDS" ]; then
      echo "  gave up after ${waited}s waiting for ${need} MiB on ${device}"
      return 1
    fi
  done
}

run_one() {
  local key="$1" device="$2"
  echo "=== ${key} on ${device} ==="
  echo "HEAD $(git rev-parse HEAD)"
  if ! wait_for_headroom "$device" "$key"; then
    echo "---- SKIPPED ${key}: insufficient headroom on ${device} ----"
    return 1
  fi
  local args=(--input-dir "$PANEL" --model-key "$key" --device "$device"
              --lock "$LOCK" --max-new-tokens "$CAP"
              --run-id "confirmatory-100f-t${CAP}-${key}")
  if [ -n "$RESUME" ]; then
    args+=(--resume)
  fi
  PYTHONPATH=src python -m causal_mllm.cli.replay "${args[@]}"
  local code=$?
  echo "---- ${key} exit: ${code} ----"
  return "$code"
}

echo "=== Iteration 11.6 confirmatory replay ==="
echo "HEAD        $(git rev-parse HEAD)"
echo "panel       ${PANEL} ($(wc -l < ${PANEL}/validated_families.jsonl) families)"
echo "cap         ${CAP}"
echo "lock        ${LOCK}"
echo "assignments ${ASSIGNMENTS}"
echo "logs        ${LOG_DIR}"
echo "tracked modifications (must be empty):"
git status --porcelain --untracked-files=no
echo "untracked, non-ignored files (must be empty):"
git status --porcelain --untracked-files=all | grep '^??' || true
echo "editable installs in this environment:"
python -m pip freeze | grep -E '^-e ' || echo "  (none)"

pids=()
keys=()
for assignment in $ASSIGNMENTS; do
  key="${assignment%%=*}"
  device="${assignment#*=}"
  if [ "$key" = "$assignment" ]; then
    echo "FATAL: assignment '$assignment' has no =device part" >&2
    exit 2
  fi
  run_one "$key" "$device" > "${LOG_DIR}/${key}.log" 2>&1 &
  pids+=($!)
  keys+=("$key")
  # Stagger the loads: four simultaneous safetensors reads contend for the
  # page cache and make each load slower than the sum of them would be.
  sleep "$STAGGER_SECONDS"
done

echo
echo "launched ${#pids[@]} worker(s); waiting"
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "  ${keys[$i]}: ok"
  else
    echo "  ${keys[$i]}: FAILED (see ${LOG_DIR}/${keys[$i]}.log)"
    failed=$((failed + 1))
  fi
done

echo
echo "=== all workers finished (${failed} failed) ==="
echo "verify with: python3 scripts/iter11_replay_checks.py --all"
exit "$((failed > 0))"
