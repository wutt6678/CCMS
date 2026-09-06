#!/usr/bin/env bash
# Iteration 11.7 — judge the confirmatory panel with the frozen ensemble.
#
#   primary A   qwen3.8-max   seed 42
#   primary B   glm-5.2       seed 43
#   adjudicator kimi-k3       seed 99, ALL A/B disagreements
#   rubric      v1.1 (ce6c2005...)
#
# None of that is configurable here. run_llm_judge_pipeline.py reads the
# frozen identities from configs/evaluation/llm_judge_credentials.conf and
# the profile selected by CCMS_SCALE; this script only decides WHICH target
# is judged WHEN, and how many sessions run at once.
#
# WHY PROCESS-LEVEL PARALLELISM AND NOT THREADS
# The pipeline and MultimodalLLMJudge are strictly serial: there is no
# ThreadPoolExecutor anywhere in either. Measured against the gateway with
# real judge-shaped payloads (~10.4k chars, images attached), one call at a
# time gives A 1.95/min, B 3.69/min and the adjudicator 1.30/min, so
# judging 2400 cells serially would take about 36 hours. Eight CONCURRENT
# requests were measured with zero errors and near-linear scaling (A rises
# 1.95 -> 5.57 -> 9.59/min at concurrency 1 -> 4 -> 8), so the same work
# done as eight processes takes about 9 hours. The pipeline already supports
# the split that makes this possible: CCMS_JUDGES selects a subset, a
# fingerprint sidecar makes a completed judge skip-safe on re-run, and a
# split-mode process never finalizes.
#
# WHY ONE TARGET PER SESSION
# prepare_blinded_items derives item_id from the index into a seed-42
# shuffle of THAT target's journal, so all four targets emit the same
# item_id range over the same 600 cells. Pooling them would collide every id
# four ways. Each target is therefore its own CCMS_SCALE profile with its
# own output_dir.
#
# PHASES
#   1. A and B for every eligible target, as separate processes (up to 2 x
#      MAX_TARGETS concurrent). Each writes llm_labels_judge_{A,B}.json and
#      stops without finalizing.
#   2. The full pipeline once per target, which skips both primaries via the
#      fingerprint sidecar and then runs agreement, adjudication of ALL
#      disagreements, labels, evaluation and per-judge sensitivity. Phase 2
#      cannot start for a target before both of its primaries are complete,
#      and it is the slower phase: at the measured 0.398-0.502 disagreement
#      rate the adjudicator alone takes 955-1205 calls across the panel.
#
# A target is skipped unless its 11.6 completion gate recorded PASS. phi4_mm
# is skipped today for exactly that reason: its first attempt was stopped at
# 197 of 600 cells by the LongRoPE defect documented under
# outputs/iteration_11/diagnostics/phi4_longrope/, and judging a journal
# that cannot pass coverage would produce labels over a censored panel.
#
# Usage:
#   bash scripts/run_iter11_judging.sh
#   TARGETS="qwen35_2b" bash scripts/run_iter11_judging.sh
#   PHASE=1 bash scripts/run_iter11_judging.sh      # primaries only
#   PHASE=1 JUDGES="A" bash scripts/run_iter11_judging.sh   # one primary only
#
# JUDGES selects WHICH primaries phase 1 launches. It exists because a
# provider-side failure is per-identity, not per-target: aliyun's input
# moderation refused 2 of the 600 cells for judge A alone, so all four of A's
# arms stopped at item-0164 while all four of B's completed. Re-running the
# pair would start a second B process against the checkpoint the first one is
# still writing. Selecting A lets the failed identity be resumed on its own --
# the fingerprint-bound checkpoints make that a continuation, not a restart.
set -uo pipefail

cd /scratch/wutiantong/CCMS
source /scratch/wutiantong/miniconda3/etc/profile.d/conda.sh
conda activate ccms-iter11

GENERATIONS="outputs/iteration_11/generations"
TARGETS="${TARGETS:-qwen35_2b qwen35_4b ministral3_3b phi4_mm}"
# Eight concurrent gateway requests is what was measured clean; 2 judges per
# target, so four targets is the ceiling that measurement supports.
MAX_TARGETS="${MAX_TARGETS:-4}"
PHASE="${PHASE:-1,2}"
JUDGES="${JUDGES:-A B}"
LOG_DIR="${LOG_DIR:-/scratch/wutiantong/logs_iter11_7}"

mkdir -p "$LOG_DIR"

eligible=()
for key in $TARGETS; do
  run_dir="${GENERATIONS}/${key}/confirmatory-100f-t1536-${key}"
  checks="${run_dir}/iteration_11_replay_checks.json"
  if [ ! -f "${run_dir}/replay_report.json" ]; then
    echo "SKIP ${key}: no replay_report.json (run incomplete)"
    continue
  fi
  if [ ! -f "$checks" ]; then
    echo "SKIP ${key}: no ${checks} — run scripts/iter11_replay_checks.py first"
    continue
  fi
  verdict=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('verdict',''))" "$checks" 2>/dev/null)
  if [ "$verdict" != "PASS" ]; then
    echo "SKIP ${key}: completion gate verdict is '${verdict:-absent}', not PASS"
    continue
  fi
  eligible+=("$key")
done

if [ "${#eligible[@]}" -eq 0 ]; then
  echo "no eligible targets; nothing to judge"
  exit 1
fi
if [ "${#eligible[@]}" -gt "$MAX_TARGETS" ]; then
  echo "FATAL: ${#eligible[@]} eligible targets exceeds MAX_TARGETS=${MAX_TARGETS},"
  echo "       which is the concurrency the gateway measurement supports." >&2
  exit 2
fi
echo "eligible: ${eligible[*]}"
echo "judges:   ${JUDGES}"
echo "logs:     ${LOG_DIR}"
echo

run_profile() {  # key, judges-label, CCMS_JUDGES value ("" = full mode)
  local key="$1" label="$2" judges="$3" log
  log="${LOG_DIR}/${key}_${label}.log"
  echo "=== ${key} ${label} $(date -Is) ===" >> "$log"
  if [ -n "$judges" ]; then
    CCMS_SCALE="iteration_11_${key}" CCMS_JUDGES="$judges" \
      PYTHONPATH=src python scripts/run_llm_judge_pipeline.py >> "$log" 2>&1
  else
    CCMS_SCALE="iteration_11_${key}" \
      PYTHONPATH=src python scripts/run_llm_judge_pipeline.py >> "$log" 2>&1
  fi
  local code=$?
  echo "---- ${key} ${label} exit ${code} ----" >> "$log"
  return "$code"
}

failed=0
if [[ ",${PHASE}," == *,1,* ]]; then
  echo "=== phase 1: primary judges ${JUDGES}, one process each ==="
  pids=(); names=()
  for key in "${eligible[@]}"; do
    for judge in $JUDGES; do
      run_profile "$key" "judge_${judge}" "$judge" &
      pids+=($!); names+=("${key}/judge_${judge}")
      # Stagger so eight processes do not open eight TLS handshakes and
      # eight 10k-char prompts at the same instant.
      sleep 5
    done
  done
  echo "launched ${#pids[@]} primary-judge process(es); waiting"
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "  ${names[$i]}: ok"
    else echo "  ${names[$i]}: FAILED"; failed=$((failed + 1)); fi
  done
fi

if [[ ",${PHASE}," == *,2,* ]]; then
  echo
  echo "=== phase 2: agreement, adjudication, labels, evaluation ==="
  pids=(); names=()
  for key in "${eligible[@]}"; do
    run_profile "$key" "finalize" "" &
    pids+=($!); names+=("${key}/finalize")
    sleep 5
  done
  echo "launched ${#pids[@]} finalize process(es); waiting"
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "  ${names[$i]}: ok"
    else echo "  ${names[$i]}: FAILED"; failed=$((failed + 1)); fi
  done
fi

echo
echo "=== done (${failed} failure(s)) ==="
echo "artifacts: outputs/iteration_11/judge/<model_key>/"
exit "$((failed > 0))"
