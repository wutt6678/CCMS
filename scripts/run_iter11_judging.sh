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
# Phase 2 also derives the CROSS-ARM common panel before it finalizes anything,
# and gates the four resulting judge_coverage.json artifacts afterwards
# (scripts/iter11_common_panel.py). build_judge_coverage unions judges A and B
# for ONE target session; it does not make the four TARGETS agree. Aliyun's
# input-moderation verdict depends on the request bytes and on the provider's
# policy at the moment of the call, and the four arms reach a given cell
# minutes apart, so one arm can lose a family the other three kept. Each would
# still print a Delta_TV and 11.8 would put the four in one table. Deriving the
# union first means the arms land on one panel by construction; deriving it
# after would mean re-running all four finalizes, and the adjudicator's binding
# fingerprint covers the restricted panel, so that re-calls it on every
# disagreement.
#
# Phase 2 ENFORCES that rather than assuming it, per target, because the
# assumption once nearly cost an arm. Phase 1 counts a failed primary in
# `failed` and then used to run phase 2 anyway; when judge A died on aliyun
# input moderation for one target, that target's finalize was one process-exit
# away from launching. A finalize started early does not fail loudly: the
# pipeline sees a checkpoint with a matching fingerprint and RESUMES the
# primary, so two processes write one checkpoint file and the arm that survives
# looks complete. Phase 2 now checks each target's artifacts before starting
# it and skips the ones that are not ready.
#
# The check is against the artifacts, not the exit codes. A primary can exit 0
# having written fewer judgments than the panel holds, and only the arithmetic
# against blinded_items.json catches that. Refusals count as accounted-for: the
# provider dropped those cells, not the panel, and judge_coverage.json excludes
# them from EVERY arm.
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
#   PHASE=2 CHECK_ONLY=1 bash scripts/run_iter11_judging.sh # ask, do not spend
#
# CHECK_ONLY reports which targets phase 2 would finalize and exits without
# launching anything: 0 if every eligible target is ready, 1 otherwise. It
# exists so a waiting driver can poll readiness through the same check phase 2
# itself uses instead of keeping a second copy of the rule, which is how the two
# would drift apart.
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
JUDGE_DIR="outputs/iteration_11/judge"
TARGETS="${TARGETS:-qwen35_2b qwen35_4b ministral3_3b phi4_mm}"
# Eight concurrent gateway requests is what was measured clean; 2 judges per
# target, so four targets is the ceiling that measurement supports.
MAX_TARGETS="${MAX_TARGETS:-4}"
PHASE="${PHASE:-1,2}"
JUDGES="${JUDGES:-A B}"
CHECK_ONLY="${CHECK_ONLY:-0}"
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

finalize_ready() {  # model_key -> 0 when both primaries are complete
  # Evidence, not exit codes. Requires, per primary: the split-mode completion
  # marker, its .fingerprint sidecar (without which phase 2 would RE-JUDGE
  # rather than skip), and n_judged + n_refused == n_blinded.
  python3 - "${JUDGE_DIR}/$1" <<'PY'
import json, sys
from pathlib import Path

d = Path(sys.argv[1])
blinded = d / "blinded_items.json"
if not blinded.exists():
    print(f"      {d.name}: no blinded_items.json")
    sys.exit(1)
items = json.loads(blinded.read_text(encoding="utf-8"))
if isinstance(items, dict):
    items = items.get("items", [])
n_blinded = len(items)

problems = []
for j in ("A", "B"):
    done = d / f"llm_labels_judge_{j}.json"
    if not done.exists():
        ck = d / f"llm_labels_judge_{j}.checkpoint.json"
        n = len(json.loads(ck.read_text(encoding="utf-8"))["judgments"]) \
            if ck.exists() else 0
        problems.append(f"judge_{j} incomplete ({n}/{n_blinded})")
        continue
    if not done.with_name(done.name + ".fingerprint").exists():
        problems.append(f"judge_{j} has no .fingerprint sidecar, so phase 2 "
                        "would re-judge it instead of skipping")
    obj = json.loads(done.read_text(encoding="utf-8"))
    n = len(obj) if isinstance(obj, list) else len(obj.get("judgments", []))
    rf = d / f"llm_labels_judge_{j}.refusals.json"
    n_ref = 0
    if rf.exists():
        robj = json.loads(rf.read_text(encoding="utf-8"))
        n_ref = len(robj) if isinstance(robj, list) \
            else len(robj.get("refusals", []))
    if n + n_ref != n_blinded:
        problems.append(f"judge_{j} accounts for {n}+{n_ref} of {n_blinded} "
                        "cells: the panel and the judgments disagree")

for p in problems:
    print(f"      {p}")
sys.exit(1 if problems else 0)
PY
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
  ready=()
  for key in "${eligible[@]}"; do
    if finalize_ready "$key"; then
      ready+=("$key")
    else
      echo "SKIP ${key}: primaries are not both complete; finalizing now "
      echo "     would resume a primary into the checkpoint another process "
      echo "     may still be writing"
      failed=$((failed + 1))
    fi
  done
  if [ "$CHECK_ONLY" = "1" ]; then
    listed="${ready[*]:-none}"
    echo "CHECK_ONLY: would finalize ${#ready[@]} of ${#eligible[@]}: ${listed}"
    if [ "${#ready[@]}" -eq "${#eligible[@]}" ]; then exit 0; else exit 1; fi
  fi
  if [ "${#ready[@]}" -eq 0 ]; then
    echo "no target is ready to finalize" >&2
    exit 1
  fi

  # One panel for all four arms, derived from the eight completed primary
  # outputs before any finalize spends anything. The producer exits non-zero
  # while any arm of the comparison is incomplete, which also makes phase 2
  # all-or-nothing over the group: finalizing three arms and one arm's own
  # exclusions is not a cross-model comparison.
  echo "deriving the cross-arm common panel"
  if ! python scripts/iter11_common_panel.py; then
    echo "FATAL: the cross-arm common panel could not be derived, so no" >&2
    echo "       target is finalized. Each would be restricted to its own" >&2
    echo "       refusals and the four analyses could describe different" >&2
    echo "       families while looking comparable." >&2
    exit 1
  fi

  echo "finalizing: ${ready[*]}"
  pids=(); names=()
  for key in "${ready[@]}"; do
    run_profile "$key" "finalize" "" &
    pids+=($!); names+=("${key}/finalize")
    sleep 5
  done
  echo "launched ${#pids[@]} finalize process(es); waiting"
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "  ${names[$i]}: ok"
    else echo "  ${names[$i]}: FAILED"; failed=$((failed + 1)); fi
  done

  # The gate, over the artifacts phase 2 actually wrote rather than over the
  # union it was handed: identical excluded cells AND identical surviving
  # families in all four. A divergence here means the analyses have to be
  # regenerated on the union the gate reports.
  echo
  echo "=== cross-arm common-panel gate ==="
  if ! python scripts/iter11_common_panel.py --verify; then
    failed=$((failed + 1))
  fi
fi

echo
echo "=== done (${failed} failure(s)) ==="
echo "artifacts: outputs/iteration_11/judge/<model_key>/"
exit "$((failed > 0))"
