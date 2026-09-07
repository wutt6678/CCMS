#!/usr/bin/env bash
# Iteration 11.4 — regenerate the phi4_mm technical preflight artifact.
#
# WHY ONE TARGET HAS ITS OWN LANE
# The other three preflight artifacts were regenerated at f5f7db1 (2026-09-05
# 13:21-13:24). phi4_mm's still names afebf85 (10:43), because that round died
# with torch.OutOfMemoryError on cuda:3: the checkpoint is 5,574,460,384 BF16
# parameters over 10.60 GiB of shards, so it needs the largest headroom of the
# four and it is the only one that could not be squeezed in. The older artifact
# is a valid PASS — same ccms-iter11 environment, same dependency-lock identity
# (pip_freeze_sha256 c03a5800...), clean tree, immutable revision — but it was
# written by an older producer, so alone among the four it does not carry the
# editable-install audit (environment.third_party_editable_installs and
# lock.dependency_lock.editable_installs). That is the field set the other
# three refuse to run without.
#
# Nothing scientific waits on this. The preflight fails closed when prompt +
# cap would cross the vendor's 4096-token LongRoPE switch and the family it
# smokes peaks at 709 + 1536, so its smoke never reaches the code shim 9
# repairs; the certifications that carry the panel bind their own later commits
# (11.5 eligibility f52db5a, 11.6 confirmatory 67aa3a5, which contains the
# shim). See the 11.4 section of README.md and
# tests/unit/test_iter11_phi4.py::TestPanelCrossedTheLongRopeBoundary.
#
# MEMORY POLICY: WAIT, DO NOT SQUEEZE
# 14000 MiB is the same calibrated requirement run_iter11_confirmatory.sh uses
# for this target (10.60 GiB of shards + ~0.8 GiB of CUDA context and
# activations + ~25% margin). On a shared machine the check and the allocation
# are not atomic, and an 11 GiB resident load beside another user's job risks
# THEIR run rather than ours — an OOM here costs minutes, an OOM there costs
# hours. So this lane polls every slot, takes the emptiest one that clears the
# threshold, and DEFERS (exit 3, having started nothing) when no slot does.
#
# The tree must be clean when this starts: the producer aborts on a dirty tree
# and could not reach status PASS anyway, since evidence written from a dirty
# tree cannot be reconstructed from the code_commit it records.
#
# --update-lock IS passed, and the first attempt at this lane showed why. It
# was originally omitted on the reasoning that phi4_mm's revision is already
# pinned, so there was nothing to update. That reasoning was wrong about the
# producer: report["lock"] is initialised to None and populated ONLY inside the
# `if args.update_lock` branch, so omitting the flag wrote a PASS artifact with
# "lock": null -- no dependency-lock path, no pip_freeze_sha256, no
# editable-install audit in the lock block. It would have been the only one of
# the four preflight artifacts with no dependency binding at all, which is the
# opposite of the gap this lane exists to close. That attempt (2026-09-07
# 15:13-15:16, exit 0, status PASS, cuda:0) is not committed.
#
# --force-lock is NOT passed, and that is what keeps this safe. update_lock()
# raises when a key is already locked to a DIFFERENT revision unless
# allow_change is explicit, so re-pinning still has to be deliberate; here the
# locked revision (93f923e1...) equals the one the smoke resolves, so the call
# cannot raise and cannot move anything. It refreshes phi4_mm's own entry and
# rewrites the dependency_lock block, whose five LOCK_IDENTITY_FIELDS are
# unchanged -- pip_freeze_sha256 c03a5800ca95b020, n_packages 100,
# pyproject_sha256, python_version 3.10.20, excluded_self_distributions
# [causal-mllm] -- so no confirmatory run's resolved_run_fingerprint moves.
# `executable` does differ between interpreters (bin/python vs bin/python3) and
# is recorded, but registry.py documents it as operational metadata outside the
# hashed identity, which is why the digest above is stable.
#
# This script commits nothing. Inspect the artifact and the diff, re-run the
# phi4 tests, then commit.
#
# Usage:
#   bash scripts/iter11_regenerate_phi4_preflight.sh
#   MIN_FREE_MIB=16000 bash scripts/iter11_regenerate_phi4_preflight.sh
#   MAX_WAIT_SECONDS=0 bash scripts/iter11_regenerate_phi4_preflight.sh  # try once
set -uo pipefail

cd /scratch/wutiantong/CCMS
source /scratch/wutiantong/miniconda3/etc/profile.d/conda.sh
conda activate ccms-iter11

ART="outputs/iteration_11/preflight/phi4_mm/preflight.json"
MIN_FREE_MIB="${MIN_FREE_MIB:-14000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-86400}"

echo "=== phi4_mm preflight regeneration ==="
echo "HEAD $(git rev-parse --short HEAD)"
echo "requirement  ${MIN_FREE_MIB} MiB free on one slot"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "ABORT: tracked modifications present, so the producer could not" >&2
  echo "       reach PASS and the artifact would not be reconstructible:" >&2
  git status --porcelain --untracked-files=no >&2
  exit 2
fi
echo "clean tree (tracked): ok"
echo "editable installs in this environment:"
python -m pip freeze | grep -E '^-e ' || echo "  (none)"

echo "committed artifact before:"
python3 -c "
import json
d = json.load(open('$ART'))
env = d.get('environment', {})
print('   status', d['status'], 'code_commit', d['code_commit'][:12],
      'ts', d['timestamp'], 'device', d['device'])
print('   carries the editable-install audit:',
      'third_party_editable_installs' in env)
"

pick_slot() {  # emptiest slot clearing the threshold, or empty
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -rn \
    | awk -F, -v need="$MIN_FREE_MIB" \
        '{gsub(/ /,"",$2); if ($2+0 >= need+0) {print $1; exit}}'
}

waited=0
idx="$(pick_slot)"
while [ -z "$idx" ]; do
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
            --format=csv,noheader | sed 's/^/   /'
  if [ "$waited" -ge "$MAX_WAIT_SECONDS" ]; then
    echo "DEFER: no slot reached ${MIN_FREE_MIB} MiB free in ${waited}s;" >&2
    echo "       nothing was started and nothing was written." >&2
    exit 3
  fi
  echo "  no slot has ${MIN_FREE_MIB} MiB free; polling in ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
  waited=$((waited + POLL_SECONDS))
  idx="$(pick_slot)"
done
echo "slot: cuda:${idx} ($(nvidia-smi --query-gpu=memory.free \
      --format=csv,noheader,nounits -i "${idx}" | tr -d ' ') MiB free)"

echo "=== running the preflight $(date -Is) ==="
PYTHONPATH=src python scripts/iter11_model_preflight.py \
  --model-key phi4_mm --device "cuda:${idx}" --gpu-smoke --update-lock
code=$?
echo "---- exit ${code} $(date -Is) ----"

echo "artifact after:"
python3 -c "
import json
d = json.load(open('$ART'))
env = d.get('environment', {})
lock = (d.get('lock') or {}).get('dependency_lock') or {}
print('   status', d['status'], 'code_commit', d['code_commit'][:12],
      'ts', d['timestamp'], 'device', d['device'])
print('   problems', d['problems'])
print('   environment.third_party_editable_installs:',
      env.get('third_party_editable_installs'))
print('   lock.dependency_lock.editable_installs:', lock.get('editable_installs'))
print('   smoke:', [(e['variant'], e['deterministic']) for e in d['gpu_smoke']])
"

echo "=== git diff --stat (this script commits nothing) ==="
git diff --stat -- "$ART" "outputs/iteration_11/preflight/resolved_models.lock.yaml"
exit "$code"
