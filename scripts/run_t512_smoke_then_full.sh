#!/usr/bin/env bash
# Iteration-9 panel chain: 512-token smoke (5 families), then — only
# if the smoke succeeded — the full 20-family panel at the same cap.
set -u
cd "$(dirname "$0")/.." || exit 1
bash scripts/launch_replay_when_gpu_free.sh 24 smoke 512 || exit 1

# Gate the full panel on a COMPLETE smoke: 5 families x 6 variants,
# zero failures, zero truncation.
SMOKE_DIR="outputs/replay_runs/smoke-$(date +%Y-%m-%d)-t512-qwen35-9b"
if [ ! -f "$SMOKE_DIR/replay_outputs.jsonl" ] \
    || [ "$(wc -l < "$SMOKE_DIR/replay_outputs.jsonl")" -ne 30 ] \
    || [ -s "$SMOKE_DIR/replay_failures.jsonl" ] \
    || ! grep -q '^    "n_truncated": 0,' "$SMOKE_DIR/replay_report.json"; then
  echo "[chain] smoke incomplete or truncated — NOT launching the full panel"
  exit 1
fi
bash scripts/launch_replay_when_gpu_free.sh 24 full 512
