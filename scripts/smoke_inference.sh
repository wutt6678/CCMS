#!/usr/bin/env bash
# Run inference smoke test on validated families
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

eval "$(~/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate midp-qwen35

cd "$PROJECT_DIR"

echo "=== Running inference smoke test ==="
python -m causal_mllm.cli.run_inference \
    --families data/families/validated \
    --model-config configs/models/qwen_mllm.yaml \
    --output outputs/inference/smoke

echo "Done."
