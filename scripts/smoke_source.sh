#!/usr/bin/env bash
# Inspect all source datasets and write schema reports
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Inspecting source datasets ==="

# Activate conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate midp-qwen35

cd "$PROJECT_DIR"

echo "Inspecting MTMCS-Bench..."
python -m causal_mllm.cli.inspect_source --dataset mtmcs --n 20 --output-dir outputs/schema

echo ""
echo "Inspecting CoSafe..."
python -m causal_mllm.cli.inspect_source --dataset cosafe --n 20 --output-dir outputs/schema

echo ""
echo "Inspecting MTID..."
python -m causal_mllm.cli.inspect_source --dataset mtid --n 20 --output-dir outputs/schema

echo ""
echo "Schema reports saved to outputs/schema/"
ls -la outputs/schema/
