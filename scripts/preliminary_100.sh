#!/usr/bin/env bash
# Run the 100-family preliminary experiment
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

eval "$(~/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate midp-qwen35

cd "$PROJECT_DIR"

echo "=== Building 100 causal families ==="
python -m causal_mllm.cli.build_families \
    --config configs/generation/mvp.yaml \
    --max-families 100

echo ""
echo "=== Validating families ==="
python -m causal_mllm.cli.validate_families \
    --input data/families/draft \
    --output data/families/validated

echo ""
echo "=== Running inference ==="
python -m causal_mllm.cli.run_inference \
    --families data/families/validated \
    --model-config configs/models/qwen_mllm.yaml \
    --output outputs/inference/preliminary_100

echo ""
echo "=== Evaluating ==="
python -m causal_mllm.cli.evaluate \
    --families data/families/validated \
    --responses outputs/inference/preliminary_100/responses.jsonl \
    --config configs/evaluation/default.yaml \
    --output outputs/evaluation/preliminary_100

echo "Preliminary 100-family experiment complete."
