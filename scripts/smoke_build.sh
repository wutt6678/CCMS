#!/usr/bin/env bash
# Build a small number of causal families (smoke test)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

eval "$(~/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate midp-qwen35

cd "$PROJECT_DIR"

echo "=== Building 5 causal families (smoke test) ==="
python -m causal_mllm.cli.build_families \
    --config configs/generation/mvp.yaml \
    --max-families 5

echo "Done."
