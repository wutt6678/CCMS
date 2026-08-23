# Contributing to CCMS

Thank you for your interest in contributing to the Causal Cross-Modal Safety-State project.

## Development Setup

```bash
git clone https://github.com/wutt6678/CCMS.git
cd CCMS
conda activate <your-env>  # Python 3.10+ with PyTorch + CUDA
pip install -e ".[dev]"
```

## Running Tests

```bash
# Fast unit tests (no network required)
python -m pytest -q -m "not slow and not integration"

# Full test suite
python -m pytest -q
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

## Project Conventions

- **No hard-coded paths**: All paths, models, and parameters go in YAML configs under `configs/`.
- **Typed schemas**: All data structures use `dataclasses` in `src/causal_mllm/data/schemas.py`.
- **Schema validation**: Every record must pass validation before being written. Use `validate_source_example()` / `validate_causal_family()`.
- **Provenance**: Every transformation records its generator, parent variant, and transformation list.
- **Terminal query invariance**: The terminal query must be byte-identical across all variants in a family. This is a hard rejection criterion.

## Pull Request Process

1. Create a feature branch from `main`.
2. Add tests for new functionality.
3. Ensure `python -m pytest -q` passes.
4. Open a pull request with a clear description.

## Reporting Issues

Open a GitHub issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, GPU, etc.)
