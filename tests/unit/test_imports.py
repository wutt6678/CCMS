"""Unit tests for package imports."""


def test_package_import():
    """The top-level package must be importable."""
    import causal_mllm
    assert hasattr(causal_mllm, "__version__")
    assert causal_mllm.__version__ == "0.1.0"


def test_data_subpackage_import():
    """The data subpackage must expose key classes."""
    from causal_mllm.data import (
        CanonicalSourceExample,
        CausalFamily,
        Message,
        SemanticAtom,
        TerminalQuery,
        VariantData,
        VariantName,
    )
    assert CanonicalSourceExample is not None
    assert CausalFamily is not None


def test_seeds_import():
    """The seeds module must be importable."""
    from causal_mllm.seeds import (
        deterministic_family_id,
        set_global_seed,
        sha256_text,
    )
    assert callable(set_global_seed)
    assert callable(deterministic_family_id)
    assert callable(sha256_text)


def test_io_import():
    """The I/O module must be importable."""
    from causal_mllm.data.io import read_jsonl, write_jsonl, load_config
    assert callable(read_jsonl)
    assert callable(write_jsonl)
    assert callable(load_config)


def test_adapter_imports():
    """All adapter modules must be importable."""
    from causal_mllm.adapters.base import DatasetAdapter
    from causal_mllm.adapters.mtmcs import MTMCSAdapter
    from causal_mllm.adapters.cosafe import CoSafeAdapter
    from causal_mllm.adapters.mtid import MTIDAdapter
    assert DatasetAdapter is not None
    assert MTMCSAdapter is not None


def test_cli_imports():
    """All CLI modules must be importable."""
    from causal_mllm.cli import inspect_source
    from causal_mllm.cli import build_families
    from causal_mllm.cli import validate_families
    from causal_mllm.cli import run_inference
    from causal_mllm.cli import evaluate
    assert callable(inspect_source.main)
    assert callable(build_families.main)
