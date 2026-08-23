"""Unit tests for CLI --help functionality."""

import subprocess
import sys


class TestCliHelp:
    """All CLI entry points must support --help without errors."""

    def _run_help(self, module_path: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", module_path, "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_inspect_source_help(self):
        result = self._run_help("causal_mllm.cli.inspect_source")
        assert result.returncode == 0
        assert "inspect" in result.stdout.lower() or "dataset" in result.stdout.lower()

    def test_build_families_help(self):
        result = self._run_help("causal_mllm.cli.build_families")
        assert result.returncode == 0
        assert "config" in result.stdout.lower() or "build" in result.stdout.lower()

    def test_validate_families_help(self):
        result = self._run_help("causal_mllm.cli.validate_families")
        assert result.returncode == 0
        assert "input" in result.stdout.lower() or "validate" in result.stdout.lower()

    def test_run_inference_help(self):
        result = self._run_help("causal_mllm.cli.run_inference")
        assert result.returncode == 0
        assert "model" in result.stdout.lower() or "inference" in result.stdout.lower()

    def test_evaluate_help(self):
        result = self._run_help("causal_mllm.cli.evaluate")
        assert result.returncode == 0
        assert "responses" in result.stdout.lower() or "evaluate" in result.stdout.lower()
