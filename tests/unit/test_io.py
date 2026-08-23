"""Unit tests for JSONL read/write helpers and config loading."""

import json
from pathlib import Path

import pytest

from causal_mllm.data.io import (
    append_jsonl,
    load_config,
    read_jsonl,
    read_jsonl_iter,
    save_config,
    write_jsonl,
)


class TestJsonlRoundtrip:
    """JSONL read/write must be lossless for valid records."""

    def test_write_and_read(self, tmp_path):
        path = tmp_path / "test.jsonl"
        records = [
            {"id": 1, "text": "hello"},
            {"id": 2, "text": "world", "nested": {"key": "val"}},
        ]
        n = write_jsonl(path, records)
        assert n == 2
        loaded = read_jsonl(path)
        assert loaded == records

    def test_read_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert read_jsonl(path) == []

    def test_skip_blank_lines(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        path.write_text('{"a":1}\n\n{"b":2}\n\n')
        loaded = read_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0] == {"a": 1}
        assert loaded[1] == {"b": 2}

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"valid":true}\n{invalid json}\n')
        with pytest.raises(json.JSONDecodeError):
            read_jsonl(path)

    def test_append_jsonl(self, tmp_path):
        path = tmp_path / "append.jsonl"
        write_jsonl(path, [{"x": 1}])
        append_jsonl(path, {"x": 2})
        loaded = read_jsonl(path)
        assert len(loaded) == 2
        assert loaded[1] == {"x": 2}

    def test_write_jsonl_append_mode(self, tmp_path):
        path = tmp_path / "append2.jsonl"
        write_jsonl(path, [{"a": 1}])
        write_jsonl(path, [{"b": 2}], append=True)
        loaded = read_jsonl(path)
        assert len(loaded) == 2

    def test_iter_jsonl(self, tmp_path):
        path = tmp_path / "iter.jsonl"
        records = [{"i": i} for i in range(10)]
        write_jsonl(path, records)
        loaded = list(read_jsonl_iter(path))
        assert loaded == records

    def test_unicode_roundtrip(self, tmp_path):
        path = tmp_path / "unicode.jsonl"
        records = [{"text": "Hello 世界 🌍"}, {"text": "café résumé"}]
        write_jsonl(path, records)
        loaded = read_jsonl(path)
        assert loaded == records

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_jsonl("/nonexistent/path.jsonl")


class TestConfigLoading:
    """YAML config loading must work correctly."""

    def test_load_valid_config(self, sample_config_file):
        config = load_config(sample_config_file)
        assert config["experiment_name"] == "test_run"
        assert config["seed"] == 42
        assert config["source"]["dataset"] == "mtmcs"
        assert len(config["family"]["variants"]) == 6

    def test_load_missing_config(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_save_and_load_roundtrip(self, tmp_path):
        config = {
            "experiment": "test",
            "seed": 123,
            "nested": {"key": "value"},
        }
        path = tmp_path / "saved.yaml"
        save_config(config, path)
        loaded = load_config(path)
        assert loaded == config

    def test_load_empty_config(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        config = load_config(path)
        assert config == {}
