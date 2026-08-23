"""Unit tests for seed utilities."""

from causal_mllm.seeds import (
    config_hash,
    deterministic_family_id,
    get_git_commit,
    set_global_seed,
    sha256_bytes,
    sha256_text,
)


class TestDeterministicFamilyId:
    """Family IDs must be stable across runs for the same inputs."""

    def test_deterministic(self):
        id1 = deterministic_family_id("mtmcs", "example_001", seed=42)
        id2 = deterministic_family_id("mtmcs", "example_001", seed=42)
        assert id1 == id2

    def test_format(self):
        fid = deterministic_family_id("mtmcs", "example_001", seed=42)
        assert fid.startswith("CMST_")
        assert len(fid) == 11  # CMST_ + 6 digits

    def test_different_inputs_different_ids(self):
        id1 = deterministic_family_id("mtmcs", "example_001", seed=42)
        id2 = deterministic_family_id("mtmcs", "example_002", seed=42)
        assert id1 != id2

    def test_different_seeds_different_ids(self):
        id1 = deterministic_family_id("mtmcs", "example_001", seed=42)
        id2 = deterministic_family_id("mtmcs", "example_001", seed=99)
        assert id1 != id2

    def test_different_datasets_different_ids(self):
        id1 = deterministic_family_id("mtmcs", "example_001", seed=42)
        id2 = deterministic_family_id("mtid", "example_001", seed=42)
        assert id1 != id2


class TestSha256:
    """SHA-256 utilities must produce correct, stable hashes."""

    def test_sha256_bytes(self):
        h = sha256_bytes(b"hello")
        assert len(h) == 64  # hex digest length
        assert h == sha256_bytes(b"hello")

    def test_sha256_text_normalizes_whitespace(self):
        h1 = sha256_text("hello   world")
        h2 = sha256_text("hello world")
        assert h1 == h2

    def test_sha256_text_strips(self):
        h1 = sha256_text("  hello  ")
        h2 = sha256_text("hello")
        assert h1 == h2

    def test_sha256_text_different_strings(self):
        h1 = sha256_text("hello")
        h2 = sha256_text("world")
        assert h1 != h2


class TestConfigHash:
    """Config hashing must be deterministic."""

    def test_deterministic(self):
        cfg = {"a": 1, "b": {"c": 2}}
        assert config_hash(cfg) == config_hash(cfg)

    def test_key_order_irrelevant(self):
        cfg1 = {"a": 1, "b": 2}
        cfg2 = {"b": 2, "a": 1}
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_different_configs(self):
        cfg1 = {"a": 1}
        cfg2 = {"a": 2}
        assert config_hash(cfg1) != config_hash(cfg2)


class TestSetGlobalSeed:
    """set_global_seed must not crash."""

    def test_set_seed_no_torch(self):
        # Should not raise even without torch
        set_global_seed(42)


class TestGetGitCommit:
    """get_git_commit should return None or a string."""

    def test_returns_string_or_none(self):
        result = get_git_commit()
        assert result is None or isinstance(result, str)
