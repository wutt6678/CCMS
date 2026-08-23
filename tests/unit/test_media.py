"""Unit tests for media path validation."""

from pathlib import Path

import pytest


def validate_media_path(path: str, base_dir: str = ".") -> bool:
    """Check that a referenced media file exists and is loadable.

    Args:
        path: Relative path to the media file.
        base_dir: Base directory for resolving the path.

    Returns:
        True if the file exists and is non-empty.
    """
    full_path = Path(base_dir) / path
    return full_path.exists() and full_path.stat().st_size > 0


class TestMediaPathValidation:
    """Media path validation must correctly verify file existence."""

    def test_existing_file(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # fake JPEG
        assert validate_media_path("test.jpg", str(tmp_path)) is True

    def test_missing_file(self, tmp_path):
        assert validate_media_path("nonexistent.jpg", str(tmp_path)) is False

    def test_empty_file(self, tmp_path):
        img = tmp_path / "empty.jpg"
        img.write_bytes(b"")
        assert validate_media_path("empty.jpg", str(tmp_path)) is False

    def test_relative_path(self, tmp_path):
        subdir = tmp_path / "media" / "source"
        subdir.mkdir(parents=True)
        img = subdir / "img.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 100)
        assert validate_media_path("media/source/img.png", str(tmp_path)) is True
