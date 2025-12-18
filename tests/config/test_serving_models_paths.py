"""Tests for serving model path normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.serving.config import ServingConfig, normalize_optional_path

if TYPE_CHECKING:
    from pathlib import Path


def test_normalize_optional_path_handles_none() -> None:
    """Helper should return None when given None."""
    if normalize_optional_path(None) is not None:
        pytest.fail("Expected None to remain None")


def test_normalize_optional_path_resolves_strings(tmp_path: Path) -> None:
    """Helper should resolve string paths to absolute Path objects."""
    rel_path = tmp_path / "db.duckdb"
    normalized = normalize_optional_path(str(rel_path))
    if normalized != rel_path.resolve():
        pytest.fail("Expected string path to resolve to Path")


def test_serving_config_defaults_db_path_when_missing(tmp_path: Path) -> None:
    """ServingConfig should derive db_path when mode=local_db and db_path is None."""
    cfg = ServingConfig(repo_root=tmp_path)
    expected = (tmp_path / "build" / "db" / "codeintel.duckdb").resolve()
    if cfg.db_path != expected:
        pytest.fail("db_path should default under repo_root/build/db")


def test_serving_config_normalizes_db_path_when_provided(tmp_path: Path) -> None:
    """ServingConfig should normalize provided db_path values."""
    raw = tmp_path / "custom.duckdb"
    cfg = ServingConfig(repo_root=tmp_path, db_path=raw)
    if cfg.db_path != raw.resolve():
        pytest.fail("db_path should be normalized to absolute Path")
