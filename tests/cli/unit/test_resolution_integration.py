"""Integration tests for the resolution layer.

Tests verify that the resolution/ package works correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.cli.resolution import ResolutionError, resolve_from_params
from tests._helpers.assertions.expectation_assertions import expect_equal


def test_resolution_from_explicit_params() -> None:
    """Test runtime resolution from explicit params."""
    params: dict[str, object | str] = {
        "repo": "test/repo",
        "commit": "abc123def456789",
        "db_path": str(Path.cwd() / "build" / "test.duckdb"),
        "repo_root": str(Path.cwd()),
    }

    runtime = resolve_from_params(params)
    expect_equal(runtime.repo, "test/repo")
    expect_equal(runtime.commit, "abc123def456789")


def test_resolution_from_project_config(tmp_path: Path) -> None:
    """Test runtime resolution from codeintel.yaml."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    db_dir = project_dir / "build" / "db"
    db_dir.mkdir(parents=True)

    config_file = project_dir / "codeintel.yaml"
    config_file.write_text("repo: test/repo\nstorage:\n  db_path: build/db/codeintel.duckdb\n")

    runtime = resolve_from_params({"project_root": project_dir})
    expect_equal(runtime.repo, "test/repo")
    expect_equal(runtime.db_path, db_dir / "codeintel.duckdb")


def test_resolution_from_env_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test runtime resolution uses CODEINTEL_REPO fallback."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    monkeypatch.setenv("CODEINTEL_REPO", "env/repo")

    runtime = resolve_from_params(
        {
            "repo_root": repo_root,
            "commit": "abc123def456789",
        }
    )
    expect_equal(runtime.repo, "env/repo")


def test_resolution_missing_params_raises_error() -> None:
    """Test resolution fails with missing params."""
    with pytest.raises(ResolutionError):
        resolve_from_params({})
