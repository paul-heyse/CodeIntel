"""Tests for CLI project discovery helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.cli.project import (
    PROJECT_FILE,
    ProjectConfigError,
    ProjectNotFoundError,
    detect_commit,
    find_project_root,
    load_project_config,
)
from tests._helpers.assertions import expect_equal


def test_find_and_load_project_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Project config should be discovered and parsed."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / PROJECT_FILE).write_text("repo: github.com/demo/repo\n", encoding="utf-8")

    discovered = find_project_root(root)
    expect_equal(discovered, root)

    cfg = load_project_config(root)
    expect_equal(cfg.repo, "github.com/demo/repo")

    monkeypatch.setenv("CODEINTEL_COMMIT", "abc123")
    expect_equal(detect_commit(root), "abc123")


def test_missing_and_invalid_project_file(tmp_path: Path) -> None:
    """Missing or invalid project files raise errors."""
    with pytest.raises(ProjectNotFoundError):
        find_project_root(tmp_path)

    root = tmp_path / "repo2"
    root.mkdir()
    (root / PROJECT_FILE).write_text("", encoding="utf-8")
    with pytest.raises(ProjectConfigError):
        load_project_config(root)
