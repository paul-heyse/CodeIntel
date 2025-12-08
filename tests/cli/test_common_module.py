"""Tests for shared CLI utilities in _common."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import typer

from tests._helpers.assertions import expect_equal, expect_true

common = importlib.import_module("codeintel.cli.commands._common")


def test_resolve_flag_and_backend_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag resolution and backend config selection behave as expected."""
    expect_true(common.resolve_flag(value=None) is False)
    expect_true(common.resolve_flag(value=True) is True)
    backend = common.build_graph_backend_config(
        common.BackendFlags(use_gpu=True, backend="cpu", strict=True)
    )
    expect_true(backend.use_gpu is True)
    expect_equal(backend.backend, "cpu")
    expect_true(backend.strict is True)

    monkeypatch.setenv("CODEINTEL_GRAPH_EAGER", "true")
    monkeypatch.setenv("CODEINTEL_GRAPH_COMMUNITY_LIMIT", "25")
    monkeypatch.setenv("CODEINTEL_GRAPH_VALIDATION_STRICT", "0")
    flags = common.build_graph_feature_flags_from_env()
    expect_true(flags.eager_hydration is True)
    expect_equal(flags.community_detection_limit, 25)
    expect_true(flags.validation_strict is False)


def test_build_config_from_options_creates_paths(tmp_path: Path) -> None:
    """Explicit options produce a valid CodeIntelConfig and build paths."""
    repo_root = tmp_path / "repo"
    db_path = tmp_path / "build" / "db" / "codeintel.duckdb"
    build_dir = tmp_path / "build"
    repo_root.mkdir()
    paths_cfg = common.CliPathsInput(
        repo_root=repo_root,
        build_dir=build_dir,
        db_path=db_path,
        document_output_dir=None,
    )
    cfg = common.build_config_from_options(
        repo="demo/repo",
        commit="deadbeef",
        paths_cfg=paths_cfg,
        backend=common.BackendFlags(),
    )
    expect_equal(cfg.repo.repo, "demo/repo")
    expect_equal(cfg.repo.commit, "deadbeef")
    expect_equal(cfg.paths.db_path, db_path)
    expect_equal(cfg.build_paths.db_path, db_path)


def test_build_runtime_or_exit_fallback_and_missing(tmp_path: Path) -> None:
    """Fallback options succeed; missing options exit with code 1."""
    repo_root = tmp_path / "repo"
    build_dir = tmp_path / "build"
    db_path = build_dir / "db" / "codeintel.duckdb"
    build_dir.mkdir(parents=True, exist_ok=True)
    repo_root.mkdir(parents=True, exist_ok=True)

    runtime = common.build_runtime_or_exit(
        common.RuntimeCliOptions(
            project_root=None,
            repo="demo/repo",
            commit="deadbeef",
            db_path=db_path,
            build_dir=build_dir,
            repo_root=repo_root,
        )
    )
    expect_equal(runtime.snapshot.repo, "demo/repo")
    expect_equal(runtime.snapshot.commit, "deadbeef")
    runtime.gateway.close()

    with pytest.raises(typer.Exit) as excinfo:
        common.build_runtime_or_exit(common.RuntimeCliOptions(project_root=tmp_path))
    expect_equal(excinfo.value.exit_code, 1)
