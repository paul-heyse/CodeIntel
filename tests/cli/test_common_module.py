"""Tests for shared CLI utilities in common_handlers and cyclopts_common."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from codeintel.cli.cli_errors import ValidationError, runtime_required
from codeintel.cli.common_handlers import (
    BackendFlags,
    OutputFormat,
    RuntimeCliOptions,
    build_config_from_options,
    build_graph_backend_config,
    build_graph_feature_flags_from_env,
    build_runtime_from_cli,
    resolve_flag,
)
from codeintel.cli.cyclopts_common import (
    OutputFormatCLI,
    RuntimeCLI,
    make_handler_context,
)
from codeintel.config.models import CliPathsInput
from tests._helpers.assertions import expect_equal, expect_true


def test_resolve_flag_and_backend_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag resolution and backend config selection behave as expected."""
    expect_true(resolve_flag(value=None) is False)
    expect_true(resolve_flag(value=True) is True)
    backend = build_graph_backend_config(BackendFlags(use_gpu=True, backend="cpu", strict=True))
    expect_true(backend.use_gpu is True)
    expect_equal(backend.backend, "cpu")
    expect_true(backend.strict is True)

    monkeypatch.setenv("CODEINTEL_GRAPH_EAGER", "true")
    monkeypatch.setenv("CODEINTEL_GRAPH_COMMUNITY_LIMIT", "25")
    monkeypatch.setenv("CODEINTEL_GRAPH_VALIDATION_STRICT", "0")
    flags = build_graph_feature_flags_from_env()
    expect_true(flags.eager_hydration is True)
    expect_equal(flags.community_detection_limit, 25)
    expect_true(flags.validation_strict is False)


def test_build_config_from_options_creates_paths(tmp_path: Path) -> None:
    """Explicit options produce a valid CodeIntelConfig and build paths."""
    repo_root = tmp_path / "repo"
    db_path = tmp_path / "build" / "db" / "codeintel.duckdb"
    build_dir = tmp_path / "build"
    repo_root.mkdir()
    paths_cfg = CliPathsInput(
        repo_root=repo_root,
        build_dir=build_dir,
        db_path=db_path,
        document_output_dir=None,
    )
    cfg = build_config_from_options(
        repo="demo/repo",
        commit="deadbeef",
        paths_cfg=paths_cfg,
        backend=BackendFlags(),
    )
    expect_equal(cfg.repo.repo, "demo/repo")
    expect_equal(cfg.repo.commit, "deadbeef")
    expect_equal(cfg.paths.db_path, db_path)
    expect_equal(cfg.build_paths.db_path, db_path)


def test_build_runtime_from_cli_fallback_and_missing(tmp_path: Path) -> None:
    """Fallback options succeed; missing options raise ValidationError."""
    repo_root = tmp_path / "repo"
    build_dir = tmp_path / "build"
    db_path = build_dir / "db" / "codeintel.duckdb"
    build_dir.mkdir(parents=True, exist_ok=True)
    repo_root.mkdir(parents=True, exist_ok=True)

    runtime = build_runtime_from_cli(
        RuntimeCliOptions(
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

    with pytest.raises(ValidationError):
        build_runtime_from_cli(RuntimeCliOptions(project_root=tmp_path))


# ---------------------------------------------------------------------------
# Tests for cyclopts_common field helpers and make_handler_context
# ---------------------------------------------------------------------------


@dataclass
class _TestRuntimeCli:
    """Test dataclass using RuntimeCLI for field helper verification."""

    runtime: RuntimeCLI | None = None


@dataclass
class _TestOutputCli:
    """Test dataclass using OutputFormatCLI for field helper verification."""

    output: OutputFormatCLI | None = None


def test_runtime_field_returns_none_by_default() -> None:
    """Verify RuntimeCLI field defaults to None (Cyclopts pattern)."""
    instance = _TestRuntimeCli()
    # In Cyclopts pattern, nested dataclass fields default to None
    # and Cyclopts instantiates them during parsing
    expect_true(instance.runtime is None)


def test_output_field_returns_none_by_default() -> None:
    """Verify OutputFormatCLI field defaults to None (Cyclopts pattern)."""
    instance = _TestOutputCli()
    # In Cyclopts pattern, nested dataclass fields default to None
    expect_true(instance.output is None)


def test_make_handler_context_extracts_all_fields() -> None:
    """Verify make_handler_context returns runtime_opts, verbose, output_format."""
    runtime_cli = RuntimeCLI(
        project_root=Path("/test"),
        repo="org/repo",
        commit="abc123",
        verbose=2,
    )
    output_cli = OutputFormatCLI(output_format=OutputFormat.JSON, json=False)

    runtime_opts, verbose, output_format = make_handler_context(
        runtime_cli, output_cli, default_output=OutputFormat.TEXT
    )

    expect_equal(runtime_opts.project_root, Path("/test"))
    expect_equal(runtime_opts.repo, "org/repo")
    expect_equal(runtime_opts.commit, "abc123")
    expect_equal(verbose, 2)
    expect_equal(output_format, OutputFormat.JSON)


def test_make_handler_context_json_flag_overrides_format() -> None:
    """Verify json flag takes precedence in make_handler_context."""
    runtime_cli = RuntimeCLI()
    output_cli = OutputFormatCLI(output_format=OutputFormat.TEXT, json=True)

    _, _, output_format = make_handler_context(
        runtime_cli, output_cli, default_output=OutputFormat.TEXT
    )

    expect_equal(output_format, OutputFormat.JSON)


def test_runtime_required_raises_for_missing_fields() -> None:
    """Verify runtime_required raises ValidationError for missing fields."""
    cli = RuntimeCLI(repo=None, commit=None, db_path=None)

    with pytest.raises(ValidationError, match="--repo"):
        runtime_required(cli, "test operation", require_repo=True, require_commit=False)

    with pytest.raises(ValidationError, match="--commit"):
        runtime_required(cli, "test operation", require_repo=False, require_commit=True)

    with pytest.raises(ValidationError, match="--db-path"):
        runtime_required(
            cli, "test operation", require_repo=False, require_commit=False, require_db_path=True
        )

    cli_with_values = RuntimeCLI(repo="org/repo", commit="abc", db_path=Path("test.db"))
    runtime_required(cli_with_values, "test op", require_repo=True, require_commit=True)


def test_runtime_required_includes_all_missing_in_message() -> None:
    """Verify multiple missing fields are listed in the error message."""
    cli = RuntimeCLI(repo=None, commit=None, db_path=None)

    with pytest.raises(ValidationError, match=r"--repo.*--commit") as exc_info:
        runtime_required(
            cli, "history command", require_repo=True, require_commit=True, require_db_path=False
        )

    expect_true("history command requires" in str(exc_info.value))
