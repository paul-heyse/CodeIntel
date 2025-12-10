"""Tests for shared CLI utilities in config/service and cyclopts_common."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from codeintel.cli.cli_errors import ValidationError, runtime_required
from codeintel.cli.cli_types import BackendFlags, OutputFormat
from codeintel.cli.config import (
    build_config_from_options,
    build_graph_backend_config,
    build_graph_feature_flags_from_env,
)
from codeintel.cli.cyclopts_common import (
    OutputFormatCLI,
    RuntimeCLI,
    get_output_format,
    get_verbose,
    resolve_output_format,
)
from codeintel.config.models import CliPathsInput
from tests._helpers.assertions import expect_equal, expect_true


def _resolve_flag(value: object) -> bool:
    """Resolve an optional flag value to a boolean (local helper for test).

    Parameters
    ----------
    value
        Flag value (may be None, bool, or other).

    Returns
    -------
    bool
        True if value is truthy and not None, False otherwise.
    """
    if value is None:
        return False
    return bool(value)


def test_resolve_flag_and_backend_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag resolution and backend config selection behave as expected."""
    expect_true(_resolve_flag(value=None) is False)
    expect_true(_resolve_flag(value=True) is True)
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


# ---------------------------------------------------------------------------
# Tests for cyclopts_common field helpers and output format resolution
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


def test_get_verbose_extracts_verbose_level() -> None:
    """Verify get_verbose extracts verbosity count from RuntimeCLI."""
    cli = RuntimeCLI(verbose=2)
    expect_equal(get_verbose(cli), 2)

    cli_default = RuntimeCLI()
    expect_equal(get_verbose(cli_default), 0)


def test_get_output_format_resolves_correctly() -> None:
    """Verify get_output_format resolves format with correct precedence."""
    # Default format
    output_cli = OutputFormatCLI()
    expect_equal(get_output_format(output_cli), OutputFormat.TEXT)

    # Explicit JSON format
    output_cli_json = OutputFormatCLI(output_format=OutputFormat.JSON)
    expect_equal(get_output_format(output_cli_json), OutputFormat.JSON)

    # JSON flag overrides explicit format
    output_cli_flag = OutputFormatCLI(output_format=OutputFormat.TEXT, json=True)
    expect_equal(get_output_format(output_cli_flag), OutputFormat.JSON)


def test_resolve_output_format_precedence() -> None:
    """Verify resolve_output_format handles precedence correctly."""
    # JSON flag takes highest precedence
    expect_equal(
        resolve_output_format(json_flag=True, explicit=OutputFormat.TEXT, default=OutputFormat.TEXT),
        OutputFormat.JSON,
    )

    # Explicit format takes precedence over default
    expect_equal(
        resolve_output_format(json_flag=False, explicit=OutputFormat.JSON, default=OutputFormat.TEXT),
        OutputFormat.JSON,
    )

    # Default is used when no override
    expect_equal(
        resolve_output_format(json_flag=False, explicit=None, default=OutputFormat.TEXT),
        OutputFormat.TEXT,
    )


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
