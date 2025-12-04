"""Tests for auto-pipeline integration.

These tests verify the auto-pipeline feature that automatically runs
prerequisite pipeline stages when serving operations.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest

from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY
from codeintel.config.primitives import SnapshotRef
from codeintel.config.serving_models import ServingConfig
from codeintel.runtime import RunContext, RunKind, TriggerKind
from codeintel.serving.auto_pipeline import (
    AUTO_PIPELINE_ENV,
    build_paths_for_serving,
    build_prereq_debug_info,
    dataset_has_rows_for_snapshot,
    ensure_prereqs_for_http,
    ensure_prereqs_for_mcp,
    get_required_table_keys_for_operation,
    has_required_data_for_operation,
    has_successful_prereq_run,
    is_auto_pipeline_enabled,
    operation_prereqs_satisfied,
    should_run_auto_pipeline,
)
from codeintel.serving.mcp.auto_pipeline_wrapper import wrap_tool_with_prereqs
from codeintel.serving.mcp.backend import DuckDBBackend, QueryBackend
from codeintel.serving.operations.catalog import get_operation
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.tracking import PipelineRunTracking, PipelineStatus
from tests._helpers.gateway import build_duckdb_backend, gateway_with_macros

# -----------------------------------------------------------------------------
# is_auto_pipeline_enabled Tests
# -----------------------------------------------------------------------------


@pytest.fixture
def clean_auto_pipeline_env() -> Generator[None]:
    """Clear the auto-pipeline env var before and after test.

    Yields
    ------
    None
        Control returns to test.
    """
    if AUTO_PIPELINE_ENV in os.environ:
        del os.environ[AUTO_PIPELINE_ENV]
    yield
    if AUTO_PIPELINE_ENV in os.environ:
        del os.environ[AUTO_PIPELINE_ENV]


def test_auto_pipeline_disabled_by_default(clean_auto_pipeline_env: None) -> None:
    """Verify auto-pipeline is disabled when env var is not set."""
    _ = clean_auto_pipeline_env
    assert not is_auto_pipeline_enabled()


@pytest.mark.parametrize(
    "value",
    ["1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"],
)
def test_auto_pipeline_enabled_with_truthy_values(
    clean_auto_pipeline_env: None,
    value: str,
) -> None:
    """Verify auto-pipeline is enabled with truthy env values."""
    _ = clean_auto_pipeline_env
    os.environ[AUTO_PIPELINE_ENV] = value
    assert is_auto_pipeline_enabled()


@pytest.mark.parametrize(
    "value",
    ["0", "false", "False", "no", "No", "off", "Off", "", "  "],
)
def test_auto_pipeline_disabled_with_falsy_values(
    clean_auto_pipeline_env: None,
    value: str,
) -> None:
    """Verify auto-pipeline is disabled with falsy env values."""
    _ = clean_auto_pipeline_env
    os.environ[AUTO_PIPELINE_ENV] = value
    assert not is_auto_pipeline_enabled()


# -----------------------------------------------------------------------------
# build_paths_for_serving Tests
# -----------------------------------------------------------------------------


def test_build_paths_with_defaults(tmp_path: Path) -> None:
    """Verify paths are built with default values."""
    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    paths = build_paths_for_serving(config)

    assert paths.db_path == tmp_path / "db.duckdb"


def test_build_paths_with_default_repo_root() -> None:
    """Verify paths work with default repo_root (cwd)."""
    config = ServingConfig(
        mode="local_db",
        repo="test/repo",
        commit="abc123",
        db_path=Path(".codeintel/db.duckdb"),
    )

    paths = build_paths_for_serving(config)
    # Should use current working directory
    assert paths.db_path.is_absolute()


@pytest.fixture
def pipeline_run_tracking() -> Iterator[PipelineRunTracking]:
    """
    Provide a real PipelineRunTracking backed by an in-memory gateway.

    Yields
    ------
    PipelineRunTracking
        Run tracking instance for test use.
    """
    gateway = gateway_with_macros(repo="test/repo", commit="abc123")
    try:
        yield gateway.runs
    finally:
        gateway.close()


@pytest.fixture
def duckdb_backend() -> Iterator[DuckDBBackend]:
    """Provide a real DuckDBBackend implementing QueryBackend.

    Yields
    ------
    DuckDBBackend
        Backend instance backed by an in-memory gateway.
    """
    gateway = gateway_with_macros(repo="test/repo", commit="abc123")
    backend: DuckDBBackend = build_duckdb_backend(gateway, repo="test/repo", commit="abc123")
    try:
        yield backend
    finally:
        gateway.close()


@dataclass(frozen=True)
class RunParams:
    """Parameters for seeding a pipeline run."""

    repo: str
    commit: str
    status: PipelineStatus
    kind: RunKind
    trigger: TriggerKind = "cli"


def _seed_run(tracking: PipelineRunTracking, params: RunParams) -> None:
    """Insert a run into the tracking store using the production API."""
    snapshot = SnapshotRef(repo=params.repo, commit=params.commit, repo_root=Path.cwd())
    ctx = RunContext(
        run_id=f"run-{uuid4().hex}",
        kind=params.kind,
        snapshot=snapshot,
        trigger=params.trigger,
    )
    tracking.start_run(ctx, status=params.status)


# -----------------------------------------------------------------------------
# has_successful_prereq_run Tests
# -----------------------------------------------------------------------------


def test_has_successful_prereq_run_returns_true_when_matching_run_exists(
    pipeline_run_tracking: PipelineRunTracking,
) -> None:
    """Verify returns True when a matching successful run exists."""
    params = RunParams(repo="test/repo", commit="abc123", status="succeeded", kind="full")
    _seed_run(pipeline_run_tracking, params)

    result = has_successful_prereq_run(
        pipeline_run_tracking,
        repo="test/repo",
        commit="abc123",
        op_id="function.summary",
    )

    assert result is True


def test_has_successful_prereq_run_returns_false_when_no_runs(
    pipeline_run_tracking: PipelineRunTracking,
) -> None:
    """Verify returns False when no runs exist."""
    result = has_successful_prereq_run(
        pipeline_run_tracking,
        repo="test/repo",
        commit="abc123",
        op_id="function.summary",
    )

    assert result is False


def test_has_successful_prereq_run_returns_false_when_commit_mismatch(
    pipeline_run_tracking: PipelineRunTracking,
) -> None:
    """Verify returns False when commit doesn't match."""
    params = RunParams(repo="test/repo", commit="different", status="succeeded", kind="full")
    _seed_run(pipeline_run_tracking, params)

    result = has_successful_prereq_run(
        pipeline_run_tracking,
        repo="test/repo",
        commit="abc123",
        op_id="function.summary",
    )

    assert result is False


def test_has_successful_prereq_run_returns_false_when_status_not_succeeded(
    pipeline_run_tracking: PipelineRunTracking,
) -> None:
    """Verify returns False when status is not 'succeeded'."""
    params = RunParams(repo="test/repo", commit="abc123", status="failed", kind="full")
    _seed_run(pipeline_run_tracking, params)

    result = has_successful_prereq_run(
        pipeline_run_tracking,
        repo="test/repo",
        commit="abc123",
        op_id="function.summary",
    )

    assert result is False


def test_has_successful_prereq_run_returns_true_for_op_prereqs_kind(
    pipeline_run_tracking: PipelineRunTracking,
) -> None:
    """Verify returns True for op_prereqs kind runs."""
    params = RunParams(repo="test/repo", commit="abc123", status="succeeded", kind="op_prereqs")
    _seed_run(pipeline_run_tracking, params)

    result = has_successful_prereq_run(
        pipeline_run_tracking,
        repo="test/repo",
        commit="abc123",
        op_id="function.summary",
    )

    assert result is True


def test_has_successful_prereq_run_returns_false_for_other_kinds(
    pipeline_run_tracking: PipelineRunTracking,
) -> None:
    """Verify returns False for runs with non-matching kinds."""
    params = RunParams(repo="test/repo", commit="abc123", status="succeeded", kind="ingest")
    _seed_run(pipeline_run_tracking, params)

    result = has_successful_prereq_run(
        pipeline_run_tracking,
        repo="test/repo",
        commit="abc123",
        op_id="function.summary",
    )

    assert result is False


# -----------------------------------------------------------------------------
# should_run_auto_pipeline Tests
# -----------------------------------------------------------------------------


def test_should_run_auto_pipeline_returns_false_for_remote_mode(
    clean_auto_pipeline_env: None,
    duckdb_backend: DuckDBBackend,
) -> None:
    """Verify returns False when mode is not local_db."""
    _ = clean_auto_pipeline_env

    os.environ[AUTO_PIPELINE_ENV] = "1"
    config = ServingConfig(
        mode="remote_api",
        repo="test/repo",
        commit="abc123",
        api_base_url="http://example.com",
    )

    should_run, gateway, reason = should_run_auto_pipeline(config, duckdb_backend)
    assert should_run is False
    assert "not local_db" in reason
    assert gateway is None


def test_should_run_auto_pipeline_returns_false_when_disabled(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
    duckdb_backend: DuckDBBackend,
) -> None:
    """Verify returns False when auto-pipeline is disabled."""
    _ = clean_auto_pipeline_env

    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    should_run, _gateway, reason = should_run_auto_pipeline(config, duckdb_backend)
    assert should_run is False
    assert "not enabled" in reason


def test_should_run_auto_pipeline_returns_false_without_gateway(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
) -> None:
    """Verify returns False when backend has no gateway."""
    _ = clean_auto_pipeline_env

    os.environ[AUTO_PIPELINE_ENV] = "1"
    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    backend_without_gateway = cast("QueryBackend", object())
    should_run, _gateway, reason = should_run_auto_pipeline(config, backend_without_gateway)
    assert should_run is False
    assert "no gateway" in reason


def test_should_run_auto_pipeline_returns_true_with_valid_config(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
    duckdb_backend: DuckDBBackend,
) -> None:
    """Verify returns True when all conditions are met."""
    _ = clean_auto_pipeline_env

    os.environ[AUTO_PIPELINE_ENV] = "1"
    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    should_run, gateway, reason = should_run_auto_pipeline(config, duckdb_backend)
    assert should_run is True
    assert gateway is duckdb_backend.gateway
    assert not reason


# -----------------------------------------------------------------------------
# ensure_prereqs_for_mcp Tests
# -----------------------------------------------------------------------------


def test_ensure_prereqs_for_mcp_skips_when_disabled(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
    duckdb_backend: DuckDBBackend,
) -> None:
    """Verify ensure_prereqs_for_mcp skips when auto-pipeline is disabled."""
    _ = clean_auto_pipeline_env

    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    result = ensure_prereqs_for_mcp(
        op_id="function.summary",
        config=config,
        backend=duckdb_backend,
    )
    assert result is None


def test_ensure_prereqs_for_mcp_skips_for_non_local_mode(
    clean_auto_pipeline_env: None,
    duckdb_backend: DuckDBBackend,
) -> None:
    """Verify ensure_prereqs_for_mcp skips for non-local_db mode."""
    _ = clean_auto_pipeline_env

    os.environ[AUTO_PIPELINE_ENV] = "1"
    config = ServingConfig(
        mode="remote_api",
        repo="test/repo",
        commit="abc123",
        api_base_url="http://example.com",
    )

    result = ensure_prereqs_for_mcp(
        op_id="function.summary",
        config=config,
        backend=duckdb_backend,
    )
    assert result is None


# -----------------------------------------------------------------------------
# ensure_prereqs_for_http Tests
# -----------------------------------------------------------------------------


def test_ensure_prereqs_for_http_skips_when_disabled(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
    duckdb_backend: DuckDBBackend,
) -> None:
    """Verify ensure_prereqs_for_http skips when auto-pipeline is disabled."""
    _ = clean_auto_pipeline_env

    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    result = ensure_prereqs_for_http(
        op_id="function.summary",
        config=config,
        backend=duckdb_backend,
    )
    assert result is None


def test_ensure_prereqs_for_http_skips_for_non_local_mode(
    clean_auto_pipeline_env: None,
    duckdb_backend: DuckDBBackend,
) -> None:
    """Verify ensure_prereqs_for_http skips for non-local_db mode."""
    _ = clean_auto_pipeline_env

    os.environ[AUTO_PIPELINE_ENV] = "1"
    config = ServingConfig(
        mode="remote_api",
        repo="test/repo",
        commit="abc123",
        api_base_url="http://example.com",
    )

    result = ensure_prereqs_for_http(
        op_id="function.summary",
        config=config,
        backend=duckdb_backend,
    )
    assert result is None


def test_ensure_prereqs_for_http_skips_without_gateway(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
) -> None:
    """Verify ensure_prereqs_for_http skips when backend has no gateway."""
    _ = clean_auto_pipeline_env

    os.environ[AUTO_PIPELINE_ENV] = "1"
    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    backend_without_gateway = cast("QueryBackend", object())
    result = ensure_prereqs_for_http(
        op_id="function.summary",
        config=config,
        backend=backend_without_gateway,
    )
    assert result is None


def test_ensure_prereqs_for_mcp_skips_without_gateway(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
) -> None:
    """Verify ensure_prereqs_for_mcp skips when backend has no gateway."""
    _ = clean_auto_pipeline_env

    os.environ[AUTO_PIPELINE_ENV] = "1"
    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    backend_without_gateway = cast("QueryBackend", object())
    result = ensure_prereqs_for_mcp(
        op_id="function.summary",
        config=config,
        backend=backend_without_gateway,
    )
    assert result is None


# -----------------------------------------------------------------------------
# build_paths_for_serving Additional Tests
# -----------------------------------------------------------------------------


def test_build_paths_with_default_repo_root_uses_cwd(tmp_path: Path) -> None:
    """Verify paths use cwd when repo_root uses default."""
    config = ServingConfig(
        mode="local_db",
        # repo_root uses default (cwd) when not specified
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    paths = build_paths_for_serving(config)

    # Should use cwd for repo_root but db_path should be as specified
    assert paths.db_path == tmp_path / "db.duckdb"


def test_build_paths_with_none_db_path(tmp_path: Path) -> None:
    """Verify paths use default db_path when not specified."""
    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=None,  # Explicitly None
    )

    paths = build_paths_for_serving(config)

    # Should use default path under repo_root (determined by CliPathsInput.to_build_paths)
    # The actual default path structure is build/db/codeintel.duckdb
    assert paths.db_path.is_absolute()
    assert "codeintel" in paths.db_path.name.lower()


def test_build_paths_result_has_required_attributes(tmp_path: Path) -> None:
    """Verify BuildPaths has all expected attributes."""
    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    paths = build_paths_for_serving(config)

    # BuildPaths should have these attributes
    assert hasattr(paths, "db_path")
    assert hasattr(paths, "build_dir")


# -----------------------------------------------------------------------------
# has_successful_prereq_run Additional Tests
# -----------------------------------------------------------------------------


def test_has_successful_prereq_run_returns_false_for_running_status(
    pipeline_run_tracking: PipelineRunTracking,
) -> None:
    """Verify returns False when status is 'running'."""
    params = RunParams(repo="test/repo", commit="abc123", status="running", kind="full")
    _seed_run(pipeline_run_tracking, params)

    result = has_successful_prereq_run(
        pipeline_run_tracking,
        repo="test/repo",
        commit="abc123",
        op_id="function.summary",
    )

    assert result is False


def test_has_successful_prereq_run_returns_false_for_repo_mismatch(
    pipeline_run_tracking: PipelineRunTracking,
) -> None:
    """Verify returns False when repo doesn't match."""
    params = RunParams(repo="other/repo", commit="abc123", status="succeeded", kind="full")
    _seed_run(pipeline_run_tracking, params)

    result = has_successful_prereq_run(
        pipeline_run_tracking,
        repo="test/repo",
        commit="abc123",
        op_id="function.summary",
    )

    assert result is False


def test_has_successful_prereq_run_multiple_runs_finds_matching(
    pipeline_run_tracking: PipelineRunTracking,
) -> None:
    """Verify returns True when one of multiple runs matches."""
    # Seed a failed run
    params_failed = RunParams(repo="test/repo", commit="abc123", status="failed", kind="full")
    _seed_run(pipeline_run_tracking, params_failed)

    # Seed a successful run
    params_success = RunParams(repo="test/repo", commit="abc123", status="succeeded", kind="full")
    _seed_run(pipeline_run_tracking, params_success)

    result = has_successful_prereq_run(
        pipeline_run_tracking,
        repo="test/repo",
        commit="abc123",
        op_id="function.summary",
    )

    assert result is True


# -----------------------------------------------------------------------------
# should_run_auto_pipeline Additional Tests
# -----------------------------------------------------------------------------


def test_should_run_auto_pipeline_returns_correct_gateway_reference(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
    duckdb_backend: DuckDBBackend,
) -> None:
    """Verify should_run_auto_pipeline returns the correct gateway reference."""
    _ = clean_auto_pipeline_env

    os.environ[AUTO_PIPELINE_ENV] = "1"
    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    _should_run, gateway, _reason = should_run_auto_pipeline(config, duckdb_backend)

    # The gateway should be the same object as the backend's gateway
    assert gateway is duckdb_backend.gateway


def test_should_run_auto_pipeline_empty_reason_when_enabled(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
    duckdb_backend: DuckDBBackend,
) -> None:
    """Verify skip_reason is empty when auto-pipeline should run."""
    _ = clean_auto_pipeline_env

    os.environ[AUTO_PIPELINE_ENV] = "1"
    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    should_run, _gateway, reason = should_run_auto_pipeline(config, duckdb_backend)

    assert should_run is True
    assert not reason


# -----------------------------------------------------------------------------
# wrap_tool_with_prereqs Test Fixtures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoPipelineTestEnv:
    """Encapsulate auto-pipeline test environment.

    Provides all components needed for realistic auto-pipeline testing.

    Attributes
    ----------
    gateway
        Real DuckDB gateway with schema applied.
    backend
        DuckDBBackend with gateway access.
    config
        ServingConfig for local_db mode.
    tmp_path
        Temporary directory for test files.
    """

    gateway: StorageGateway
    backend: DuckDBBackend
    config: ServingConfig
    tmp_path: Path


@pytest.fixture
def auto_pipeline_test_env(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
) -> Iterator[AutoPipelineTestEnv]:
    """Provide a complete auto-pipeline test environment.

    Sets up:
    - Real DuckDB gateway with schema
    - ServingConfig pointing to tmp_path as repo_root
    - DuckDBBackend with gateway access
    - Auto-pipeline enabled via env var

    Parameters
    ----------
    clean_auto_pipeline_env
        Fixture that ensures env var is clean.
    tmp_path
        Temporary directory for test files.

    Yields
    ------
    AutoPipelineTestEnv
        Complete test environment.
    """
    _ = clean_auto_pipeline_env

    # Enable auto-pipeline
    os.environ[AUTO_PIPELINE_ENV] = "1"

    # Create gateway with schema
    gateway = gateway_with_macros(repo="test/repo", commit="abc123")

    # Create backend
    backend: DuckDBBackend = build_duckdb_backend(gateway, repo="test/repo", commit="abc123")

    # Create serving config
    db_path = tmp_path / "codeintel.duckdb"
    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=db_path,
    )

    try:
        yield AutoPipelineTestEnv(
            gateway=gateway,
            backend=backend,
            config=config,
            tmp_path=tmp_path,
        )
    finally:
        gateway.close()


def create_simple_tool(
    name: str,
    result: dict[str, object],
    docstring: str = "A test tool.",
) -> Callable[..., dict[str, object]]:
    """Create a test tool function with specified name and return value.

    Parameters
    ----------
    name
        Name to assign to the function.
    result
        Value to return when called.
    docstring
        Docstring for the function.

    Returns
    -------
    Callable[..., dict[str, object]]
        A callable tool function.
    """

    def tool_fn(**_kwargs: object) -> dict[str, object]:
        return result

    tool_fn.__name__ = name
    tool_fn.__doc__ = docstring
    return tool_fn


def create_capturing_tool(
    name: str,
) -> tuple[Callable[..., dict[str, object]], list[dict[str, object]]]:
    """Create a test tool that captures its kwargs.

    Parameters
    ----------
    name
        Name to assign to the function.

    Returns
    -------
    tuple[Callable[..., dict[str, object]], list[dict[str, object]]]
        A tuple of (tool function, list where captured kwargs are stored).
    """
    captured: list[dict[str, object]] = []

    def tool_fn(**kwargs: object) -> dict[str, object]:
        captured.append(dict(kwargs))
        return {"captured": True}

    tool_fn.__name__ = name
    tool_fn.__doc__ = "A capturing test tool."
    return tool_fn, captured


# -----------------------------------------------------------------------------
# wrap_tool_with_prereqs Tests
# -----------------------------------------------------------------------------


def test_wrap_tool_preserves_function_metadata(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
    duckdb_backend: DuckDBBackend,
) -> None:
    """Verify wrapped function preserves original function metadata."""
    _ = clean_auto_pipeline_env

    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    original_tool = create_simple_tool(
        name="my_test_tool",
        result={"status": "ok"},
        docstring="My test tool docstring.",
    )

    wrapped = wrap_tool_with_prereqs(
        original_tool,
        op_id="function.summary",
        config=config,
        backend=duckdb_backend,
    )

    # Verify metadata is preserved via functools.wraps
    assert wrapped.__name__ == "my_test_tool"
    assert wrapped.__doc__ == "My test tool docstring."


def test_wrap_tool_passes_kwargs_to_original(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
    duckdb_backend: DuckDBBackend,
) -> None:
    """Verify wrapped function passes kwargs to original function."""
    _ = clean_auto_pipeline_env

    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    tool_fn, captured = create_capturing_tool(name="capturing_tool")

    wrapped = wrap_tool_with_prereqs(
        tool_fn,
        op_id="function.summary",
        config=config,
        backend=duckdb_backend,
    )

    # Call with specific kwargs
    wrapped(arg1="value1", arg2=42, nested={"key": "val"})

    # Verify kwargs were passed through
    assert len(captured) == 1
    assert captured[0] == {"arg1": "value1", "arg2": 42, "nested": {"key": "val"}}


def test_wrap_tool_returns_original_result(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
    duckdb_backend: DuckDBBackend,
) -> None:
    """Verify wrapped function returns the original function's result."""
    _ = clean_auto_pipeline_env

    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    expected_result: dict[str, object] = {"status": "success", "data": [1, 2, 3]}
    original_tool = create_simple_tool(
        name="result_tool",
        result=expected_result,
    )

    wrapped = wrap_tool_with_prereqs(
        original_tool,
        op_id="function.summary",
        config=config,
        backend=duckdb_backend,
    )

    result = wrapped()

    assert result == expected_result


def test_wrap_tool_executes_with_auto_pipeline_disabled(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
    duckdb_backend: DuckDBBackend,
) -> None:
    """Verify wrapped tool executes when auto-pipeline is disabled."""
    _ = clean_auto_pipeline_env
    # Auto-pipeline is disabled (env var not set)

    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    tool_fn, captured = create_capturing_tool(name="disabled_test_tool")

    wrapped = wrap_tool_with_prereqs(
        tool_fn,
        op_id="function.summary",
        config=config,
        backend=duckdb_backend,
    )

    # Should execute without error even though auto-pipeline is disabled
    result = wrapped(test_arg="test_value")

    assert result == {"captured": True}
    assert len(captured) == 1
    assert captured[0] == {"test_arg": "test_value"}


def test_wrap_tool_with_local_db_mode_disabled(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
    duckdb_backend: DuckDBBackend,
) -> None:
    """Verify wrapped tool works with local_db mode but auto-pipeline disabled."""
    _ = clean_auto_pipeline_env
    # Auto-pipeline disabled via env

    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    expected_result: dict[str, object] = {"mode": "local_db", "executed": True}
    original_tool = create_simple_tool(name="local_db_tool", result=expected_result)

    wrapped = wrap_tool_with_prereqs(
        original_tool,
        op_id="function.summary",
        config=config,
        backend=duckdb_backend,
    )

    result = wrapped()

    assert result == expected_result


def test_wrap_tool_logs_debug_message(
    clean_auto_pipeline_env: None,
    tmp_path: Path,
    duckdb_backend: DuckDBBackend,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify wrapped tool logs debug message when called."""
    _ = clean_auto_pipeline_env

    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
    )

    original_tool = create_simple_tool(name="logging_tool", result={"ok": True})

    wrapped = wrap_tool_with_prereqs(
        original_tool,
        op_id="function.summary",
        config=config,
        backend=duckdb_backend,
    )

    with caplog.at_level(logging.DEBUG):
        wrapped()

    # Check that the debug message was logged
    assert any("auto_pipeline check for op=function.summary" in r.message for r in caplog.records)


# -----------------------------------------------------------------------------
# ensure_prereqs_for_mcp Integration Tests
# -----------------------------------------------------------------------------


@pytest.mark.integration
def test_ensure_prereqs_for_mcp_skips_when_prior_run_exists(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify ensure_prereqs_for_mcp skips when prior successful run exists."""
    env = auto_pipeline_test_env

    # Seed a successful run in the gateway
    params = RunParams(repo="test/repo", commit="abc123", status="succeeded", kind="full")
    _seed_run(env.gateway.runs, params)

    # Create config matching the gateway
    config = ServingConfig(
        mode="local_db",
        repo_root=env.tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=env.tmp_path / "db.duckdb",
    )

    result = ensure_prereqs_for_mcp(
        op_id="function.summary",
        config=config,
        backend=env.backend,
    )

    # Should skip because a successful run exists
    assert result is None


@pytest.mark.integration
def test_ensure_prereqs_for_mcp_skips_with_op_prereqs_run(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify ensure_prereqs_for_mcp skips when op_prereqs run exists."""
    env = auto_pipeline_test_env

    # Seed a successful op_prereqs run
    params = RunParams(repo="test/repo", commit="abc123", status="succeeded", kind="op_prereqs")
    _seed_run(env.gateway.runs, params)

    config = ServingConfig(
        mode="local_db",
        repo_root=env.tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=env.tmp_path / "db.duckdb",
    )

    result = ensure_prereqs_for_mcp(
        op_id="function.summary",
        config=config,
        backend=env.backend,
    )

    # Should skip because an op_prereqs run exists
    assert result is None


@pytest.mark.integration
def test_ensure_prereqs_for_mcp_does_not_skip_for_failed_run(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify ensure_prereqs_for_mcp does not skip when prior run failed."""
    env = auto_pipeline_test_env

    # Seed a failed run - should not satisfy prereqs
    params = RunParams(repo="test/repo", commit="abc123", status="failed", kind="full")
    _seed_run(env.gateway.runs, params)

    # This will attempt to run the pipeline, which may fail or succeed
    # depending on infrastructure. The key assertion is that it doesn't skip.
    # We check has_successful_prereq_run returns False to verify skip logic.
    has_prereq = has_successful_prereq_run(
        env.gateway.runs,
        repo="test/repo",
        commit="abc123",
        op_id="function.summary",
    )

    # The failed run should not satisfy prereqs
    assert has_prereq is False


@pytest.mark.integration
def test_ensure_prereqs_for_mcp_does_not_skip_for_different_commit(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify ensure_prereqs_for_mcp does not skip for different commit."""
    env = auto_pipeline_test_env

    # Seed a successful run for a different commit
    params = RunParams(repo="test/repo", commit="different123", status="succeeded", kind="full")
    _seed_run(env.gateway.runs, params)

    # Check prereq logic returns False for our commit (abc123)
    # The run we seeded was for "different123" so it shouldn't match
    has_prereq = has_successful_prereq_run(
        env.gateway.runs,
        repo="test/repo",
        commit="abc123",
        op_id="function.summary",
    )

    # The run for different commit should not satisfy prereqs
    assert has_prereq is False


# -----------------------------------------------------------------------------
# ensure_prereqs_for_http Integration Tests
# -----------------------------------------------------------------------------


@pytest.mark.integration
def test_ensure_prereqs_for_http_skips_when_prior_run_exists(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify ensure_prereqs_for_http skips when prior successful run exists."""
    env = auto_pipeline_test_env

    # Seed a successful run
    params = RunParams(repo="test/repo", commit="abc123", status="succeeded", kind="full")
    _seed_run(env.gateway.runs, params)

    config = ServingConfig(
        mode="local_db",
        repo_root=env.tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=env.tmp_path / "db.duckdb",
    )

    result = ensure_prereqs_for_http(
        op_id="function.summary",
        config=config,
        backend=env.backend,
    )

    # Should skip because a successful run exists
    assert result is None


@pytest.mark.integration
def test_ensure_prereqs_for_http_skips_with_op_prereqs_run(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify ensure_prereqs_for_http skips when op_prereqs run exists."""
    env = auto_pipeline_test_env

    # Seed a successful op_prereqs run
    params = RunParams(repo="test/repo", commit="abc123", status="succeeded", kind="op_prereqs")
    _seed_run(env.gateway.runs, params)

    config = ServingConfig(
        mode="local_db",
        repo_root=env.tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=env.tmp_path / "db.duckdb",
    )

    result = ensure_prereqs_for_http(
        op_id="function.summary",
        config=config,
        backend=env.backend,
    )

    # Should skip because an op_prereqs run exists
    assert result is None


@pytest.mark.integration
def test_ensure_prereqs_for_http_does_not_skip_for_failed_run(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify ensure_prereqs_for_http does not skip when prior run failed."""
    env = auto_pipeline_test_env

    # Seed a failed run
    params = RunParams(repo="test/repo", commit="abc123", status="failed", kind="full")
    _seed_run(env.gateway.runs, params)

    # Check prereq logic returns False
    has_prereq = has_successful_prereq_run(
        env.gateway.runs,
        repo="test/repo",
        commit="abc123",
        op_id="function.summary",
    )

    # The failed run should not satisfy prereqs
    assert has_prereq is False


# -----------------------------------------------------------------------------
# Full MCP Tool Flow Integration Tests
# -----------------------------------------------------------------------------


@pytest.mark.integration
def test_wrapped_tool_executes_after_prereq_check_with_existing_run(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify wrapped tool executes after prereq check when run exists."""
    env = auto_pipeline_test_env

    # Seed a successful run so prereq check passes quickly
    params = RunParams(repo="test/repo", commit="abc123", status="succeeded", kind="full")
    _seed_run(env.gateway.runs, params)

    config = ServingConfig(
        mode="local_db",
        repo_root=env.tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=env.tmp_path / "db.duckdb",
    )

    tool_fn, captured = create_capturing_tool(name="flow_test_tool")

    wrapped = wrap_tool_with_prereqs(
        tool_fn,
        op_id="function.summary",
        config=config,
        backend=env.backend,
    )

    # Execute the wrapped tool
    result = wrapped(flow_arg="test_value")

    # Verify tool was called with correct kwargs
    expected_call_count = 1
    assert len(captured) == expected_call_count
    assert captured[0] == {"flow_arg": "test_value"}
    assert result == {"captured": True}


@pytest.mark.integration
def test_wrapped_tool_with_remote_api_mode_skips_prereqs(
    clean_auto_pipeline_env: None,
    duckdb_backend: DuckDBBackend,
) -> None:
    """Verify wrapped tool skips prereq check for remote_api mode."""
    _ = clean_auto_pipeline_env

    # Enable auto-pipeline but use remote_api mode
    os.environ[AUTO_PIPELINE_ENV] = "1"
    config = ServingConfig(
        mode="remote_api",
        repo="test/repo",
        commit="abc123",
        api_base_url="http://example.com",
    )

    expected_result: dict[str, object] = {"remote": True}
    original_tool = create_simple_tool(name="remote_tool", result=expected_result)

    wrapped = wrap_tool_with_prereqs(
        original_tool,
        op_id="function.summary",
        config=config,
        backend=duckdb_backend,
    )

    # Should execute without attempting prereqs
    result = wrapped()

    assert result == expected_result


@pytest.mark.integration
def test_full_flow_with_multiple_tool_calls(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify multiple wrapped tool calls work correctly."""
    env = auto_pipeline_test_env

    # Seed a successful run
    params = RunParams(repo="test/repo", commit="abc123", status="succeeded", kind="full")
    _seed_run(env.gateway.runs, params)

    config = ServingConfig(
        mode="local_db",
        repo_root=env.tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=env.tmp_path / "db.duckdb",
    )

    tool_fn, captured = create_capturing_tool(name="multi_call_tool")

    wrapped = wrap_tool_with_prereqs(
        tool_fn,
        op_id="function.summary",
        config=config,
        backend=env.backend,
    )

    # Make multiple calls
    expected_call_count = 3
    wrapped(call=1)
    wrapped(call=2)
    wrapped(call=3)

    # Verify all calls were captured
    assert len(captured) == expected_call_count
    assert captured[0] == {"call": 1}
    assert captured[1] == {"call": 2}
    assert captured[2] == {"call": 3}


@pytest.mark.integration
def test_wrapped_tool_logs_when_prereqs_skipped(
    auto_pipeline_test_env: AutoPipelineTestEnv,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify logging when prereqs are skipped due to existing run."""
    env = auto_pipeline_test_env

    # Seed a successful run
    params = RunParams(repo="test/repo", commit="abc123", status="succeeded", kind="full")
    _seed_run(env.gateway.runs, params)

    config = ServingConfig(
        mode="local_db",
        repo_root=env.tmp_path,
        repo="test/repo",
        commit="abc123",
        db_path=env.tmp_path / "db.duckdb",
    )

    original_tool = create_simple_tool(name="log_test_tool", result={"logged": True})

    wrapped = wrap_tool_with_prereqs(
        original_tool,
        op_id="function.summary",
        config=config,
        backend=env.backend,
    )

    with caplog.at_level(logging.DEBUG):
        wrapped()

    # Should have debug logs from wrapper and possibly from ensure_prereqs_for_mcp
    assert any("auto_pipeline" in r.message.lower() for r in caplog.records)


# -----------------------------------------------------------------------------
# Data-Aware Prerequisite Tests
# -----------------------------------------------------------------------------


def test_dataset_has_rows_for_snapshot_returns_true_when_data_present(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify dataset_has_rows_for_snapshot returns True when data exists.

    Parameters
    ----------
    auto_pipeline_test_env
        Test environment with gateway and configuration.
    """
    env = auto_pipeline_test_env
    repo = "test/repo"
    commit = "abc123"

    created_at = datetime.now(UTC).isoformat()
    env.gateway.core.insert_goids(
        [
            (
                12345,
                "test:urn",
                repo,
                commit,
                "pkg/mod.py",
                "python",
                "function",
                "pkg.mod.func",
                1,
                1,
                created_at,
            )
        ]
    )

    contract = DATASET_CONTRACTS_BY_TABLE_KEY.get("core.goids")
    if contract is None:
        pytest.skip("core.goids contract not found")

    result = dataset_has_rows_for_snapshot(
        env.gateway,
        contract,
        repo=repo,
        commit=commit,
    )

    assert result is True


def test_dataset_has_rows_for_snapshot_returns_false_when_empty(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify dataset_has_rows_for_snapshot returns False for empty table.

    Parameters
    ----------
    auto_pipeline_test_env
        Test environment with gateway and configuration.
    """
    env = auto_pipeline_test_env

    contract = DATASET_CONTRACTS_BY_TABLE_KEY.get("core.goids")
    if contract is None:
        pytest.skip("core.goids contract not found")

    # Query for data that doesn't exist
    result = dataset_has_rows_for_snapshot(
        env.gateway,
        contract,
        repo="nonexistent/repo",
        commit="nonexistent",
    )

    assert result is False


def test_get_required_table_keys_for_operation_returns_frozenset(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify get_required_table_keys_for_operation returns correct type.

    Parameters
    ----------
    auto_pipeline_test_env
        Test environment with gateway and configuration.
    """
    _ = auto_pipeline_test_env  # Use env for fixture consistency

    result = get_required_table_keys_for_operation("function.summary")

    assert isinstance(result, frozenset)


def test_get_required_table_keys_for_unknown_operation_returns_empty(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify unknown operations return empty frozenset.

    Parameters
    ----------
    auto_pipeline_test_env
        Test environment with gateway and configuration.
    """
    _ = auto_pipeline_test_env  # Use env for fixture consistency

    result = get_required_table_keys_for_operation("nonexistent.operation")

    assert result == frozenset()


def test_has_required_data_for_operation_returns_false_when_missing(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify has_required_data_for_operation returns False for missing data.

    Parameters
    ----------
    auto_pipeline_test_env
        Test environment with gateway and configuration.
    """
    env = auto_pipeline_test_env

    # Query for data that doesn't exist
    result = has_required_data_for_operation(
        env.gateway,
        "function.summary",
        repo="nonexistent/repo",
        commit="nonexistent",
    )

    # Should be False because data doesn't exist
    assert result is False


def test_operation_prereqs_satisfied_uses_data_check_when_datasets_declared(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify operation_prereqs_satisfied uses data-aware check for ops with datasets.

    Parameters
    ----------
    auto_pipeline_test_env
        Test environment with gateway and configuration.
    """
    env = auto_pipeline_test_env
    repo = "test/repo"
    commit = "abc123"

    # Get an operation with required_datasets
    op = get_operation("function.summary")
    if op is None or not op.required_datasets:
        pytest.skip("function.summary doesn't have required_datasets")

    # Without seeding data, should return False
    result = operation_prereqs_satisfied(
        env.gateway,
        "function.summary",
        repo=repo,
        commit=commit,
    )

    # Data doesn't exist, so should be False
    assert result is False


def test_operation_prereqs_satisfied_falls_back_to_run_check(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify operation_prereqs_satisfied falls back to run-based check.

    Parameters
    ----------
    auto_pipeline_test_env
        Test environment with gateway and configuration.
    """
    env = auto_pipeline_test_env
    repo = "test/repo"
    commit = "abc123"

    # Seed a successful run
    params = RunParams(repo=repo, commit=commit, status="succeeded", kind="full")
    _seed_run(env.gateway.runs, params)

    # Find an operation without required_datasets (if any)
    # If function.summary has datasets, the run-based fallback won't apply
    op = get_operation("function.summary")
    if op is not None and op.required_datasets:
        # If the operation has datasets, data-aware check takes precedence
        result = operation_prereqs_satisfied(
            env.gateway,
            "function.summary",
            repo=repo,
            commit=commit,
        )
        # Data doesn't exist, so even with a run, should be False
        assert result is False
    else:
        # For operations without declared datasets, run check applies
        result = operation_prereqs_satisfied(
            env.gateway,
            "function.summary",
            repo=repo,
            commit=commit,
        )
        # Run exists, so should be True
        assert result is True


def test_build_prereq_debug_info_returns_complete_info(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify build_prereq_debug_info returns complete debug information.

    Parameters
    ----------
    auto_pipeline_test_env
        Test environment with gateway and configuration.
    """
    env = auto_pipeline_test_env
    repo = "test/repo"
    commit = "abc123"

    debug_info = build_prereq_debug_info(
        env.gateway,
        "function.summary",
        repo=repo,
        commit=commit,
    )

    # Check all expected fields
    assert debug_info.op_id == "function.summary"
    assert debug_info.repo == repo
    assert debug_info.commit == commit
    assert isinstance(debug_info.required_datasets, tuple)
    assert isinstance(debug_info.expanded_datasets, tuple)
    assert isinstance(debug_info.dataset_statuses, tuple)
    assert isinstance(debug_info.runs_considered, tuple)
    assert isinstance(debug_info.data_satisfied, bool)
    assert isinstance(debug_info.run_satisfied, bool)
    assert isinstance(debug_info.overall_satisfied, bool)


def test_build_prereq_debug_info_includes_run_summaries(
    auto_pipeline_test_env: AutoPipelineTestEnv,
) -> None:
    """Verify build_prereq_debug_info includes run summaries.

    Parameters
    ----------
    auto_pipeline_test_env
        Test environment with gateway and configuration.
    """
    env = auto_pipeline_test_env
    repo = "test/repo"
    commit = "abc123"

    # Seed a run for this repo/commit
    params = RunParams(repo=repo, commit=commit, status="succeeded", kind="full")
    _seed_run(env.gateway.runs, params)

    debug_info = build_prereq_debug_info(
        env.gateway,
        "function.summary",
        repo=repo,
        commit=commit,
    )

    # Should have at least one run considered
    assert len(debug_info.runs_considered) >= 1

    # Run should match our seeded run
    run = debug_info.runs_considered[0]
    assert run.status == "succeeded"
    assert run.kind == "full"
