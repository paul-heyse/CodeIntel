"""Tests for CstExtractPlugin wiring and fallbacks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from codeintel.build.context import TargetExecutionContext
from codeintel.ingestion.compute import CstExtractStep
from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.plugins.cst_extract import CstExtractPlugin
from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort
from codeintel.ingestion.ports.storage import IngestStoragePort
from codeintel.storage.gateway import StorageGateway
from tests._helpers import DEFAULT_COMMIT, DEFAULT_REPO, build_repo_tree, make_target_context
from tests._helpers.assertions import expect_equal, expect_in, expect_true
from tests._helpers.env import create_test_env
from tests._helpers.fakes.ingestion_plugins import (
    RecordingDiscoveryAdapter,
    RecordingStep,
    RecordingStorageAdapter,
    StepCallCapture,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from codeintel.storage.gateway import DuckDBConnection


class _RecordingConnection:
    """Connection wrapper that records executions and delegates to real connection."""

    def __init__(
        self,
        real_con: DuckDBConnection,
        executions: list[tuple[str, list[object]]],
    ) -> None:
        self._real_con = real_con
        self._executions = executions

    def execute(self, sql: str, params: Sequence[object] | None = None) -> _RecordingConnection:
        """Record and forward SQL execution.

        Parameters
        ----------
        sql
            SQL query to execute.
        params
            Query parameters.

        Returns
        -------
        _RecordingConnection
            Self for chaining.
        """
        self._executions.append((sql, list(params or [])))
        self._real_con.execute(sql, params)
        return self

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Fetch all results from the underlying connection.

        Returns
        -------
        list[tuple[Any, ...]]
            All rows from the query.
        """
        return self._real_con.fetchall()

    def __getattr__(self, item: str) -> object:
        """Delegate unknown attributes to the real connection.

        Returns
        -------
        object
            Attribute from the underlying connection.
        """
        return getattr(self._real_con, item)


class ConnectionRecordingGateway:
    """Gateway wrapper that records con.execute() calls.

    This wrapper intercepts calls through the `con` property to record
    SQL executions that go through `gateway.con.execute()`.
    """

    def __init__(self, gateway: StorageGateway) -> None:
        self._gateway = gateway
        self.executions: list[tuple[str, list[object]]] = []
        self._recording_con = _RecordingConnection(gateway.con, self.executions)

    @property
    def con(self) -> DuckDBConnection:
        """Return the recording connection wrapper.

        Returns
        -------
        DuckDBConnection
            Recording connection that tracks executions.
        """
        return cast("DuckDBConnection", self._recording_con)

    def close(self) -> None:
        """Close the underlying gateway."""
        self._gateway.close()

    def __getattr__(self, item: str) -> object:
        """Delegate unknown attributes to the real gateway.

        Returns
        -------
        object
            Attribute from the underlying gateway.
        """
        return getattr(self._gateway, item)


class FailingGateway:
    """Gateway that raises on execute for testing error recovery.

    This is a proper test double that implements the gateway interface
    but raises OSError on execute to simulate database failures.
    """

    def __init__(self, error_message: str = "no db") -> None:
        self._error_message = error_message
        self.records: list[tuple[str, tuple[object, ...]]] = []

    @property
    def con(self) -> DuckDBConnection:
        """Return a failing connection proxy.

        Returns
        -------
        DuckDBConnection
            A proxy that fails on execute.
        """
        return cast("DuckDBConnection", _FailingConnectionProxy(self._error_message))

    def execute(self, sql: str, params: Iterable[object] | None = None) -> DuckDBConnection:
        """Record and raise on SQL execution.

        Raises
        ------
        OSError
            Always raises to simulate database failure.
        """
        self.records.append((sql, tuple(params or ())))
        raise OSError(self._error_message)

    def close(self) -> None:
        """No-op close."""


class _FailingConnectionProxy:
    """Connection proxy that fails on execute."""

    def __init__(self, error_message: str) -> None:
        self._error_message = error_message

    def execute(self, sql: str, params: object = None) -> _FailingConnectionProxy:
        """Raise OSError to simulate database failure.

        Raises
        ------
        OSError
            Always raises to simulate database failure.
        """
        _ = sql, params
        raise OSError(self._error_message)


def _make_plugin(
    capture: StepCallCapture,
    *,
    result: StepResult | None = None,
    table_key: str = "core.cst_nodes",
) -> CstExtractPlugin:
    return CstExtractPlugin(
        storage_adapter_factory=lambda gateway: RecordingStorageAdapter(gateway, capture=capture),
        discovery_adapter_factory=lambda repo_root: RecordingDiscoveryAdapter(
            repo_root, capture=capture
        ),
        step_factory=_build_step_factory(
            capture=capture,
            table_key=table_key,
            result=result,
        ),
    )


def _build_step_factory(
    *,
    capture: StepCallCapture,
    table_key: str,
    result: StepResult | None,
) -> Callable[
    [
        IngestStoragePort,
        ModuleDiscoveryPort,
    ],
    CstExtractStep,
]:
    def _factory(storage: IngestStoragePort, discovery: ModuleDiscoveryPort) -> CstExtractStep:
        return cast(
            "CstExtractStep",
            RecordingStep(
                storage,
                discovery,
                capture=capture,
                table_key=table_key,
                result=result,
            ),
        )

    return _factory


@pytest.mark.anyio
async def test_execute_logs_errors_and_succeeds(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Errors from the step should be logged but still return a success result."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/cst_mod.py": "y = 2\n"})
    ctx = make_target_context(repo_root=repo_root, modules=("pkg/cst_mod.py",))
    captured = StepCallCapture()
    failing_result = StepResult(errors=["bad cst"])

    with caplog.at_level("WARNING", logger="codeintel.ingestion.plugins.cst_extract"):
        result = await _make_plugin(captured, result=failing_result).execute(
            cast("TargetExecutionContext", ctx)
        )

    expect_true(result.success is True)
    expect_equal(result.row_counts, {})
    expect_true(isinstance(captured.storage, RecordingStorageAdapter))
    expect_equal(captured.repo_root, repo_root)
    expect_equal(captured.repo, DEFAULT_REPO)
    expect_equal(captured.commit, DEFAULT_COMMIT)
    module_record = captured.modules[0]
    expect_equal(module_record.rel_path, "pkg/cst_mod.py")
    expect_equal(module_record.file_path, repo_root / "pkg/cst_mod.py")
    expect_true(any("bad cst" in record.getMessage() for record in caplog.records))


@pytest.mark.anyio
async def test_execute_queries_gateway_when_modules_missing(tmp_path: Path) -> None:
    """Gateway results should seed module records when resources.modules is empty."""
    repo_root = build_repo_tree(tmp_path / "repo", {})

    # Create a real gateway and seed test data
    env = create_test_env(tmp_path / "env", repo_root=repo_root)
    env.gateway.con.execute(
        "INSERT INTO core.modules (module, path, repo, commit) VALUES (?, ?, ?, ?)",
        ["pkg.from_db", "pkg/from_db.py", DEFAULT_REPO, DEFAULT_COMMIT],
    )

    # Wrap with ConnectionRecordingGateway to track con.execute() calls
    recording_gateway = ConnectionRecordingGateway(env.gateway)

    ctx = make_target_context(
        repo_root=repo_root, modules=(), gateway=cast("StorageGateway", recording_gateway)
    )
    captured = StepCallCapture()

    try:
        result = await _make_plugin(captured).execute(cast("TargetExecutionContext", ctx))

        expect_equal(result.row_counts, {"core.cst_nodes": 1})
        # Verify the SQL query was made
        sql, params = recording_gateway.executions[0]
        expect_in("core.modules", sql)
        expect_equal(params, [DEFAULT_REPO, DEFAULT_COMMIT])
        module_record = captured.modules[0]
        expect_equal(module_record.rel_path, "pkg/from_db.py")
        expect_equal(module_record.file_path, repo_root / "pkg/from_db.py")
        expect_equal(captured.repo, DEFAULT_REPO)
        expect_equal(captured.commit, DEFAULT_COMMIT)
    finally:
        env.close()


@pytest.mark.anyio
async def test_execute_handles_gateway_errors(tmp_path: Path) -> None:
    """Database errors should be swallowed and an empty module list passed through."""
    repo_root = build_repo_tree(tmp_path / "repo", {})
    # Use a FailingGateway that raises OSError on execute
    gateway = FailingGateway(error_message="no db")
    ctx = make_target_context(
        repo_root=repo_root, modules=(), gateway=cast("StorageGateway", gateway)
    )
    captured = StepCallCapture()

    result = await _make_plugin(captured).execute(cast("TargetExecutionContext", ctx))

    expect_true(result.success is True)
    expect_equal(captured.modules, [])
    expect_equal(captured.repo, DEFAULT_REPO)
    expect_equal(captured.commit, DEFAULT_COMMIT)
