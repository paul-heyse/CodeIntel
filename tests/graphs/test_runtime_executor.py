"""Tests for graph runtime executor.

This module tests the graph plugin execution infrastructure including
plugin batch execution, timeout handling, retry logic, and fatal error
propagation.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import (
    GraphPluginPolicy,
    GraphPluginRetryPolicy,
)
from codeintel.graphs.core.context import GraphExecutionContext
from codeintel.graphs.core.protocol import (
    FunctionalGraphPlugin,
    GraphPluginMetadata,
    GraphPluginProtocol,
)
from codeintel.graphs.core.registry import get_graph_registry, register_graph_plugin
from codeintel.graphs.core.result import GraphPluginResult, GraphPluginRunRecord
from codeintel.graphs.runtime.executor import (
    GraphExecutorContext,
    GraphRunReport,
    PluginFatalError,
    run_graph_plugin_batch,
    run_graph_plugins,
)
from codeintel.graphs.runtime.planning import (
    GraphPlanContext,
    plan_graph_plugin_run,
)
from codeintel.storage.schemas import apply_all_schemas
from tests._helpers.gateway import open_ingestion_gateway_with_macros

EXPECTED_PLUGIN_COUNT: Final = 3
EXPECTED_BATCH_COUNT: Final = 2


class _PluginRegistrar:
    """Context manager for registering and cleaning up test plugins."""

    def __init__(self, plugins: list[GraphPluginProtocol]) -> None:
        """Initialize with plugins to register.

        Parameters
        ----------
        plugins
            Plugins to register.
        """
        self._plugins = plugins
        self._registry = get_graph_registry()

    def __enter__(self) -> None:
        """Register plugins on entry."""
        for plugin in self._plugins:
            register_graph_plugin(plugin)

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Unregister plugins on exit."""
        for plugin in self._plugins:
            self._registry.unregister(plugin.metadata.name)


def _make_test_plugin(
    name: str,
    *,
    succeed: bool = True,
    row_counts: dict[str, int] | None = None,
    raise_exception: type[Exception] | None = None,
    delay_ms: int = 0,
) -> GraphPluginProtocol:
    """Create a test plugin for execution tests.

    Parameters
    ----------
    name
        Plugin name.
    succeed
        Whether the plugin should succeed.
    row_counts
        Optional row counts to return.
    raise_exception
        Optional exception type to raise.
    delay_ms
        Optional delay in milliseconds before returning.

    Returns
    -------
    GraphPluginProtocol
        Test plugin instance.
    """

    def execute(_ctx: GraphExecutionContext) -> GraphPluginResult:
        if delay_ms > 0:
            time.sleep(delay_ms / 1000)
        if raise_exception is not None:
            error_msg = f"Test exception from {name}"
            raise raise_exception(error_msg)
        if succeed:
            return GraphPluginResult.ok(row_counts=row_counts)
        return GraphPluginResult.fail(f"Plugin {name} failed")

    metadata = GraphPluginMetadata(
        name=name,
        description=f"Test plugin {name}",
        kind="builder",
        stage="goid",
    )

    return FunctionalGraphPlugin(_metadata=metadata, _execute_fn=execute)


def _make_context(tmp_path: Path) -> tuple[object, SnapshotRef]:
    """Create execution context for tests.

    Parameters
    ----------
    tmp_path
        Temporary path for test data.

    Returns
    -------
    tuple
        Gateway and snapshot for testing.
    """
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    apply_all_schemas(gateway.con)
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)
    return gateway, snapshot


def test_run_graph_plugins_basic_success(tmp_path: Path) -> None:
    """Execute a simple plugin batch successfully.

    Raises
    ------
    AssertionError
        If plugin execution fails.
    """
    gateway, snapshot = _make_context(tmp_path)
    try:
        plugin = _make_test_plugin("basic_success_plugin", succeed=True, row_counts={"t": 5})

        with _PluginRegistrar([plugin]):
            context = GraphPlanContext(
                runtime_snapshot=snapshot,
                policy=GraphPluginPolicy(),
            )
            plan = plan_graph_plugin_run(
                plugin_names=[plugin.metadata.name],
                context=context,
            )

            executor_context = GraphExecutorContext(
                gateway=gateway,
                snapshot=snapshot,
            )

            report = run_graph_plugins(plan=plan, context=executor_context)

            if report.success_count != 1:
                msg = f"Expected 1 success, got {report.success_count}"
                raise AssertionError(msg)
            if report.failure_count != 0:
                msg = f"Expected 0 failures, got {report.failure_count}"
                raise AssertionError(msg)
    finally:
        gateway.close()


def test_run_graph_plugins_with_failure(tmp_path: Path) -> None:
    """Plugin failure handling records the failure correctly.

    Raises
    ------
    AssertionError
        If failure is not recorded correctly.
    """
    gateway, snapshot = _make_context(tmp_path)
    try:
        plugin = _make_test_plugin("failing_plugin", succeed=False)

        with _PluginRegistrar([plugin]):
            context = GraphPlanContext(
                runtime_snapshot=snapshot,
                policy=GraphPluginPolicy(default_severity="soft_fail"),
            )
            plan = plan_graph_plugin_run(
                plugin_names=[plugin.metadata.name],
                context=context,
            )

            executor_context = GraphExecutorContext(
                gateway=gateway,
                snapshot=snapshot,
            )

            report = run_graph_plugins(plan=plan, context=executor_context)

            if report.failure_count != 1:
                msg = f"Expected 1 failure, got {report.failure_count}"
                raise AssertionError(msg)
            if report.success_count != 0:
                msg = f"Expected 0 successes, got {report.success_count}"
                raise AssertionError(msg)
    finally:
        gateway.close()


def test_run_graph_plugins_timeout(tmp_path: Path) -> None:
    """Timeout handling produces failed record with timeout error.

    Raises
    ------
    AssertionError
        If timeout failure is not recorded correctly.
    """
    gateway, snapshot = _make_context(tmp_path)
    try:
        # Use a plugin that delays long enough to trigger timeout
        plugin = _make_test_plugin("slow_plugin", succeed=True, delay_ms=2000)

        with _PluginRegistrar([plugin]):
            context = GraphPlanContext(
                runtime_snapshot=snapshot,
                policy=GraphPluginPolicy(
                    timeouts_ms={plugin.metadata.name: 50},
                    default_severity="soft_fail",
                ),
            )
            plan = plan_graph_plugin_run(
                plugin_names=[plugin.metadata.name],
                context=context,
            )

            executor_context = GraphExecutorContext(
                gateway=gateway,
                snapshot=snapshot,
            )

            report = run_graph_plugins(plan=plan, context=executor_context)

            if report.failure_count != 1:
                msg = f"Expected 1 failure from timeout, got {report.failure_count}"
                raise AssertionError(msg)
            # Check the record has timeout error
            if not report.records:
                msg = "Expected at least one record"
                raise AssertionError(msg)
            rec = report.records[0]
            if rec.error != "timeout":
                msg = f"Expected error 'timeout', got '{rec.error}'"
                raise AssertionError(msg)
    finally:
        gateway.close()


def test_run_graph_plugins_retry_logic(tmp_path: Path) -> None:
    """Retry logic executes multiple attempts before failing.

    Raises
    ------
    AssertionError
        If retry count is incorrect.
    """
    gateway, snapshot = _make_context(tmp_path)
    try:
        plugin = _make_test_plugin("retry_plugin", raise_exception=RuntimeError)

        with _PluginRegistrar([plugin]):
            context = GraphPlanContext(
                runtime_snapshot=snapshot,
                policy=GraphPluginPolicy(
                    default_severity="soft_fail",
                    retries={plugin.metadata.name: GraphPluginRetryPolicy(max_attempts=3)},
                ),
            )
            plan = plan_graph_plugin_run(
                plugin_names=[plugin.metadata.name],
                context=context,
            )

            executor_context = GraphExecutorContext(
                gateway=gateway,
                snapshot=snapshot,
            )

            report = run_graph_plugins(plan=plan, context=executor_context)

            if report.failure_count != 1:
                msg = f"Expected 1 failure after retries, got {report.failure_count}"
                raise AssertionError(msg)
            if not report.records:
                msg = "Expected at least one record"
                raise AssertionError(msg)
            rec = report.records[0]
            if rec.attempts != EXPECTED_PLUGIN_COUNT:
                msg = f"Expected {EXPECTED_PLUGIN_COUNT} attempts, got {rec.attempts}"
                raise AssertionError(msg)
    finally:
        gateway.close()


def test_run_graph_plugins_fatal_error(tmp_path: Path) -> None:
    """PluginFatalError stops execution and is recorded.

    Raises
    ------
    AssertionError
        If fatal error handling fails.
    """
    gateway, snapshot = _make_context(tmp_path)
    try:
        # First plugin raises exception with fatal severity + fail_fast
        fatal_plugin = _make_test_plugin("fatal_plugin", raise_exception=ValueError)
        # Second plugin should not execute
        second_plugin = _make_test_plugin("second_plugin", succeed=True)

        with _PluginRegistrar([fatal_plugin, second_plugin]):
            context = GraphPlanContext(
                runtime_snapshot=snapshot,
                policy=GraphPluginPolicy(
                    default_severity="fatal",
                    fail_fast=True,
                ),
            )
            plan = plan_graph_plugin_run(
                plugin_names=[fatal_plugin.metadata.name, second_plugin.metadata.name],
                context=context,
            )

            executor_context = GraphExecutorContext(
                gateway=gateway,
                snapshot=snapshot,
            )

            report = run_graph_plugins(plan=plan, context=executor_context)

            if not report.fatal_error:
                msg = "Expected fatal_error to be True"
                raise AssertionError(msg)
            # Only one record should exist (fatal stopped execution)
            if len(report.records) != 1:
                msg = f"Expected 1 record (fatal stopped), got {len(report.records)}"
                raise AssertionError(msg)
    finally:
        gateway.close()


def test_run_graph_plugin_batch_convenience(tmp_path: Path) -> None:
    """run_graph_plugin_batch convenience function executes plugins.

    Raises
    ------
    AssertionError
        If batch execution fails.
    """
    gateway, snapshot = _make_context(tmp_path)
    try:
        plugins = [
            _make_test_plugin("batch_plugin_1", succeed=True),
            _make_test_plugin("batch_plugin_2", succeed=True),
        ]

        with _PluginRegistrar(plugins):
            report = run_graph_plugin_batch(
                plugins=plugins,
                gateway=gateway,
                snapshot=snapshot,
            )

            if report.success_count != EXPECTED_BATCH_COUNT:
                msg = f"Expected {EXPECTED_BATCH_COUNT} successes, got {report.success_count}"
                raise AssertionError(msg)
            if report.repo != "demo/repo":
                msg = f"Expected repo 'demo/repo', got '{report.repo}'"
                raise AssertionError(msg)
    finally:
        gateway.close()


def test_run_graph_plugins_multiple_success(tmp_path: Path) -> None:
    """Multiple successful plugins all complete and are recorded.

    Raises
    ------
    AssertionError
        If not all plugins succeeded.
    """
    gateway, snapshot = _make_context(tmp_path)
    try:
        plugins = [_make_test_plugin(f"multi_plugin_{i}", succeed=True) for i in range(3)]

        with _PluginRegistrar(plugins):
            context = GraphPlanContext(
                runtime_snapshot=snapshot,
                policy=GraphPluginPolicy(),
            )
            plan = plan_graph_plugin_run(
                plugin_names=[p.metadata.name for p in plugins],
                context=context,
            )

            executor_context = GraphExecutorContext(
                gateway=gateway,
                snapshot=snapshot,
            )

            report = run_graph_plugins(plan=plan, context=executor_context)

            if report.success_count != EXPECTED_PLUGIN_COUNT:
                msg = f"Expected {EXPECTED_PLUGIN_COUNT} successes, got {report.success_count}"
                raise AssertionError(msg)
            if len(report.records) != EXPECTED_PLUGIN_COUNT:
                msg = f"Expected {EXPECTED_PLUGIN_COUNT} records, got {len(report.records)}"
                raise AssertionError(msg)
    finally:
        gateway.close()


def test_run_graph_plugins_includes_timing(tmp_path: Path) -> None:
    """Report includes duration and timestamps.

    Raises
    ------
    AssertionError
        If timing information is missing or invalid.
    """
    gateway, snapshot = _make_context(tmp_path)
    try:
        plugin = _make_test_plugin("timing_plugin", succeed=True)

        with _PluginRegistrar([plugin]):
            context = GraphPlanContext(
                runtime_snapshot=snapshot,
                policy=GraphPluginPolicy(),
            )
            plan = plan_graph_plugin_run(
                plugin_names=[plugin.metadata.name],
                context=context,
            )

            executor_context = GraphExecutorContext(
                gateway=gateway,
                snapshot=snapshot,
            )

            report = run_graph_plugins(plan=plan, context=executor_context)

            if report.duration_ms < 0:
                msg = "Expected non-negative duration"
                raise AssertionError(msg)
            if not report.started_at:
                msg = "Expected started_at to be set"
                raise AssertionError(msg)
            if not report.ended_at:
                msg = "Expected ended_at to be set"
                raise AssertionError(msg)
            # Validate ISO format parses correctly
            datetime.fromisoformat(report.started_at)
            datetime.fromisoformat(report.ended_at)
    finally:
        gateway.close()


def test_graph_run_report_attributes() -> None:
    """GraphRunReport captures all execution attributes correctly.

    Raises
    ------
    AssertionError
        If attributes are not correctly captured.
    """
    record = GraphPluginRunRecord(
        name="test",
        status="succeeded",
        started_at=datetime.now(tz=UTC).isoformat(),
        ended_at=datetime.now(tz=UTC).isoformat(),
        duration_ms=100.0,
    )
    report = GraphRunReport(
        run_id="run-123",
        repo="demo/repo",
        commit="abc123",
        records=(record,),
        success_count=1,
        failure_count=0,
        skip_count=0,
        duration_ms=150.0,
        started_at=datetime.now(tz=UTC).isoformat(),
        ended_at=datetime.now(tz=UTC).isoformat(),
    )

    if report.run_id != "run-123":
        msg = f"Expected run_id 'run-123', got '{report.run_id}'"
        raise AssertionError(msg)
    if len(report.records) != 1:
        msg = f"Expected 1 record, got {len(report.records)}"
        raise AssertionError(msg)
    if report.fatal_error:
        msg = "Expected fatal_error to be False"
        raise AssertionError(msg)


def test_plugin_fatal_error_exception() -> None:
    """PluginFatalError preserves execution record and original message.

    Raises
    ------
    AssertionError
        If record or message is not preserved.
    """
    record = GraphPluginRunRecord(
        name="fatal_test",
        status="failed",
        started_at=datetime.now(tz=UTC).isoformat(),
        ended_at=datetime.now(tz=UTC).isoformat(),
        duration_ms=50.0,
        error="test error",
    )
    original = ValueError("Original error")
    exc = PluginFatalError(record, original)

    if exc.record.name != "fatal_test":
        msg = f"Expected record name 'fatal_test', got '{exc.record.name}'"
        raise AssertionError(msg)
    if "Original error" not in str(exc):
        msg = "Expected original error in exception message"
        raise AssertionError(msg)


def test_run_graph_plugins_manifest_update(tmp_path: Path) -> None:
    """Successful plugins update the manifest with execution metadata.

    Raises
    ------
    AssertionError
        If manifest is not updated correctly.
    """
    gateway, snapshot = _make_context(tmp_path)
    try:
        plugin = _make_test_plugin("manifest_plugin", succeed=True)

        with _PluginRegistrar([plugin]):
            context = GraphPlanContext(
                runtime_snapshot=snapshot,
                policy=GraphPluginPolicy(),
            )
            plan = plan_graph_plugin_run(
                plugin_names=[plugin.metadata.name],
                context=context,
            )

            executor_context = GraphExecutorContext(
                gateway=gateway,
                snapshot=snapshot,
            )

            report = run_graph_plugins(plan=plan, context=executor_context)

            if plugin.metadata.name not in report.manifest:
                msg = f"Expected plugin '{plugin.metadata.name}' in manifest"
                raise AssertionError(msg)
            manifest_entry = report.manifest[plugin.metadata.name]
            if "executed_at" not in manifest_entry:
                msg = "Expected 'executed_at' in manifest entry"
                raise AssertionError(msg)
    finally:
        gateway.close()


def test_run_graph_plugins_skip_on_error_severity(tmp_path: Path) -> None:
    """Skip_on_error severity skips plugin on exception.

    Raises
    ------
    AssertionError
        If plugin is not skipped correctly.
    """
    gateway, snapshot = _make_context(tmp_path)
    try:
        plugin = _make_test_plugin("skip_error_plugin", raise_exception=ValueError)

        with _PluginRegistrar([plugin]):
            context = GraphPlanContext(
                runtime_snapshot=snapshot,
                policy=GraphPluginPolicy(
                    default_severity="skip_on_error",
                ),
            )
            plan = plan_graph_plugin_run(
                plugin_names=[plugin.metadata.name],
                context=context,
            )

            executor_context = GraphExecutorContext(
                gateway=gateway,
                snapshot=snapshot,
            )

            report = run_graph_plugins(plan=plan, context=executor_context)

            if report.skip_count != 1:
                msg = f"Expected 1 skip, got {report.skip_count}"
                raise AssertionError(msg)
            if report.failure_count != 0:
                msg = f"Expected 0 failures (should be skipped), got {report.failure_count}"
                raise AssertionError(msg)
    finally:
        gateway.close()
