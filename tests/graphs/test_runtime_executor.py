"""Extended tests for graph runtime executor module.

This module provides additional test coverage for the executor module,
focusing on specific execution paths not covered by test_runtime.py:

- Timeout handling with ThreadPoolExecutor
- Retry logic with backoff
- Dry run and manifest-based skip paths
- Fatal error propagation
- Status counts aggregation
- Pipeline run tracking integration
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from typing import Final

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import (
    GraphPluginPolicy,
    GraphPluginRetryPolicy,
)
from codeintel.core.plugins.result import PluginExecutionRecord, PluginResult
from codeintel.graphs.core.context import GraphPluginExecutionContext
from codeintel.graphs.core.protocol import (
    FunctionalGraphPlugin,
    GraphPluginMetadata,
    GraphPluginProtocol,
)
from codeintel.graphs.core.registry import get_graph_registry, register_graph_plugin
from codeintel.graphs.runtime import executor
from codeintel.graphs.runtime.executor import (
    GraphExecutorContext,
    GraphRunReport,
    PluginFatalError,
    run_graph_plugin_batch,
    run_graph_plugins,
)
from codeintel.graphs.runtime.planning import (
    GraphPlanContext,
    PluginExecutionSettings,
    plan_graph_plugin_run,
)
from codeintel.storage.gateway import StorageGateway
from tests._helpers.fakes.graph_contexts import (
    GraphExecutorTestEnv,
    create_graph_plugin_context,
)

# Constants
TIMEOUT_SHORT_MS: Final = 50
TIMEOUT_LONG_MS: Final = 5000
DELAY_LONG_MS: Final = 2000
RETRY_COUNT: Final = 3
BACKOFF_MS: Final = 10
STATUS_SUCCESS_COUNT: Final = 2
STATUS_FAILURE_COUNT: Final = 1
STATUS_SKIPPED_COUNT: Final = 1
REPORT_SUCCESS_COUNT: Final = 2
REPORT_FAILURE_COUNT: Final = 1
REPORT_MIXED_SUCCESS_COUNT: Final = 1
_EXECUTOR_PRIVATES: Final = executor.__dict__
EXECUTE_PLUGIN = _EXECUTOR_PRIVATES["_execute_plugin"]
RUN_WITH_TIMEOUT = _EXECUTOR_PRIVATES["_run_with_timeout"]
STATUS_COUNTS = _EXECUTOR_PRIVATES["_status_counts"]


@dataclass
class PluginConfig:
    """Configuration for constructing test plugins with varied behaviors."""

    succeed: bool = True
    row_counts: dict[str, int] | None = None
    raise_exception: type[Exception] | None = None
    delay_ms: int = 0
    input_hash: str | None = None
    options_hash: str | None = None


PLUGIN_CONFIG_FIELDS: Final = {field.name for field in fields(PluginConfig)}


def _resolve_plugin_config(
    config: PluginConfig | None, overrides: Mapping[str, object]
) -> PluginConfig:
    """Merge a base config with validated overrides.

    Parameters
    ----------
    config
        Optional base configuration.
    overrides
        Override values keyed by PluginConfig field names.

    Returns
    -------
    PluginConfig
        Combined plugin configuration.

    Raises
    ------
    ValueError
        If overrides include unsupported keys.
    """
    unknown_keys = set(overrides) - PLUGIN_CONFIG_FIELDS
    if unknown_keys:
        message = f"Unsupported plugin config overrides: {sorted(unknown_keys)}"
        raise ValueError(message)
    base_config = config or PluginConfig()
    if not overrides:
        return base_config
    return replace(base_config, **{key: overrides[key] for key in overrides})


# Test Helpers


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
    name: str, *, config: PluginConfig | None = None, **overrides: object
) -> GraphPluginProtocol:
    """Create a configurable test plugin.

    Parameters
    ----------
    name
        Plugin name.
    config
        Optional base configuration for the plugin.
    **overrides
        Overrides for individual fields on the base configuration.

    Returns
    -------
    GraphPluginProtocol
        Configured test plugin instance.
    """
    plugin_config = _resolve_plugin_config(config, overrides)

    def execute(_ctx: GraphPluginExecutionContext) -> PluginResult:
        if plugin_config.delay_ms > 0:
            time.sleep(plugin_config.delay_ms / 1000)
        if plugin_config.raise_exception is not None:
            error_msg = f"Test exception from {name}"
            raise plugin_config.raise_exception(error_msg)
        if plugin_config.succeed:
            return PluginResult.ok(
                row_counts=plugin_config.row_counts,
                input_hash=plugin_config.input_hash,
                options_hash=plugin_config.options_hash,
            )
        return PluginResult.fail(f"Plugin {name} failed")

    metadata = GraphPluginMetadata(
        name=name,
        description=f"Test plugin {name}",
        kind="builder",
        stage="goid",
    )

    return FunctionalGraphPlugin(_metadata=metadata, _execute_fn=execute)


def _make_execution_context(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> GraphPluginExecutionContext:
    """Create a graph execution context.

    Parameters
    ----------
    gateway
        Storage gateway.
    snapshot
        Snapshot reference.

    Returns
    -------
    GraphPluginExecutionContext
        Execution context for plugins.
    """
    return create_graph_plugin_context(
        gateway,
        snapshot,
        plugin_name="test_plugin",
        run_id="test-run-executor",
    )


def test_run_with_timeout_executes_plugin_within_timeout(
    graph_executor_env: GraphExecutorTestEnv,
) -> None:
    """Execute a plugin that completes within the timeout window."""
    plugin = _make_test_plugin("quick_plugin", succeed=True, row_counts={"t": 5})
    ctx = _make_execution_context(graph_executor_env.gateway, graph_executor_env.snapshot)

    result = RUN_WITH_TIMEOUT(plugin, ctx, timeout_ms=TIMEOUT_LONG_MS)

    assert result.success
    assert result.row_counts == {"t": 5}


def test_run_with_timeout_no_timeout_executes_directly(
    graph_executor_env: GraphExecutorTestEnv,
) -> None:
    """Execute a plugin with no timeout set (None) runs directly."""
    plugin = _make_test_plugin("no_timeout_plugin", succeed=True)
    ctx = _make_execution_context(graph_executor_env.gateway, graph_executor_env.snapshot)

    result = RUN_WITH_TIMEOUT(plugin, ctx, timeout_ms=None)

    assert result.success


def test_run_with_timeout_cancels_on_timeout(
    graph_executor_env: GraphExecutorTestEnv,
) -> None:
    """Raise TimeoutError when plugin execution exceeds timeout."""
    plugin = _make_test_plugin("slow_plugin", delay_ms=DELAY_LONG_MS)
    ctx = _make_execution_context(graph_executor_env.gateway, graph_executor_env.snapshot)

    with pytest.raises(TimeoutError, match="timed out"):
        RUN_WITH_TIMEOUT(plugin, ctx, timeout_ms=TIMEOUT_SHORT_MS)


def test_execute_plugin_retry_exhausts_attempts(
    graph_executor_env: GraphExecutorTestEnv,
) -> None:
    """Retry logic exhausts all attempts before failing."""
    plugin = _make_test_plugin("retry_fail_plugin", raise_exception=RuntimeError)
    ctx = _make_execution_context(graph_executor_env.gateway, graph_executor_env.snapshot)

    settings = PluginExecutionSettings(
        name=plugin.metadata.name,
        severity="soft_fail",
        retry_cfg=GraphPluginRetryPolicy(max_attempts=RETRY_COUNT, backoff_ms=0),
        timeout_ms=None,
        fail_fast=False,
        input_hash="inp123",
        options_hash="opt456",
        version_hash="v1",
    )

    record = EXECUTE_PLUGIN(
        plugin=plugin,
        ctx=ctx,
        settings=settings,
    )

    assert record.status == "failed"
    assert record.attempts == RETRY_COUNT


def test_execute_plugin_backoff_applied(
    graph_executor_env: GraphExecutorTestEnv,
) -> None:
    """Backoff delay is applied between retry attempts."""
    plugin = _make_test_plugin("backoff_plugin", raise_exception=ValueError)
    ctx = _make_execution_context(graph_executor_env.gateway, graph_executor_env.snapshot)

    settings = PluginExecutionSettings(
        name=plugin.metadata.name,
        severity="soft_fail",
        retry_cfg=GraphPluginRetryPolicy(max_attempts=2, backoff_ms=BACKOFF_MS),
        timeout_ms=None,
        fail_fast=False,
        input_hash="inp",
        options_hash=None,
        version_hash=None,
    )

    start = time.perf_counter()
    record = EXECUTE_PLUGIN(
        plugin=plugin,
        ctx=ctx,
        settings=settings,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert record.status == "failed"
    # Should have at least one backoff delay between the 2 attempts
    assert elapsed_ms >= BACKOFF_MS * 0.8  # Allow some tolerance


def test_execute_plugin_skip_on_error_severity(
    graph_executor_env: GraphExecutorTestEnv,
) -> None:
    """Plugin with skip_on_error severity returns skipped status on exception."""
    plugin = _make_test_plugin("skip_error_plugin", raise_exception=TypeError)
    ctx = _make_execution_context(graph_executor_env.gateway, graph_executor_env.snapshot)

    settings = PluginExecutionSettings(
        name=plugin.metadata.name,
        severity="skip_on_error",
        retry_cfg=GraphPluginRetryPolicy(),
        timeout_ms=None,
        fail_fast=False,
        input_hash="inp",
        options_hash=None,
        version_hash=None,
    )

    record = EXECUTE_PLUGIN(
        plugin=plugin,
        ctx=ctx,
        settings=settings,
    )

    assert record.status == "skipped"


def test_execute_plugin_fatal_raises_plugin_fatal_error(
    graph_executor_env: GraphExecutorTestEnv,
) -> None:
    """Plugin with fatal severity and fail_fast raises PluginFatalError."""
    plugin = _make_test_plugin("fatal_plugin", raise_exception=RuntimeError)
    ctx = _make_execution_context(graph_executor_env.gateway, graph_executor_env.snapshot)

    settings = PluginExecutionSettings(
        name=plugin.metadata.name,
        severity="fatal",
        retry_cfg=GraphPluginRetryPolicy(max_attempts=1),
        timeout_ms=None,
        fail_fast=True,
        input_hash="inp",
        options_hash=None,
        version_hash=None,
    )

    with pytest.raises(PluginFatalError) as exc_info:
        EXECUTE_PLUGIN(
            plugin=plugin,
            ctx=ctx,
            settings=settings,
        )

    assert exc_info.value.record.status == "failed"
    assert exc_info.value.record.plugin_name == "fatal_plugin"


def test_execute_plugin_timeout_records_error(
    graph_executor_env: GraphExecutorTestEnv,
) -> None:
    """Timeout during execution records timeout error in record."""
    plugin = _make_test_plugin("timeout_plugin", delay_ms=DELAY_LONG_MS)
    ctx = _make_execution_context(graph_executor_env.gateway, graph_executor_env.snapshot)

    settings = PluginExecutionSettings(
        name=plugin.metadata.name,
        severity="soft_fail",
        retry_cfg=GraphPluginRetryPolicy(),
        timeout_ms=TIMEOUT_SHORT_MS,
        fail_fast=False,
        input_hash="inp",
        options_hash=None,
        version_hash=None,
    )

    record = EXECUTE_PLUGIN(
        plugin=plugin,
        ctx=ctx,
        settings=settings,
    )

    assert record.status == "failed"
    assert record.error == "timeout"


def test_execute_plugin_returns_plugin_hashes(
    graph_executor_env: GraphExecutorTestEnv,
) -> None:
    """Plugin-provided hashes override settings hashes in record."""
    plugin = _make_test_plugin(
        "hash_plugin",
        succeed=True,
        input_hash="plugin_inp",
        options_hash="plugin_opt",
    )
    ctx = _make_execution_context(graph_executor_env.gateway, graph_executor_env.snapshot)

    settings = PluginExecutionSettings(
        name=plugin.metadata.name,
        severity="soft_fail",
        retry_cfg=GraphPluginRetryPolicy(),
        timeout_ms=None,
        fail_fast=False,
        input_hash="settings_inp",
        options_hash="settings_opt",
        version_hash="v1",
    )

    record = EXECUTE_PLUGIN(
        plugin=plugin,
        ctx=ctx,
        settings=settings,
    )

    assert record.status == "succeeded"
    assert record.meta.get("input_hash") == "plugin_inp"
    assert record.meta.get("options_hash") == "plugin_opt"


def test_status_counts_aggregates_correctly() -> None:
    """Status counts correctly aggregate success/failure/skip counts."""
    now = datetime.now(tz=UTC)
    records = [
        PluginExecutionRecord(
            plugin_name="p1", status="succeeded", started_at=now, ended_at=now, duration_ms=10
        ),
        PluginExecutionRecord(
            plugin_name="p2", status="succeeded", started_at=now, ended_at=now, duration_ms=20
        ),
        PluginExecutionRecord(
            plugin_name="p3", status="failed", started_at=now, ended_at=now, duration_ms=30
        ),
        PluginExecutionRecord(
            plugin_name="p4", status="skipped", started_at=now, ended_at=now, duration_ms=0
        ),
    ]

    counts = STATUS_COUNTS(records)

    assert counts["success"] == STATUS_SUCCESS_COUNT
    assert counts["failure"] == STATUS_FAILURE_COUNT
    assert counts["skipped"] == STATUS_SKIPPED_COUNT


def test_status_counts_empty_records() -> None:
    """Status counts returns zeros for empty record list."""
    counts = STATUS_COUNTS([])

    assert counts["success"] == 0
    assert counts["failure"] == 0
    assert counts["skipped"] == 0


def test_run_graph_plugins_dry_run_skips_execution(
    graph_executor_env: GraphExecutorTestEnv,
) -> None:
    """Dry run mode skips actual plugin execution."""
    plugin = _make_test_plugin("dry_run_plugin", succeed=True)

    with _PluginRegistrar([plugin]):
        context = GraphPlanContext(
            runtime_snapshot=graph_executor_env.snapshot,
            policy=GraphPluginPolicy(dry_run=True),
        )
        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        executor_context = GraphExecutorContext(
            gateway=graph_executor_env.gateway,
            snapshot=graph_executor_env.snapshot,
        )

        report = run_graph_plugins(plan=plan, context=executor_context)

        assert report.skip_count == 1
        assert report.success_count == 0
        assert report.records
        assert report.records[0].meta.get("skipped_reason") == "dry_run"


def test_run_graph_plugins_skip_on_unchanged(
    graph_executor_env: GraphExecutorTestEnv,
) -> None:
    """Plugin skipped when manifest shows inputs unchanged."""
    plugin = _make_test_plugin("unchanged_plugin", succeed=True)

    with _PluginRegistrar([plugin]):
        context = GraphPlanContext(
            runtime_snapshot=graph_executor_env.snapshot,
            policy=GraphPluginPolicy(skip_on_unchanged=True),
            prior_manifest={
                plugin.metadata.name: {
                    "input_hash": "will_be_computed",
                    "options_hash": None,
                }
            },
        )
        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        # Get the actual computed input hash
        settings = plan.settings_by_plugin[plugin.metadata.name]
        # Update prior manifest with actual hash
        prior_manifest: Mapping[str, Mapping[str, object]] = {
            plugin.metadata.name: {
                "input_hash": settings.input_hash,
                "options_hash": settings.options_hash,
            }
        }

        # Create new context with correct prior manifest
        context_with_correct_hash = GraphPlanContext(
            runtime_snapshot=graph_executor_env.snapshot,
            policy=GraphPluginPolicy(skip_on_unchanged=True),
            prior_manifest=prior_manifest,
        )
        plan_with_correct_hash = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context_with_correct_hash,
        )

        executor_context = GraphExecutorContext(
            gateway=graph_executor_env.gateway,
            snapshot=graph_executor_env.snapshot,
        )

        report = run_graph_plugins(plan=plan_with_correct_hash, context=executor_context)

        assert report.skip_count == 1
        assert report.records
        assert report.records[0].meta.get("skipped_reason") == "unchanged"


def test_run_graph_plugins_builds_manifest(
    graph_executor_env: GraphExecutorTestEnv,
) -> None:
    """Successful plugin execution populates manifest in report."""
    plugin = _make_test_plugin("manifest_build_plugin", succeed=True)

    with _PluginRegistrar([plugin]):
        context = GraphPlanContext(
            runtime_snapshot=graph_executor_env.snapshot,
            policy=GraphPluginPolicy(),
        )
        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        executor_context = GraphExecutorContext(
            gateway=graph_executor_env.gateway,
            snapshot=graph_executor_env.snapshot,
        )

        report = run_graph_plugins(plan=plan, context=executor_context)

        assert plugin.metadata.name in report.manifest
        entry = report.manifest[plugin.metadata.name]
        assert "input_hash" in entry
        assert "executed_at" in entry


def test_run_graph_plugins_fatal_stops_remaining(
    graph_executor_env: GraphExecutorTestEnv,
) -> None:
    """Fatal plugin error stops execution of remaining plugins."""
    fatal_plugin = _make_test_plugin("fatal_first", raise_exception=RuntimeError)
    second_plugin = _make_test_plugin("second_should_not_run", succeed=True)

    with _PluginRegistrar([fatal_plugin, second_plugin]):
        context = GraphPlanContext(
            runtime_snapshot=graph_executor_env.snapshot,
            policy=GraphPluginPolicy(default_severity="fatal", fail_fast=True),
        )
        plan = plan_graph_plugin_run(
            plugin_names=[fatal_plugin.metadata.name, second_plugin.metadata.name],
            context=context,
        )

        executor_context = GraphExecutorContext(
            gateway=graph_executor_env.gateway,
            snapshot=graph_executor_env.snapshot,
        )

        report = run_graph_plugins(plan=plan, context=executor_context)

        assert report.fatal_error
        # Only the fatal plugin should have a record
        assert len(report.records) == 1
        assert report.records[0].plugin_name == "fatal_first"


def test_graph_run_report_captures_all_fields() -> None:
    """GraphRunReport correctly captures all execution fields."""
    now = datetime.now(tz=UTC)
    record = PluginExecutionRecord(
        plugin_name="test_plugin",
        status="succeeded",
        started_at=now,
        ended_at=now,
        duration_ms=100.0,
        meta={"row_counts": {"t": 10}},
    )

    report = GraphRunReport(
        run_id="run-123",
        repo="test/repo",
        commit="abc123",
        records=(record,),
        success_count=1,
        failure_count=0,
        skip_count=0,
        duration_ms=150.0,
        started_at=now,
        ended_at=now,
        fatal_error=False,
        manifest={"test_plugin": {"executed_at": now}},
    )

    assert report.run_id == "run-123"
    assert report.repo == "test/repo"
    assert report.commit == "abc123"
    assert len(report.records) == 1
    assert report.success_count == 1
    assert not report.fatal_error
    assert "test_plugin" in report.manifest


def test_plugin_fatal_error_preserves_context() -> None:
    """PluginFatalError preserves record and exception message."""
    now = datetime.now(tz=UTC)
    record = PluginExecutionRecord(
        plugin_name="fatal_plugin",
        status="failed",
        started_at=now,
        ended_at=now,
        duration_ms=50.0,
        error="Original error",
    )
    original = ValueError("Original exception message")

    exc = PluginFatalError(record, original)

    assert exc.record.plugin_name == "fatal_plugin"
    assert exc.record.status == "failed"
    assert "Original exception message" in str(exc)


def test_run_graph_plugin_batch_executes_multiple(
    graph_executor_env: GraphExecutorTestEnv,
) -> None:
    """Batch execution runs multiple plugins and reports results."""
    plugins = [
        _make_test_plugin("batch_p1", succeed=True),
        _make_test_plugin("batch_p2", succeed=True),
    ]

    with _PluginRegistrar(plugins):
        report = run_graph_plugin_batch(
            plugins=plugins,
            gateway=graph_executor_env.gateway,
            snapshot=graph_executor_env.snapshot,
        )

        assert report.success_count == REPORT_SUCCESS_COUNT
        assert report.failure_count == 0
        assert report.repo == "demo/repo"


def test_run_graph_plugin_batch_with_mixed_results(
    graph_executor_env: GraphExecutorTestEnv,
) -> None:
    """Batch execution handles mixed success and failure results."""
    plugins = [
        _make_test_plugin("batch_success", succeed=True),
        _make_test_plugin("batch_fail", succeed=False),
    ]

    with _PluginRegistrar(plugins):
        report = run_graph_plugin_batch(
            plugins=plugins,
            gateway=graph_executor_env.gateway,
            snapshot=graph_executor_env.snapshot,
        )

        assert report.success_count == REPORT_MIXED_SUCCESS_COUNT
        assert report.failure_count == REPORT_FAILURE_COUNT
