"""Consolidated tests for graph runtime modules.

This module tests the graph plugin runtime infrastructure including:

- Execution (plugin batch execution, timeout handling, retry logic)
- Planning (plan generation, dependency resolution, policy application)
- Manifest (skip detection based on input hashes)
- Telemetry (span management, metric recording, OpenTelemetry integration)
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import (
    GraphPluginPolicy,
    GraphPluginRetryPolicy,
    GraphRunScope,
)
from codeintel.graphs.core.context import GraphExecutionContext, GraphRuntimeScratch
from codeintel.graphs.core.protocol import (
    FunctionalGraphPlugin,
    GraphPluginMetadata,
    GraphPluginProtocol,
    GraphPluginSeverity,
)
from codeintel.graphs.core.registry import get_graph_registry, register_graph_plugin
from codeintel.graphs.core.result import GraphPluginResult, GraphPluginRunRecord
from codeintel.graphs.resources.container import ResourceContainer
from codeintel.graphs.runtime.executor import (
    GraphExecutorContext,
    GraphRunReport,
    PluginFatalError,
    run_graph_plugin_batch,
    run_graph_plugins,
)
from codeintel.graphs.runtime.manifest import (
    GraphPluginManifest,
    InputHashPayload,
    ManifestState,
    RecordParams,
    compute_input_hash,
    compute_options_hash,
    dry_run_record,
    is_unchanged,
    skip_record,
)
from codeintel.graphs.runtime.planning import (
    GraphPlanContext,
    GraphPluginRunOptions,
    plan_graph_plugin_run,
)
from codeintel.graphs.runtime.telemetry import (
    GraphPluginSpan,
    GraphRuntimeTelemetry,
    get_graph_telemetry,
)
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.schemas import apply_all_schemas
from tests._helpers.gateway import open_ingestion_gateway_with_macros

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_PLUGIN_COUNT: Final = 3
EXPECTED_BATCH_COUNT: Final = 2
EXPECTED_TIMEOUT_MS: Final = 5000
EXPECTED_HASH_LENGTH: Final = 16
EXPECTED_ATTR_VALUE: Final = 42


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------


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


def _make_executor_test_plugin(
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


def _make_planning_test_plugin(
    name: str,
    *,
    depends_on: tuple[str, ...] = (),
    provides: tuple[str, ...] = (),
    severity: GraphPluginSeverity = "fatal",
) -> GraphPluginProtocol:
    """Create a test plugin for planning tests.

    Parameters
    ----------
    name
        Plugin name.
    depends_on
        Plugin dependencies.
    provides
        Capabilities provided.
    severity
        Failure severity.

    Returns
    -------
    GraphPluginProtocol
        Test plugin instance.
    """

    def execute(_ctx: GraphExecutionContext) -> GraphPluginResult:
        return GraphPluginResult.ok()

    metadata = GraphPluginMetadata(
        name=name,
        description=f"Test plugin {name}",
        kind="builder",
        stage="goid",
        depends_on=depends_on,
        provides=provides,
        severity=severity,
    )

    return FunctionalGraphPlugin(_metadata=metadata, _execute_fn=execute)


def _make_telemetry_test_plugin(name: str) -> GraphPluginProtocol:
    """Create a test plugin for telemetry tests.

    Parameters
    ----------
    name
        Plugin name.

    Returns
    -------
    GraphPluginProtocol
        Test plugin instance.
    """

    def execute(_ctx: GraphExecutionContext) -> GraphPluginResult:
        return GraphPluginResult.ok()

    metadata = GraphPluginMetadata(
        name=name,
        description=f"Test plugin {name}",
        kind="builder",
        stage="goid",
    )

    return FunctionalGraphPlugin(_metadata=metadata, _execute_fn=execute)


def _make_executor_context(tmp_path: Path) -> tuple[StorageGateway, SnapshotRef]:
    """Create execution context for executor tests.

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


def _make_telemetry_context() -> tuple[GraphExecutionContext, StorageGateway]:
    """Create a test execution context for telemetry tests.

    Returns
    -------
    tuple
        A tuple of (GraphExecutionContext, gateway).
        Caller is responsible for closing the gateway.
    """
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    apply_all_schemas(gateway.con)
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=Path())
    scratch = GraphRuntimeScratch()
    ctx = GraphExecutionContext(
        snapshot=snapshot,
        resources=ResourceContainer(),
        _gateway=gateway,
        scratch=scratch,
        plugin_name="test_plugin",
        run_id="test-run-123",
    )
    return ctx, gateway


# ===========================================================================
# SECTION 1: Executor Tests
# ===========================================================================


def test_run_graph_plugins_basic_success(tmp_path: Path) -> None:
    """Execute a simple plugin batch successfully."""
    gateway, snapshot = _make_executor_context(tmp_path)
    try:
        plugin = _make_executor_test_plugin(
            "basic_success_plugin", succeed=True, row_counts={"t": 5}
        )

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

            assert report.success_count == 1
            assert report.failure_count == 0
    finally:
        gateway.close()


def test_run_graph_plugins_with_failure(tmp_path: Path) -> None:
    """Plugin failure handling records the failure correctly."""
    gateway, snapshot = _make_executor_context(tmp_path)
    try:
        plugin = _make_executor_test_plugin("failing_plugin", succeed=False)

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

            assert report.failure_count == 1
            assert report.success_count == 0
    finally:
        gateway.close()


def test_run_graph_plugins_timeout(tmp_path: Path) -> None:
    """Timeout handling produces failed record with timeout error."""
    gateway, snapshot = _make_executor_context(tmp_path)
    try:
        # Use a plugin that delays long enough to trigger timeout
        plugin = _make_executor_test_plugin("slow_plugin", succeed=True, delay_ms=2000)

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

            assert report.failure_count == 1
            assert report.records
            rec = report.records[0]
            assert rec.error == "timeout"
    finally:
        gateway.close()


def test_run_graph_plugins_retry_logic(tmp_path: Path) -> None:
    """Retry logic executes multiple attempts before failing."""
    gateway, snapshot = _make_executor_context(tmp_path)
    try:
        plugin = _make_executor_test_plugin("retry_plugin", raise_exception=RuntimeError)

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

            assert report.failure_count == 1
            assert report.records
            rec = report.records[0]
            assert rec.attempts == EXPECTED_PLUGIN_COUNT
    finally:
        gateway.close()


def test_run_graph_plugins_fatal_error(tmp_path: Path) -> None:
    """PluginFatalError stops execution and is recorded."""
    gateway, snapshot = _make_executor_context(tmp_path)
    try:
        # First plugin raises exception with fatal severity + fail_fast
        fatal_plugin = _make_executor_test_plugin("fatal_plugin", raise_exception=ValueError)
        # Second plugin should not execute
        second_plugin = _make_executor_test_plugin("second_plugin", succeed=True)

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

            assert report.fatal_error
            # Only one record should exist (fatal stopped execution)
            assert len(report.records) == 1
    finally:
        gateway.close()


def test_run_graph_plugin_batch_convenience(tmp_path: Path) -> None:
    """run_graph_plugin_batch convenience function executes plugins."""
    gateway, snapshot = _make_executor_context(tmp_path)
    try:
        plugins = [
            _make_executor_test_plugin("batch_plugin_1", succeed=True),
            _make_executor_test_plugin("batch_plugin_2", succeed=True),
        ]

        with _PluginRegistrar(plugins):
            report = run_graph_plugin_batch(
                plugins=plugins,
                gateway=gateway,
                snapshot=snapshot,
            )

            assert report.success_count == EXPECTED_BATCH_COUNT
            assert report.repo == "demo/repo"
    finally:
        gateway.close()


def test_run_graph_plugins_multiple_success(tmp_path: Path) -> None:
    """Multiple successful plugins all complete and are recorded."""
    gateway, snapshot = _make_executor_context(tmp_path)
    try:
        plugins = [_make_executor_test_plugin(f"multi_plugin_{i}", succeed=True) for i in range(3)]

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

            assert report.success_count == EXPECTED_PLUGIN_COUNT
            assert len(report.records) == EXPECTED_PLUGIN_COUNT
    finally:
        gateway.close()


def test_run_graph_plugins_includes_timing(tmp_path: Path) -> None:
    """Report includes duration and timestamps."""
    gateway, snapshot = _make_executor_context(tmp_path)
    try:
        plugin = _make_executor_test_plugin("timing_plugin", succeed=True)

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

            assert report.duration_ms >= 0
            assert report.started_at
            assert report.ended_at
            # Validate ISO format parses correctly
            datetime.fromisoformat(report.started_at)
            datetime.fromisoformat(report.ended_at)
    finally:
        gateway.close()


def test_graph_run_report_attributes() -> None:
    """GraphRunReport captures all execution attributes correctly."""
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

    assert report.run_id == "run-123"
    assert len(report.records) == 1
    assert not report.fatal_error


def test_plugin_fatal_error_exception() -> None:
    """PluginFatalError preserves execution record and original message."""
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

    assert exc.record.name == "fatal_test"
    assert "Original error" in str(exc)


def test_run_graph_plugins_manifest_update(tmp_path: Path) -> None:
    """Successful plugins update the manifest with execution metadata."""
    gateway, snapshot = _make_executor_context(tmp_path)
    try:
        plugin = _make_executor_test_plugin("manifest_plugin", succeed=True)

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

            assert plugin.metadata.name in report.manifest
            manifest_entry = report.manifest[plugin.metadata.name]
            assert "executed_at" in manifest_entry
    finally:
        gateway.close()


def test_run_graph_plugins_skip_on_error_severity(tmp_path: Path) -> None:
    """Skip_on_error severity skips plugin on exception."""
    gateway, snapshot = _make_executor_context(tmp_path)
    try:
        plugin = _make_executor_test_plugin("skip_error_plugin", raise_exception=ValueError)

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

            assert report.skip_count == 1
            assert report.failure_count == 0
    finally:
        gateway.close()


# ===========================================================================
# SECTION 2: Planning Tests
# ===========================================================================


def test_plan_graph_plugin_run_basic(tmp_path: Path) -> None:
    """Basic plan generation produces valid execution plan."""
    plugin = _make_planning_test_plugin("basic_plan_plugin")
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)

    with _PluginRegistrar([plugin]):
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(),
        )
        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        assert plan.plan_id
        assert plan.run_id
        assert plan.repo == "demo/repo"
        assert plan.commit == "deadbeef"
        assert len(plan.plugins) == 1


def test_plan_with_dependencies() -> None:
    """Dependency resolution orders plugins correctly."""
    # Plugin B depends on Plugin A
    plugin_a = _make_planning_test_plugin("dep_a", provides=("capability_a",))
    plugin_b = _make_planning_test_plugin("dep_b", depends_on=("dep_a",))

    with _PluginRegistrar([plugin_a, plugin_b]):
        snapshot = SnapshotRef(repo="demo/repo", commit="abc123", repo_root=Path())
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(),
        )
        plan = plan_graph_plugin_run(
            plugin_names=["dep_a", "dep_b"],
            context=context,
        )

        # A should come before B in ordered names
        names = plan.ordered_names
        assert "dep_a" in names
        assert "dep_b" in names
        assert names.index("dep_a") < names.index("dep_b")


def test_plan_with_custom_policy() -> None:
    """Custom policy settings are applied to plan."""
    plugin = _make_planning_test_plugin("policy_plugin")

    with _PluginRegistrar([plugin]):
        snapshot = SnapshotRef(repo="demo/repo", commit="abc123", repo_root=Path())
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(
                default_severity="soft_fail",
                fail_fast=False,
                timeouts_ms={"policy_plugin": EXPECTED_TIMEOUT_MS},
                retries={"policy_plugin": GraphPluginRetryPolicy(max_attempts=3)},
            ),
        )
        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        settings = plan.settings_by_plugin.get(plugin.metadata.name)
        assert settings is not None
        assert settings.severity == "soft_fail"
        assert settings.timeout_ms == EXPECTED_TIMEOUT_MS
        assert settings.retry_cfg.max_attempts == EXPECTED_PLUGIN_COUNT


def test_plugin_execution_settings_hashes() -> None:
    """Execution settings include computed hashes."""
    plugin = _make_planning_test_plugin("hash_plugin")

    with _PluginRegistrar([plugin]):
        snapshot = SnapshotRef(repo="demo/repo", commit="abc123", repo_root=Path())
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(),
        )
        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        settings = plan.settings_by_plugin.get(plugin.metadata.name)
        assert settings is not None
        assert settings.input_hash


def test_plan_with_explicit_target() -> None:
    """Plan can use explicit target tuple instead of snapshot."""
    plugin = _make_planning_test_plugin("target_plugin")

    with _PluginRegistrar([plugin]):
        context = GraphPlanContext(
            target=("explicit/repo", "explicit_commit"),
            policy=GraphPluginPolicy(),
        )
        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        assert plan.repo == "explicit/repo"
        assert plan.commit == "explicit_commit"


def test_plan_missing_target_raises() -> None:
    """Plan without any target source raises ValueError."""
    plugin = _make_planning_test_plugin("no_target_plugin")

    with _PluginRegistrar([plugin]):
        context = GraphPlanContext(
            policy=GraphPluginPolicy(),
        )
        with pytest.raises(ValueError, match="missing snapshot"):
            plan_graph_plugin_run(
                plugin_names=[plugin.metadata.name],
                context=context,
            )


def test_plan_with_run_options() -> None:
    """Runtime options override config settings."""
    plugin = _make_planning_test_plugin("options_plugin")

    with _PluginRegistrar([plugin]):
        snapshot = SnapshotRef(repo="demo/repo", commit="abc123", repo_root=Path())
        run_options = GraphPluginRunOptions(
            scope=GraphRunScope(paths=("src/",), modules=("mymodule",)),
        )
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(),
            run_options=run_options,
        )
        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

    assert plan.scope.paths == ("src/",)


# ===========================================================================
# SECTION 3: Manifest Tests
# ===========================================================================


def test_compute_input_hash_deterministic() -> None:
    """Input hash is deterministic for same inputs."""
    payload = InputHashPayload(
        repo="demo/repo",
        commit="abc123",
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(),
        options_hash="opt123",
    )

    hash1 = compute_input_hash(payload)
    hash2 = compute_input_hash(payload)

    assert hash1 == hash2
    assert len(hash1) == EXPECTED_HASH_LENGTH


def test_compute_input_hash_varies_with_inputs() -> None:
    """Input hash varies when inputs change."""
    payload1 = InputHashPayload(
        repo="demo/repo",
        commit="abc123",
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(),
        options_hash="opt123",
    )
    payload2 = InputHashPayload(
        repo="demo/repo",
        commit="different_commit",  # Changed
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(),
        options_hash="opt123",
    )

    hash1 = compute_input_hash(payload1)
    hash2 = compute_input_hash(payload2)

    assert hash1 != hash2


def test_compute_options_hash_with_options() -> None:
    """Options hash is computed when options are provided."""
    plugin = _make_planning_test_plugin("opt_hash_plugin")
    options = {"key": "value", "number": 42}

    hash_val = compute_options_hash(plugin, options)

    assert hash_val is not None
    assert len(hash_val) == EXPECTED_HASH_LENGTH


def test_compute_options_hash_none_returns_none() -> None:
    """Options hash is None when options are None."""
    plugin = _make_planning_test_plugin("no_opt_plugin")

    hash_val = compute_options_hash(plugin, None)

    assert hash_val is None


def test_is_unchanged_when_hashes_match() -> None:
    """Skip detection returns True when hashes match."""
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    try:
        apply_all_schemas(gateway.con)

        prior_manifest = {
            "test_plugin": {
                "input_hash": "abc123",
                "options_hash": "opt456",
            }
        }

        state = ManifestState(
            plugin_name="test_plugin",
            row_count_tables=(),
            gateway=gateway,
            repo="demo/repo",
            commit="deadbeef",
            input_hash="abc123",
            options_hash="opt456",
        )

        result = is_unchanged(prior_manifest, state)

        assert result
    finally:
        gateway.close()


def test_is_unchanged_when_hashes_differ() -> None:
    """Skip detection returns False when hashes differ."""
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    try:
        apply_all_schemas(gateway.con)

        prior_manifest = {
            "test_plugin": {
                "input_hash": "old_hash",
                "options_hash": "opt456",
            }
        }

        state = ManifestState(
            plugin_name="test_plugin",
            row_count_tables=(),
            gateway=gateway,
            repo="demo/repo",
            commit="deadbeef",
            input_hash="new_hash",  # Different
            options_hash="opt456",
        )

        result = is_unchanged(prior_manifest, state)

        assert not result
    finally:
        gateway.close()


def test_is_unchanged_no_prior_manifest() -> None:
    """Skip detection returns False when no prior manifest."""
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    try:
        apply_all_schemas(gateway.con)

        state = ManifestState(
            plugin_name="test_plugin",
            row_count_tables=(),
            gateway=gateway,
            repo="demo/repo",
            commit="deadbeef",
            input_hash="abc123",
            options_hash="opt456",
        )

        result = is_unchanged(None, state)

        assert not result
    finally:
        gateway.close()


def test_dry_run_record() -> None:
    """Dry run mode produces skipped record with correct reason."""
    plugin = _make_planning_test_plugin("dry_run_plugin")
    params = RecordParams(
        severity="soft_fail",
        timeout_ms=1000,
        version_hash="v1",
        input_hash="inp123",
        options_hash="opt456",
        options=None,
    )

    record = dry_run_record(plugin=plugin, params=params)

    assert record.status == "skipped"
    assert record.meta.get("skipped_reason") == "dry_run"
    assert record.name == "dry_run_plugin"


def test_skip_record() -> None:
    """Skip record includes reason and metadata."""
    plugin = _make_planning_test_plugin("skip_plugin")
    params = RecordParams(
        severity="soft_fail",
        timeout_ms=1000,
        version_hash="v1",
        input_hash="inp123",
        options_hash="opt456",
        options=None,
    )

    record = skip_record(plugin=plugin, params=params, reason="unchanged")

    assert record.status == "skipped"
    assert record.meta.get("skipped_reason") == "unchanged"


def test_graph_plugin_manifest_record() -> None:
    """Manifest records execution metadata correctly."""
    manifest = GraphPluginManifest()

    manifest.record(
        plugin_name="test_plugin",
        input_hash="inp123",
        options_hash="opt456",
        version_hash="v1",
        row_counts={"table1": 100},
    )

    entries = manifest.to_dict()
    assert "test_plugin" in entries

    entry = entries["test_plugin"]
    assert entry.get("input_hash") == "inp123"
    assert entry.get("row_counts") == {"table1": 100}


def test_record_params_defaults() -> None:
    """RecordParams has correct defaults."""
    params = RecordParams(
        severity="soft_fail",
        timeout_ms=None,
        version_hash=None,
        input_hash=None,
        options_hash=None,
        options=None,
    )

    assert not params.requires_isolation
    assert params.policy_fail_fast is True


# ===========================================================================
# SECTION 4: Telemetry Tests
# ===========================================================================


def test_graph_runtime_telemetry_initialization() -> None:
    """Telemetry initializes without raising errors."""
    # Creating telemetry should not raise
    telemetry = GraphRuntimeTelemetry(
        service_name="test.service",
        enable_tracing=False,
        enable_metrics=False,
    )

    # Verify it's a valid instance
    assert telemetry is not None


def test_start_plugin_creates_span() -> None:
    """Starting a plugin creates a telemetry span with correct attributes."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )
    plugin = _make_telemetry_test_plugin("span_test_plugin")
    ctx, gateway = _make_telemetry_context()

    try:
        span = telemetry.start_plugin(plugin, "run-123", ctx)

        assert span.plugin_name == "span_test_plugin"
        assert span.run_id == "run-123"
        assert span.start_time_ns > 0
        assert span.attributes.get("plugin.name") == "span_test_plugin"
        assert span.attributes.get("repo") == "demo/repo"
    finally:
        gateway.close()


def test_finish_plugin_records_duration() -> None:
    """Finishing a plugin span completes without error."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )
    plugin = _make_telemetry_test_plugin("duration_test_plugin")
    ctx, gateway = _make_telemetry_context()

    try:
        span = telemetry.start_plugin(plugin, "run-123", ctx)

        # Simulate some execution time
        time.sleep(0.01)

        record = GraphPluginRunRecord(
            name="duration_test_plugin",
            status="succeeded",
            started_at=datetime.now(tz=UTC).isoformat(),
            ended_at=datetime.now(tz=UTC).isoformat(),
            duration_ms=10.0,
        )

        # Should not raise
        telemetry.finish_plugin(span, record)
    finally:
        gateway.close()


def test_start_run_creates_run_span() -> None:
    """Starting a run creates a run-level telemetry span."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )

    span = telemetry.start_run(
        run_id="run-456",
        repo="demo/repo",
        commit="abc123",
        plugin_count=EXPECTED_PLUGIN_COUNT,
    )

    assert span.plugin_name == "__run__"
    assert span.run_id == "run-456"
    assert span.attributes.get("plugin_count") == EXPECTED_PLUGIN_COUNT


def test_finish_run_completes_span() -> None:
    """Finishing a run span completes without error."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )

    span = telemetry.start_run(
        run_id="run-789",
        repo="demo/repo",
        commit="abc123",
        plugin_count=3,
    )

    # Should not raise
    telemetry.finish_run(span, success_count=2, failure_count=1, skip_count=0)


def test_record_metrics_without_otel() -> None:
    """Recording metrics without OpenTelemetry does not raise."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )

    record = GraphPluginRunRecord(
        name="metrics_test",
        status="succeeded",
        started_at=datetime.now(tz=UTC).isoformat(),
        ended_at=datetime.now(tz=UTC).isoformat(),
        duration_ms=100.0,
    )

    # Should not raise even with metrics disabled
    telemetry.record_metrics(record, scope=None)


def test_get_graph_telemetry_singleton() -> None:
    """get_graph_telemetry returns singleton instance."""
    telemetry1 = get_graph_telemetry()
    telemetry2 = get_graph_telemetry()

    assert telemetry1 is telemetry2


def test_graph_plugin_span_attributes() -> None:
    """GraphPluginSpan stores attributes correctly."""
    span = GraphPluginSpan(
        plugin_name="test_plugin",
        run_id="run-123",
        start_time_ns=time.perf_counter_ns(),
        attributes={"key1": "value1", "key2": EXPECTED_ATTR_VALUE},
        context_data={"otel_span": None},
    )

    assert span.plugin_name == "test_plugin"
    assert span.attributes.get("key1") == "value1"
    assert span.attributes.get("key2") == EXPECTED_ATTR_VALUE


def test_telemetry_handles_failed_plugin() -> None:
    """Telemetry correctly handles failed plugin records without error."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )
    plugin = _make_telemetry_test_plugin("failed_plugin")
    ctx, gateway = _make_telemetry_context()

    try:
        span = telemetry.start_plugin(plugin, "run-fail", ctx)

        record = GraphPluginRunRecord(
            name="failed_plugin",
            status="failed",
            started_at=datetime.now(tz=UTC).isoformat(),
            ended_at=datetime.now(tz=UTC).isoformat(),
            duration_ms=50.0,
            error="Test error message",
        )

        # Should handle failed status without raising
        telemetry.finish_plugin(span, record)
    finally:
        gateway.close()


def test_telemetry_handles_skipped_plugin() -> None:
    """Telemetry correctly handles skipped plugin records without error."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )
    plugin = _make_telemetry_test_plugin("skipped_plugin")
    ctx, gateway = _make_telemetry_context()

    try:
        span = telemetry.start_plugin(plugin, "run-skip", ctx)

        record = GraphPluginRunRecord(
            name="skipped_plugin",
            status="skipped",
            started_at=datetime.now(tz=UTC).isoformat(),
            ended_at=datetime.now(tz=UTC).isoformat(),
            duration_ms=0.0,
        )

        # Should handle skipped status without raising
        telemetry.finish_plugin(span, record)
    finally:
        gateway.close()


def test_telemetry_with_multiple_attempts() -> None:
    """Telemetry correctly records retry attempts without error."""
    telemetry = GraphRuntimeTelemetry(
        enable_tracing=False,
        enable_metrics=False,
    )
    plugin = _make_telemetry_test_plugin("retry_plugin")
    ctx, gateway = _make_telemetry_context()

    try:
        span = telemetry.start_plugin(plugin, "run-retry", ctx)

        record = GraphPluginRunRecord(
            name="retry_plugin",
            status="failed",
            started_at=datetime.now(tz=UTC).isoformat(),
            ended_at=datetime.now(tz=UTC).isoformat(),
            duration_ms=150.0,
            attempts=3,
            error="Failed after retries",
        )

        # Should handle multiple attempts without raising
        telemetry.finish_plugin(span, record)
    finally:
        gateway.close()
