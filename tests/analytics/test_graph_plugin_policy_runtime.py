"""Unit tests for graph plugin policies (retries, severity, timeouts)."""

from __future__ import annotations

import time
from importlib import import_module
from pathlib import Path
from typing import Protocol

import pytest

from codeintel.analytics.graph_runtime import GraphRuntimeOptions, resolve_graph_runtime
from codeintel.analytics.graph_service_runtime import (
    GraphPluginRunOptions,
    GraphPluginRunRecord,
    GraphRuntimeTelemetry,
    GraphServiceRuntime,
)
from codeintel.analytics.graphs.contracts import PluginContractResult
from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
    register_graph_metric_plugin,
    unregister_graph_metric_plugin,
)
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.config.steps_graphs import (
    GraphMetricsStepConfig,
    GraphPluginPolicy,
    GraphPluginRetryPolicy,
    GraphRunScope,
)
from codeintel.storage.gateway import open_memory_gateway

RETRY_ATTEMPTS = 2
TIMEOUT_MS = 10
EXPECTED_RECORDS = 2
_FATAL_ERROR = import_module("codeintel.analytics.graph_service_runtime")._PluginFatalError  # noqa: SLF001


def _make_service(
    repo: str = "demo/repo", commit: str = "deadbeef"
) -> tuple[GraphServiceRuntime, GraphMetricsStepConfig]:
    """
    Construct a runtime and config for testing.

    Returns
    -------
    tuple[GraphServiceRuntime, GraphMetricsStepConfig]
        Service and config prepared for tests.
    """
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=Path())
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    runtime = resolve_graph_runtime(
        gateway,
        snapshot,
        GraphRuntimeOptions(snapshot=snapshot, backend=GraphBackendConfig()),
    )
    service = GraphServiceRuntime(gateway=gateway, runtime=runtime)
    cfg = GraphMetricsStepConfig(snapshot=snapshot)
    return service, cfg


def test_soft_fail_severity_continues() -> None:
    """Soft failures should not stop subsequent plugins."""
    attempts: list[int] = []

    def _fails(_ctx: object) -> None:
        attempts.append(1)
        message = "boom"
        raise RuntimeError(message)

    ran_success: list[int] = []

    def _succeeds(_ctx: object) -> None:
        ran_success.append(1)

    failing = GraphMetricPlugin(
        name="soft_fail_plugin",
        description="fails softly",
        stage="core",
        enabled_by_default=False,
        run=_fails,
        severity="soft_fail",
    )
    succeeding = GraphMetricPlugin(
        name="after_soft_fail",
        description="runs after soft failure",
        stage="core",
        enabled_by_default=False,
        run=_succeeds,
    )

    register_graph_metric_plugin(failing)
    register_graph_metric_plugin(succeeding)
    service, cfg = _make_service()
    try:
        report = service.run_plugins(
            ("soft_fail_plugin", "after_soft_fail"),
            cfg=cfg,
        )
    finally:
        unregister_graph_metric_plugin("soft_fail_plugin")
        unregister_graph_metric_plugin("after_soft_fail")

    if len(report.records) != EXPECTED_RECORDS:
        pytest.fail("Expected two plugin records in report")
    first, second = report.records
    if first.status != "failed":
        pytest.fail("Soft-fail plugin should record failed status")
    if first.severity != "soft_fail":
        pytest.fail("Soft-fail plugin should preserve severity in report")
    if not first.partial:
        pytest.fail("Soft-fail plugin should be marked partial")
    if second.status != "succeeded":
        pytest.fail("Subsequent plugin should still run after soft failure")
    if not ran_success:
        pytest.fail("Success plugin should run even after soft failure")
    if not attempts:
        pytest.fail("Failing plugin should have been invoked")


class _TelemetryRecorder(Protocol):
    starts: list[tuple[str, str, GraphRunScope]]
    finishes: list[GraphPluginRunRecord]
    metrics: list[GraphPluginRunRecord]
    scopes: list[GraphRunScope]


def _assert_telemetry_expectations(
    record: GraphPluginRunRecord,
    report_run_id: str,
    telemetry: _TelemetryRecorder,
    scope: GraphRunScope,
) -> None:
    """Validate telemetry hooks captured expected metadata."""
    failures: list[str] = []
    if record.run_id != report_run_id:
        failures.append("Run ID should propagate to report and record")
    if not telemetry.starts or telemetry.starts[0][1] != record.run_id:
        failures.append("Telemetry start should capture run_id")
    if not telemetry.finishes:
        failures.append("Telemetry finish should be recorded")
    if not telemetry.metrics or telemetry.scopes[0] != scope:
        failures.append("Telemetry metrics should include scope")
    if failures:
        pytest.fail("; ".join(failures))


def test_retry_policy_allows_retries() -> None:
    """Plugins should honor retry policy before surfacing failure."""
    call_count = {"n": 0}

    def _flaky(_ctx: object) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            message = "transient"
            raise RuntimeError(message)

    plugin_name = "retry_plugin"
    plugin = GraphMetricPlugin(
        name=plugin_name,
        description="flaky plugin",
        stage="core",
        enabled_by_default=False,
        run=_flaky,
    )
    register_graph_metric_plugin(plugin)
    service, cfg = _make_service()
    cfg = GraphMetricsStepConfig(
        snapshot=cfg.snapshot,
        plugin_policy=GraphPluginPolicy(
            retries={
                plugin_name: GraphPluginRetryPolicy(max_attempts=RETRY_ATTEMPTS, backoff_ms=0)
            },
        ),
    )
    try:
        report = service.run_plugins((plugin_name,), cfg=cfg)
    finally:
        unregister_graph_metric_plugin(plugin_name)
    record = report.records[0]
    if record.status != "succeeded":
        pytest.fail("Flaky plugin should succeed after retry")
    if record.attempts != RETRY_ATTEMPTS:
        pytest.fail("Retry attempts should be recorded")


def test_timeout_marks_partial_and_continues() -> None:
    """Timeouts should mark partial failure but allow downstream plugins."""

    def _slow(_ctx: object) -> None:
        time.sleep(0.1)

    ran_second: list[int] = []

    def _fast(_ctx: object) -> None:
        ran_second.append(1)

    slow_name = "timeout_plugin"
    fast_name = "next_plugin"
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=slow_name,
            description="times out",
            stage="core",
            enabled_by_default=False,
            run=_slow,
            severity="soft_fail",
        )
    )
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=fast_name,
            description="runs after timeout",
            stage="core",
            enabled_by_default=False,
            run=_fast,
        )
    )
    service, cfg = _make_service()
    cfg = GraphMetricsStepConfig(
        snapshot=cfg.snapshot,
        plugin_policy=GraphPluginPolicy(
            timeouts_ms={slow_name: TIMEOUT_MS},
        ),
    )
    try:
        report = service.run_plugins((slow_name, fast_name), cfg=cfg)
    finally:
        unregister_graph_metric_plugin(slow_name)
        unregister_graph_metric_plugin(fast_name)
    first, second = report.records
    if first.status != "failed":
        pytest.fail("Timeout should mark plugin as failed")
    if not first.partial:
        pytest.fail("Timeout should mark partial=True")
    if first.timeout_ms != TIMEOUT_MS:
        pytest.fail("Timeout value should be recorded in manifest")
    if second.status != "succeeded":
        pytest.fail("Subsequent plugin should run after soft timeout failure")
    if not ran_second:
        pytest.fail("Fast plugin should still run after timeout")


def test_contract_failure_marks_plugin_failed_and_continues() -> None:
    """Contract failures should mark plugin failed but not halt when soft."""

    def _noop(_ctx: object) -> None:
        return None

    def _contract_fail(_ctx: object) -> PluginContractResult:
        return PluginContractResult(name="c1", status="failed", message="bad")

    ran_second: list[int] = []

    def _second(_ctx: object) -> None:
        ran_second.append(1)

    first_name = "contract_fail_plugin"
    second_name = "after_contract_fail"
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=first_name,
            description="contract fail",
            stage="core",
            enabled_by_default=False,
            run=_noop,
            contract_checkers=(_contract_fail,),
            severity="soft_fail",
        )
    )
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=second_name,
            description="runs after contract failure",
            stage="core",
            enabled_by_default=False,
            run=_second,
        )
    )
    service, cfg = _make_service()
    try:
        report = service.run_plugins((first_name, second_name), cfg=cfg)
    finally:
        unregister_graph_metric_plugin(first_name)
        unregister_graph_metric_plugin(second_name)
    first, second = report.records
    if first.status != "failed":
        pytest.fail("Contract failure should mark plugin status as failed")
    if not first.partial:
        pytest.fail("Contract failure should mark partial")
    if not first.contracts or first.contracts[0].status != "failed":
        pytest.fail("Contract results should be captured")
    if second.status != "succeeded":
        pytest.fail("Subsequent plugin should continue after soft failure")
    if not ran_second:
        pytest.fail("Second plugin should execute after contract failure")


def test_contract_failure_respects_fatal_fail_fast() -> None:
    """Fatal severity with fail-fast should raise on contract failure."""

    def _noop(_ctx: object) -> None:
        return None

    def _contract_fail(_ctx: object) -> PluginContractResult:
        return PluginContractResult(name="c1", status="failed", message="bad")

    plugin_name = "fatal_contract_plugin"
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=plugin_name,
            description="fatal contract",
            stage="core",
            enabled_by_default=False,
            run=_noop,
            contract_checkers=(_contract_fail,),
            severity="fatal",
        )
    )
    service, cfg = _make_service()
    try:
        with pytest.raises(_FATAL_ERROR):
            service.run_plugins(
                (plugin_name,),
                cfg=GraphMetricsStepConfig(
                    snapshot=cfg.snapshot,
                    plugin_policy=GraphPluginPolicy(fail_fast=True),
                ),
            )
    finally:
        unregister_graph_metric_plugin(plugin_name)


def test_isolated_plugin_executes_in_subprocess(tmp_path: Path) -> None:
    """Plugins marked for isolation should execute via worker."""
    plugin_name = "isolated_plugin"
    output_file = tmp_path / "iso.txt"

    def _isolated(ctx: object) -> None:
        if not isinstance(ctx, GraphMetricExecutionContext):
            message = "context not provided"
            raise TypeError(message)
        path = Path(ctx.options["path"])  # type: ignore[index]
        path.write_text(ctx.repo, encoding="utf-8")

    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=plugin_name,
            description="isolated",
            stage="core",
            enabled_by_default=False,
            run=_isolated,
            requires_isolation=True,
        )
    )
    service, cfg = _make_service()
    try:
        report = service.run_plugins(
            (plugin_name,),
            cfg=cfg,
            run_options=GraphPluginRunOptions(
                plugin_options={plugin_name: {"path": str(output_file)}}
            ),
        )
    finally:
        unregister_graph_metric_plugin(plugin_name)
    if report.records[0].status != "succeeded":
        pytest.fail("Isolated plugin should succeed")
    if not report.records[0].requires_isolation:
        pytest.fail("Isolation flag should be recorded on the run record")
    if not output_file.exists():
        pytest.fail("Isolated plugin should write output via options")
    if output_file.read_text(encoding="utf-8") != cfg.repo:
        pytest.fail("Isolated plugin should receive repo context in child process")


def test_runtime_emits_run_id_and_telemetry_hooks() -> None:
    """Runtime should propagate run_id and invoke telemetry hooks."""
    plugin_name = "telemetry_plugin"

    def _noop(_ctx: object) -> None:
        return None

    class _Telemetry(GraphRuntimeTelemetry):
        def __init__(self) -> None:
            self.starts: list[tuple[str, str, GraphRunScope]] = []
            self.finishes: list[GraphPluginRunRecord] = []
            self.metrics: list[GraphPluginRunRecord] = []
            self.scopes: list[GraphRunScope] = []

        def start_plugin(
            self,
            plugin: GraphMetricPlugin,
            run_id: str,
            ctx: GraphMetricExecutionContext,
        ) -> str:
            self.starts.append((plugin.name, run_id, ctx.scope))
            return "span"

        def finish_plugin(self, span: object, record: GraphPluginRunRecord) -> None:
            _ = span
            self.finishes.append(record)

        def record_metrics(self, record: GraphPluginRunRecord, scope: GraphRunScope) -> None:
            self.metrics.append(record)
            self.scopes.append(scope)

    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=plugin_name,
            description="telemetry",
            stage="core",
            enabled_by_default=False,
            run=_noop,
        )
    )
    service, cfg = _make_service()
    scope = GraphRunScope(paths=("a.py",))
    telemetry = _Telemetry()
    service.telemetry = telemetry
    try:
        report = service.run_plugins(
            (plugin_name,),
            cfg=GraphMetricsStepConfig(snapshot=cfg.snapshot, scope=scope),
            run_options=GraphPluginRunOptions(scope=scope),
        )
    finally:
        unregister_graph_metric_plugin(plugin_name)
    if not report.records:
        pytest.fail("Expected telemetry plugin to produce record")
    record = report.records[0]
    if not record.policy_fail_fast:
        pytest.fail("Policy fail_fast should be reflected on plugin record")
    _assert_telemetry_expectations(
        record=record,
        report_run_id=report.run_id,
        telemetry=telemetry,
        scope=scope,
    )
