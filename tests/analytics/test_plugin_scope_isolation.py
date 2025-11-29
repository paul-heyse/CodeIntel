"""Tests covering scope propagation, isolation flags, and planning edges."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from codeintel.analytics.graph_service_runtime import GraphPluginRunOptions
from codeintel.analytics.graphs.contracts import PluginContractResult
from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
    GraphPluginResult,
    plan_graph_metric_plugins,
)
from codeintel.config.steps_graphs import GraphRunScope
from tests.analytics.conftest import (
    PluginTestHarness,
    make_isolation_plugin,
    make_scope_plugin,
)


def _simple_result() -> GraphPluginResult:
    """Return a minimal plugin result with no-op row counts.

    Returns
    -------
    GraphPluginResult
        Result containing a placeholder row count.
    """
    return GraphPluginResult(row_counts={"analytics.test": 1})


def test_scope_propagates_to_plugin_and_report(
    plugin_harness: PluginTestHarness, tmp_path: Path
) -> None:
    """Scopes from run options should reach plugin context and report."""
    observed: dict[str, object] = {}

    def scoped_plugin(ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        observed["paths"] = tuple(ctx.scope.paths)
        observed["modules"] = tuple(ctx.scope.modules)
        observed["time_window"] = ctx.scope.time_window
        return _simple_result()

    plugin = GraphMetricPlugin(
        name="harness_scope_plugin",
        description="scope propagation demo",
        stage="core",
        enabled_by_default=False,
        run=scoped_plugin,
        scope_aware=True,
        supported_scopes=("paths", "modules", "time_window"),
    )
    plugin_harness.register(plugin)
    window = (datetime.now(tz=UTC), datetime.now(tz=UTC) + timedelta(days=1))
    scope = GraphRunScope(paths=("src/foo.py",), modules=("pkg.mod",), time_window=window)
    report = plugin_harness.service.run_plugins(
        (plugin.name,),
        cfg=plugin_harness.cfg,
        run_options=GraphPluginRunOptions(
            scope=scope,
            manifest_path=tmp_path / "scope-manifest.json",
        ),
    )
    record = report.records[0]
    if observed["paths"] != scope.paths:
        message = "Scope paths should reach plugin context"
        pytest.fail(message)
    if observed["modules"] != scope.modules:
        message = "Scope modules should reach plugin context"
        pytest.fail(message)
    if observed["time_window"] != scope.time_window:
        message = "Scope time window should reach plugin context"
        pytest.fail(message)
    if report.scope != scope:
        message = "Report should preserve scope"
        pytest.fail(message)
    if record.status != "succeeded":
        message = "Plugin should succeed with provided scope"
        pytest.fail(message)
    manifest = json.loads((tmp_path / "scope-manifest.json").read_text(encoding="utf-8"))
    if manifest["scope"]["paths"] != list(scope.paths):
        message = "Manifest should record scope paths"
        pytest.fail(message)
    if manifest["records"][0]["status"] != "succeeded":
        message = "Manifest should reflect success"
        pytest.fail(message)


def test_dry_run_skips_execution(plugin_harness: PluginTestHarness, tmp_path: Path) -> None:
    """Dry-run mode should skip execution and mark skipped reason."""
    executed = False

    def dry_plugin(_ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        nonlocal executed
        executed = True
        return _simple_result()

    plugin = GraphMetricPlugin(
        name="harness_dry_run_plugin",
        description="dry run demo",
        stage="core",
        enabled_by_default=False,
        run=dry_plugin,
    )
    plugin_harness.register(plugin)
    report = plugin_harness.service.run_plugins(
        (plugin.name,),
        cfg=plugin_harness.cfg,
        run_options=GraphPluginRunOptions(
            dry_run=True,
            manifest_path=tmp_path / "dry-run-manifest.json",
        ),
    )
    record = report.records[0]
    if record.status != "skipped" or record.skipped_reason != "dry_run":
        message = "Dry-run should skip execution with reason 'dry_run'"
        pytest.fail(message)
    if executed:
        message = "Dry-run should not call plugin implementation"
        pytest.fail(message)
    manifest = json.loads((tmp_path / "dry-run-manifest.json").read_text(encoding="utf-8"))
    if manifest["records"][0]["status"] != "skipped":
        message = "Manifest should show skipped status for dry-run"
        pytest.fail(message)


def test_isolation_flag_in_run_record(plugin_harness: PluginTestHarness, tmp_path: Path) -> None:
    """Isolation metadata should be reflected in run records."""

    def isolated_plugin(_ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        return _simple_result()

    plugin = GraphMetricPlugin(
        name="harness_isolated_plugin",
        description="isolation demo",
        stage="core",
        enabled_by_default=False,
        run=isolated_plugin,
        requires_isolation=True,
        isolation_kind="process",
    )
    plugin_harness.register(plugin)
    report = plugin_harness.service.run_plugins(
        (plugin.name,),
        cfg=plugin_harness.cfg,
        run_options=GraphPluginRunOptions(manifest_path=tmp_path / "isolation.json"),
    )
    record = report.records[0]
    if record.requires_isolation is not True:
        message = "Run record should reflect isolation requirement"
        pytest.fail(message)
    if record.isolation_kind != "process":
        message = "Isolation kind should be preserved in run record"
        pytest.fail(message)
    if record.status != "succeeded":
        message = "Isolated plugin should succeed"
        pytest.fail(message)


def test_scope_required_without_scope_skips(plugin_harness: PluginTestHarness) -> None:
    """Scope-aware plugin should skip or fail when scope is missing, per severity."""
    plugin = make_scope_plugin(name="scope_required_skip", severity="skip_on_error")
    plugin_harness.register(plugin)
    report = plugin_harness.service.run_plugins((plugin.name,), cfg=plugin_harness.cfg)
    record = report.records[0]
    if record.status != "skipped":
        message = "Scope-required plugin should skip when scope is absent"
        pytest.fail(message)


def test_isolation_worker_returns_row_counts_and_contracts(
    plugin_harness: PluginTestHarness, tmp_path: Path
) -> None:
    """Isolation worker should return row counts and contracts with run_id preserved."""

    def _contract(_ctx: GraphMetricExecutionContext) -> PluginContractResult:
        return PluginContractResult(name="iso_contract", status="passed")

    plugin = replace(
        make_isolation_plugin(name="iso_worker_plugin"),
        contract_checkers=(_contract,),
    )
    plugin_harness.register(plugin)
    report = plugin_harness.service.run_plugins(
        (plugin.name,),
        cfg=plugin_harness.cfg,
        run_options=GraphPluginRunOptions(manifest_path=tmp_path / "iso-worker.json"),
    )
    record = report.records[0]
    if record.requires_isolation is not True or record.isolation_kind != "process":
        message = "Isolation metadata should be preserved"
        pytest.fail(message)
    if record.row_counts is None or not record.row_counts:
        message = "Row counts should flow back from isolation worker"
        pytest.fail(message)
    if not record.contracts or record.contracts[0].status != "passed":
        message = "Contracts should flow back from isolation worker"
        pytest.fail(message)
    if not record.run_id or record.run_id != report.run_id:
        message = "Run ID should be consistent between parent and child"
        pytest.fail(message)


def test_planner_rejects_cycles(plugin_harness: PluginTestHarness) -> None:
    """Planner should fail fast on dependency cycles."""

    def _no_op(_ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        return _simple_result()

    plugin_a = GraphMetricPlugin(
        name="cycle_a",
        description="cycle a",
        stage="core",
        enabled_by_default=False,
        run=_no_op,
        depends_on=("cycle_b",),
    )
    plugin_b = GraphMetricPlugin(
        name="cycle_b",
        description="cycle b",
        stage="core",
        enabled_by_default=False,
        run=_no_op,
        depends_on=("cycle_a",),
    )
    plugin_harness.register(plugin_a)
    plugin_harness.register(plugin_b)
    with pytest.raises(ValueError, match="cycle"):
        plan_graph_metric_plugins((plugin_a.name, plugin_b.name))


def test_planner_requires_providers(plugin_harness: PluginTestHarness) -> None:
    """Planner should flag missing providers for declared requirements."""

    def _no_op(_ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        return _simple_result()

    consumer = GraphMetricPlugin(
        name="requires_provider_consumer",
        description="requires provider",
        stage="core",
        enabled_by_default=False,
        run=_no_op,
        requires=("missing_resource",),
    )
    plugin_harness.register(consumer)
    with pytest.raises(ValueError, match="missing"):
        plan_graph_metric_plugins((consumer.name,))
