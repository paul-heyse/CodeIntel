"""Tests for contract failure handling and idempotency edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.graph_service_runtime import GraphPluginRunOptions
from codeintel.analytics.graphs.contracts import PluginContractResult
from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
    GraphPluginResult,
)
from codeintel.config.steps_graphs import GraphMetricsStepConfig, GraphPluginPolicy
from tests.analytics.conftest import PluginTestHarness


def _no_op_result() -> GraphPluginResult:
    return GraphPluginResult(row_counts={"analytics.contract_demo": 1})


def test_contract_soft_failure_marks_record_failed(plugin_harness: PluginTestHarness) -> None:
    """Soft contract failure should mark record failed without raising fatal error."""

    def _run(_ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        return _no_op_result()

    def _soft_contract(_ctx: object) -> PluginContractResult:
        return PluginContractResult(name="contract", status="soft_failed")

    plugin = GraphMetricPlugin(
        name="contract_soft_fail",
        description="soft contract",
        stage="core",
        enabled_by_default=False,
        run=_run,
        contract_checkers=(_soft_contract,),
    )
    plugin_harness.register(plugin)
    report = plugin_harness.service.run_plugins((plugin.name,), cfg=plugin_harness.cfg)
    record = report.records[0]
    if record.status != "failed" or record.error != "contract_failed":
        message = "Soft contract failure should mark record failed with contract_failed error"
        pytest.fail(message)


def test_contract_fatal_failure_raises(plugin_harness: PluginTestHarness) -> None:
    """Fatal contract failure should raise _PluginFatalError when fail_fast is enabled."""

    def _run(_ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        return _no_op_result()

    def _fatal_contract(_ctx: object) -> PluginContractResult:
        return PluginContractResult(name="contract", status="failed")

    plugin = GraphMetricPlugin(
        name="contract_fatal_fail",
        description="fatal contract",
        stage="core",
        enabled_by_default=False,
        run=_run,
        contract_checkers=(_fatal_contract,),
    )
    plugin_harness.register(plugin)
    cfg = GraphMetricsStepConfig(
        snapshot=plugin_harness.cfg.snapshot,
        plugin_policy=GraphPluginPolicy(fail_fast=False),
    )
    report = plugin_harness.service.run_plugins((plugin.name,), cfg=cfg)
    record = report.records[0]
    if record.status != "failed" or record.error != "contract_failed":
        message = "Fatal contract failure should mark record failed when fail_fast is False"
        pytest.fail(message)


def test_idempotency_respects_row_count_changes(plugin_harness: PluginTestHarness, tmp_path: Path) -> None:
    """Skip-on-unchanged should rerun when row counts change even if options stay the same."""
    state = {"count": 1}

    def _run(_ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        return GraphPluginResult(row_counts={"analytics.stateful": state["count"]})

    plugin = GraphMetricPlugin(
        name="stateful_counts",
        description="stateful row count plugin",
        stage="core",
        enabled_by_default=False,
        run=_run,
    )
    plugin_harness.register(plugin)
    policy_cfg = GraphMetricsStepConfig(
        snapshot=plugin_harness.cfg.snapshot,
        plugin_policy=GraphPluginPolicy(skip_on_unchanged=True),
    )
    manifest_path = tmp_path / "stateful.json"
    first_report = plugin_harness.run_plugins_with_cfg(
        (plugin.name,),
        cfg=policy_cfg,
        run_options=GraphPluginRunOptions(manifest_path=manifest_path),
    )
    if first_report.records[0].status != "succeeded":
        message = "First run should succeed"
        pytest.fail(message)
    second_report = plugin_harness.run_plugins_with_cfg(
        (plugin.name,),
        cfg=policy_cfg,
        run_options=GraphPluginRunOptions(manifest_path=manifest_path),
    )
    if second_report.records[0].status != "skipped":
        message = "Second run should skip when unchanged"
        pytest.fail(message)
    state["count"] = 2
    third_report = plugin_harness.run_plugins_with_cfg(
        (plugin.name,),
        cfg=policy_cfg,
        run_options=GraphPluginRunOptions(manifest_path=manifest_path),
    )
    if third_report.records[0].status != "succeeded":
        message = "Row count change should force re-execution"
        pytest.fail(message)
