"""Tests for skip-on-unchanged using row_counts and options hashes."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.graph_service_runtime import GraphPluginRunOptions
from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
    GraphPluginResult,
)
from codeintel.config.steps_graphs import GraphMetricsStepConfig, GraphPluginPolicy
from tests.analytics.conftest import PluginTestHarness


def _option_count_plugin(ctx: GraphMetricExecutionContext) -> GraphPluginResult:
    """Return row counts keyed by provided option count.

    Returns
    -------
    GraphPluginResult
        Result containing a row count derived from options.
    """
    opts = ctx.options or {}
    count = int(opts.get("count", 1)) if isinstance(opts, dict) else 1
    return GraphPluginResult(row_counts={"analytics.option_count": count})


def test_skip_on_unchanged_respects_row_counts_and_options(
    plugin_harness: PluginTestHarness, tmp_path: Path
) -> None:
    """Skip-on-unchanged should skip identical runs and rerun on changed options/counts."""
    plugin = GraphMetricPlugin(
        name="option_count_plugin",
        description="uses options for row count",
        stage="core",
        enabled_by_default=False,
        run=_option_count_plugin,
    )
    plugin_harness.register(plugin)
    policy_cfg = GraphMetricsStepConfig(
        snapshot=plugin_harness.cfg.snapshot,
        plugin_policy=GraphPluginPolicy(skip_on_unchanged=True),
    )
    manifest_path = tmp_path / "option-count.json"
    # First run seeds manifest.
    first_report = plugin_harness.run_plugins_with_cfg(
        (plugin.name,),
        cfg=policy_cfg,
        run_options=GraphPluginRunOptions(
            plugin_options={plugin.name: {"count": 1}},
            manifest_path=manifest_path,
        ),
    )
    first_record = first_report.records[0]
    if first_record.status != "succeeded":
        message = "First run should succeed"
        pytest.fail(message)
    # Second run should skip as unchanged (same options/count).
    second_report = plugin_harness.run_plugins_with_cfg(
        (plugin.name,),
        cfg=policy_cfg,
        run_options=GraphPluginRunOptions(
            plugin_options={plugin.name: {"count": 1}},
            manifest_path=manifest_path,
        ),
    )
    second_record = second_report.records[0]
    if second_record.status != "skipped" or second_record.skipped_reason != "unchanged":
        message = "Second run should skip due to unchanged options/row_counts"
        pytest.fail(message)
    # Changing options should force execution.
    third_report = plugin_harness.run_plugins_with_cfg(
        (plugin.name,),
        cfg=policy_cfg,
        run_options=GraphPluginRunOptions(
            plugin_options={plugin.name: {"count": 2}},
            manifest_path=manifest_path,
        ),
    )
    third_record = third_report.records[0]
    if third_record.status != "succeeded":
        message = "Changed options should force re-execution"
        pytest.fail(message)
