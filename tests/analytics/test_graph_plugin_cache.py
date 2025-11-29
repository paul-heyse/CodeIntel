"""Cache/scratch lifecycle coverage for graph plugins."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
    GraphPluginResult,
)
from tests.analytics.conftest import PluginTestHarness


def test_scratch_declare_consume_and_cleanup(plugin_harness: PluginTestHarness, tmp_path: Path) -> None:
    """Values declared in scratch should be consumable across plugins and cleaned afterwards."""
    marker = tmp_path / "scratch_cleanup.txt"
    cache_key = "scratch_token"

    def _populate(ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        ctx.scratch.declare(cache_key, "value")

        def _write_marker() -> None:
            marker.write_text("cleaned", encoding="utf-8")

        ctx.scratch.register_cleanup(_write_marker)
        return GraphPluginResult(row_counts={"analytics.cache_populator": 1})

    def _consume(ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        cached = ctx.scratch.consume(cache_key)
        if cached != "value":
            message = "scratch value should be available to dependent plugin"
            pytest.fail(message)
        return GraphPluginResult(row_counts={"analytics.cache_consumer": 1})

    populator = GraphMetricPlugin(
        name="cache_populator",
        description="populates scratch cache",
        stage="core",
        enabled_by_default=False,
        run=_populate,
        cache_populates=(cache_key,),
    )
    consumer = GraphMetricPlugin(
        name="cache_consumer",
        description="consumes scratch cache",
        stage="core",
        enabled_by_default=False,
        run=_consume,
        depends_on=("cache_populator",),
        cache_consumes=(cache_key,),
    )
    plugin_harness.register(populator)
    plugin_harness.register(consumer)

    report = plugin_harness.run_plugins_with_cfg(("cache_populator", "cache_consumer"))
    statuses = {record.name: record.status for record in report.records}
    if statuses.get("cache_populator") != "succeeded" or statuses.get("cache_consumer") != "succeeded":
        message = "both cache plugins should succeed"
        pytest.fail(message)
    if not marker.exists():
        message = "cleanup callback should execute after run"
        pytest.fail(message)
    if marker.read_text(encoding="utf-8") != "cleaned":
        message = "cleanup marker should contain expected text"
        pytest.fail(message)


def test_isolation_plugin_runs_with_scratch(plugin_harness: PluginTestHarness) -> None:
    """Scratch should not break isolation-capable plugins."""

    def _run(ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        ctx.scratch.declare("isolated", value=True)
        return GraphPluginResult(row_counts={"analytics.isolation_scratch": 1})

    isolation_plugin = GraphMetricPlugin(
        name="isolation_scratch",
        description="isolation plugin with scratch usage",
        stage="core",
        enabled_by_default=False,
        run=_run,
        requires_isolation=True,
        isolation_kind="process",
        cache_populates=("isolated",),
    )
    plugin_harness.register(isolation_plugin)
    report = plugin_harness.run_plugins_with_cfg(("isolation_scratch",))
    record = report.records[0]
    if record.status != "succeeded":
        message = "isolation plugin with scratch should still succeed"
        pytest.fail(message)
