"""Tests for dependency planning, skips, and dep graph propagation."""

from __future__ import annotations

import pytest

from codeintel.analytics.graphs.plugins import plan_graph_metric_plugins
from tests.analytics.conftest import (
    PluginTestHarness,
    make_consumer_plugin,
    make_provider_plugin,
)


def test_planner_orders_providers_before_consumers(plugin_harness: PluginTestHarness) -> None:
    """Planner should order providers before dependent consumers and expose dep graph."""
    provider = make_provider_plugin(name="dep_provider", provides=("resource_x",))
    consumer = make_consumer_plugin(
        name="dep_consumer",
        requires=("resource_x",),
        depends_on=("dep_provider",),
    )
    plugin_harness.register(provider)
    plugin_harness.register(consumer)
    plan = plan_graph_metric_plugins((consumer.name, provider.name))
    ordered_names = tuple(plugin.name for plugin in plan.plugins)
    if ordered_names != ("dep_provider", "dep_consumer"):
        message = "Provider should precede consumer in plan order"
        pytest.fail(message)
    expected_dep_graph = {"dep_consumer": ("dep_provider",), "dep_provider": ()}
    if plan.dep_graph != expected_dep_graph:
        message = "Dependency graph should include provider relation"
        pytest.fail(message)
    report = plugin_harness.run_plugins_with_cfg((consumer.name, provider.name))
    if report.ordered_plugins != ("dep_provider", "dep_consumer"):
        message = "Runtime should respect planned order"
        pytest.fail(message)
    if report.dep_graph != expected_dep_graph:
        message = "Runtime dep graph should match planner"
        pytest.fail(message)


def test_plan_records_disabled_plugins() -> None:
    """Planner should record disabled plugins as skipped."""
    plan = plan_graph_metric_plugins(plugin_names=("disabled_one",), disabled=("disabled_one",))
    if plan.plugins:
        message = "Disabled plugin should not appear in plan.plugins"
        pytest.fail(message)
    if not plan.skipped_plugins or plan.skipped_plugins[0].reason != "disabled":
        message = "Disabled plugin should be recorded as skipped with reason"
        pytest.fail(message)
