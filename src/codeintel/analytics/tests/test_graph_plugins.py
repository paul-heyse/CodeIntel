"""Unit tests for graph metric plugins registry."""

from __future__ import annotations

import pytest

from codeintel.analytics.graphs.contracts import PluginContractResult
from codeintel.analytics.graphs.plugins import (
    DEFAULT_GRAPH_METRIC_PLUGINS,
    GraphMetricPlugin,
    get_graph_metric_plugin,
    list_graph_metric_plugins,
    load_graph_metric_plugins_from_entrypoints,
    plan_graph_metric_plugins,
    register_graph_metric_plugin,
    unregister_graph_metric_plugin,
)


def test_default_plugins_registered() -> None:
    """Ensure all default plugin names are registered."""
    names = {plugin.name for plugin in list_graph_metric_plugins()}
    for name in DEFAULT_GRAPH_METRIC_PLUGINS:
        if name not in names:
            pytest.fail(f"Plugin {name} was not registered")


def test_get_graph_metric_plugin_round_trip() -> None:
    """Lookup returns registered plugin metadata."""
    for name in DEFAULT_GRAPH_METRIC_PLUGINS:
        plugin = get_graph_metric_plugin(name)
        if plugin.name != name:
            pytest.fail(f"Lookup returned incorrect plugin: {plugin.name}")
        if not plugin.description:
            pytest.fail(f"Plugin {name} is missing description text")


def test_contract_checker_registration_preserved() -> None:
    """Plugins can attach contract checkers."""

    def _checker(_ctx: object) -> PluginContractResult:
        return PluginContractResult(name="demo", status="passed")

    plugin = GraphMetricPlugin(
        name="plugin_with_contract",
        description="has contract",
        stage="core",
        enabled_by_default=False,
        run=lambda _ctx: None,
        contract_checkers=(_checker,),
    )
    register_graph_metric_plugin(plugin)
    try:
        looked_up = get_graph_metric_plugin("plugin_with_contract")
        if looked_up.contract_checkers != (_checker,):
            pytest.fail("Contract checkers should be preserved on registration")
    finally:
        unregister_graph_metric_plugin("plugin_with_contract")


def test_plan_graph_metric_plugins_preserves_order() -> None:
    """Planning maintains input order when no dependencies reorder."""
    plan = plan_graph_metric_plugins(DEFAULT_GRAPH_METRIC_PLUGINS)
    if plan.ordered_names != DEFAULT_GRAPH_METRIC_PLUGINS:
        pytest.fail("Planning should preserve declared plugin order when no deps are set")


def test_plan_graph_metric_plugins_reports_missing_dependency() -> None:
    """Missing dependencies raise a clear error."""
    plugin_name = "tmp_missing_dep_plugin"
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=plugin_name,
            description="temporary plugin",
            stage="core",
            enabled_by_default=False,
            depends_on=("nonexistent_dep",),
            run=lambda _ctx: None,
        )
    )
    try:
        with pytest.raises(ValueError, match="depends on 'nonexistent_dep'"):
            plan_graph_metric_plugins((plugin_name,))
    finally:
        unregister_graph_metric_plugin(plugin_name)


def test_plan_graph_metric_plugins_orders_dependencies() -> None:
    """Dependencies execute before dependents."""
    base_name = "core_graph_metrics"
    plugin_name = "tmp_dep_plugin"
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=plugin_name,
            description="temporary plugin with dep",
            stage="core",
            enabled_by_default=False,
            depends_on=(base_name,),
            run=lambda _ctx: None,
        )
    )
    try:
        plan = plan_graph_metric_plugins((plugin_name, base_name))
        ordered = plan.ordered_names[:2]
        if ordered != (base_name, plugin_name):
            pytest.fail("Dependency should run before dependent plugin")
    finally:
        unregister_graph_metric_plugin(plugin_name)


def test_plan_graph_metric_plugins_respects_disabled() -> None:
    """Disabled plugins are skipped and recorded in the plan."""
    disabled_name = "core_graph_metrics"
    plan = plan_graph_metric_plugins(
        plugin_names=DEFAULT_GRAPH_METRIC_PLUGINS,
        disabled=(disabled_name,),
    )
    if disabled_name in plan.ordered_names:
        pytest.fail("Disabled plugin should not appear in ordered plan")
    if not plan.skipped_plugins or plan.skipped_plugins[0].name != disabled_name:
        pytest.fail("Disabled plugin should be recorded as skipped")


def test_plan_graph_metric_plugins_enabled_overrides_defaults() -> None:
    """Enabled list should replace defaults when provided."""
    only_name = "cfg_metrics"
    plan = plan_graph_metric_plugins(
        plugin_names=None,
        enabled=(only_name,),
        defaults=DEFAULT_GRAPH_METRIC_PLUGINS,
    )
    if plan.ordered_names != (only_name,):
        pytest.fail("Enabled list should be used verbatim")
    if plan.dep_graph.get(only_name, ()) != ():
        pytest.fail("Dep graph should reflect the enabled list")


def test_plan_graph_metric_plugins_requires_provider_present() -> None:
    """Planning fails fast when a required capability has no provider."""
    capability = "demo_capability"
    consumer_name = "tmp_requires_capability"
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=consumer_name,
            description="requires capability",
            stage="core",
            enabled_by_default=False,
            requires=(capability,),
            run=lambda _ctx: None,
        )
    )
    try:
        with pytest.raises(ValueError, match="requires capability 'demo_capability'"):
            plan_graph_metric_plugins((consumer_name,))
    finally:
        unregister_graph_metric_plugin(consumer_name)


def test_plan_graph_metric_plugins_orders_unique_provider_before_consumer() -> None:
    """A single provider is treated as a dependency for ordering."""
    capability = "unique_cap"
    provider_name = "tmp_unique_provider"
    consumer_name = "tmp_capability_consumer"
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=provider_name,
            description="provides capability",
            stage="core",
            enabled_by_default=False,
            provides=(capability,),
            run=lambda _ctx: None,
        )
    )
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=consumer_name,
            description="consumes capability",
            stage="core",
            enabled_by_default=False,
            requires=(capability,),
            run=lambda _ctx: None,
        )
    )
    try:
        plan = plan_graph_metric_plugins((consumer_name, provider_name))
        ordered = plan.ordered_names[:2]
        if ordered != (provider_name, consumer_name):
            pytest.fail("Provider should be ordered before the consumer")
    finally:
        unregister_graph_metric_plugin(provider_name)
        unregister_graph_metric_plugin(consumer_name)


def test_plan_graph_metric_plugins_requires_disambiguation_for_multiple_providers() -> None:
    """Ambiguous capability providers require explicit depends_on declarations."""
    capability = "shared_capability"
    provider_one = "tmp_capability_provider_one"
    provider_two = "tmp_capability_provider_two"
    consumer_name = "tmp_capability_consumer_ambiguous"
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=provider_one,
            description="provider one",
            stage="core",
            enabled_by_default=False,
            provides=(capability,),
            run=lambda _ctx: None,
        )
    )
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=provider_two,
            description="provider two",
            stage="core",
            enabled_by_default=False,
            provides=(capability,),
            run=lambda _ctx: None,
        )
    )
    register_graph_metric_plugin(
        GraphMetricPlugin(
            name=consumer_name,
            description="ambiguous consumer",
            stage="core",
            enabled_by_default=False,
            requires=(capability,),
            run=lambda _ctx: None,
        )
    )
    try:
        with pytest.raises(ValueError, match="multiple providers are available"):
            plan_graph_metric_plugins((consumer_name, provider_one, provider_two))
    finally:
        unregister_graph_metric_plugin(provider_one)
        unregister_graph_metric_plugin(provider_two)
        unregister_graph_metric_plugin(consumer_name)


def test_load_graph_metric_plugins_from_entrypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entrypoint discovery registers plugins."""

    class _DummyEntryPoint:
        def __init__(self, plugin: GraphMetricPlugin) -> None:
            self.name = plugin.name
            self._plugin = plugin

        def load(self) -> GraphMetricPlugin:
            return self._plugin

    class _DummyEntrypoints:
        def __init__(self, ep: _DummyEntryPoint) -> None:
            self._ep = ep

        def select(self, group: str) -> list[_DummyEntryPoint]:
            if group == "codeintel.graph_metric_plugins":
                return [self._ep]
            return []

    plugin = GraphMetricPlugin(
        name="entrypoint_plugin",
        description="from entrypoint",
        stage="core",
        enabled_by_default=False,
        run=lambda _ctx: None,
    )
    dummy_ep = _DummyEntryPoint(plugin)
    monkeypatch.setattr("importlib.metadata.entry_points", lambda: _DummyEntrypoints(dummy_ep))
    try:
        loaded = load_graph_metric_plugins_from_entrypoints(force=True)
        if not loaded or loaded[0].name != "entrypoint_plugin":
            pytest.fail("Entrypoint loader should return discovered plugin")
    finally:
        unregister_graph_metric_plugin("entrypoint_plugin")
