"""Tests for graph plugin registry.

This module tests the plugin registry including registration, lookup,
filtering, dependency resolution, and topological sorting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pytest

from codeintel.graphs.core.context import GraphExecutionContext
from codeintel.graphs.core.protocol import (
    FunctionalGraphPlugin,
    GraphPluginKind,
    GraphPluginMetadata,
    GraphPluginProtocol,
    GraphPluginStage,
)
from codeintel.graphs.core.registry import (
    GraphPluginRegistry,
    get_graph_registry,
    list_graph_plugins,
    register_graph_plugin,
    unregister_graph_plugin,
)
from codeintel.graphs.core.result import GraphPluginResult

EXPECTED_PLUGIN_COUNT: Final = 3


@dataclass(frozen=True)
class PluginConfig:
    """Configuration for creating test plugins."""

    name: str
    kind: GraphPluginKind = "builder"
    stage: GraphPluginStage = "goid"
    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    produces_tables: tuple[str, ...] = ()


def _make_test_plugin(config: PluginConfig) -> GraphPluginProtocol:
    """Create a test plugin for registry tests.

    Parameters
    ----------
    config
        Plugin configuration.

    Returns
    -------
    GraphPluginProtocol
        Test plugin instance.
    """

    def execute(_ctx: GraphExecutionContext) -> GraphPluginResult:
        return GraphPluginResult.ok()

    metadata = GraphPluginMetadata(
        name=config.name,
        description=f"Test plugin {config.name}",
        kind=config.kind,
        stage=config.stage,
        depends_on=config.depends_on,
        provides=config.provides,
        requires=config.requires,
        produces_tables=config.produces_tables,
    )

    return FunctionalGraphPlugin(_metadata=metadata, _execute_fn=execute)


# =============================================================================
# Registration Tests
# =============================================================================


def test_register_plugin() -> None:
    """Register a plugin successfully.

    Raises
    ------
    AssertionError
        If registration fails.
    """
    registry = GraphPluginRegistry()
    plugin = _make_test_plugin(PluginConfig(name="test_register"))

    registry.register(plugin)

    if not registry.contains("test_register"):
        msg = "Expected plugin to be registered"
        raise AssertionError(msg)


def test_register_duplicate_raises() -> None:
    """Registering duplicate plugin raises ValueError."""
    registry = GraphPluginRegistry()
    plugin = _make_test_plugin(PluginConfig(name="duplicate_plugin"))

    registry.register(plugin)
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(plugin)


def test_unregister_plugin() -> None:
    """Unregister a plugin successfully.

    Raises
    ------
    AssertionError
        If unregistration fails.
    """
    registry = GraphPluginRegistry()
    plugin = _make_test_plugin(PluginConfig(name="test_unregister"))

    registry.register(plugin)
    registry.unregister("test_unregister")

    if registry.contains("test_unregister"):
        msg = "Expected plugin to be unregistered"
        raise AssertionError(msg)


def test_unregister_nonexistent_is_silent() -> None:
    """Unregistering nonexistent plugin does not raise."""
    registry = GraphPluginRegistry()
    # Should not raise
    registry.unregister("nonexistent")


# =============================================================================
# Lookup Tests
# =============================================================================


def test_get_plugin() -> None:
    """Get a registered plugin by name.

    Raises
    ------
    AssertionError
        If plugin is not returned.
    """
    registry = GraphPluginRegistry()
    plugin = _make_test_plugin(PluginConfig(name="test_get"))
    registry.register(plugin)

    retrieved = registry.get("test_get")

    if retrieved.metadata.name != "test_get":
        msg = f"Expected plugin name 'test_get', got '{retrieved.metadata.name}'"
        raise AssertionError(msg)


def test_get_unknown_raises() -> None:
    """Getting unknown plugin raises KeyError."""
    registry = GraphPluginRegistry()
    with pytest.raises(KeyError, match="Unknown"):
        registry.get("unknown_plugin")


def test_contains() -> None:
    """Check if plugin is registered.

    Raises
    ------
    AssertionError
        If contains check fails.
    """
    registry = GraphPluginRegistry()
    plugin = _make_test_plugin(PluginConfig(name="test_contains"))
    registry.register(plugin)

    if not registry.contains("test_contains"):
        msg = "Expected contains to return True"
        raise AssertionError(msg)
    if registry.contains("nonexistent"):
        msg = "Expected contains to return False for nonexistent"
        raise AssertionError(msg)


# =============================================================================
# Listing Tests
# =============================================================================


def test_list_all() -> None:
    """List all registered plugins.

    Raises
    ------
    AssertionError
        If plugin count is wrong.
    """
    registry = GraphPluginRegistry()
    plugins = [
        _make_test_plugin(PluginConfig(name=f"list_all_{i}")) for i in range(EXPECTED_PLUGIN_COUNT)
    ]
    for p in plugins:
        registry.register(p)

    all_plugins = registry.list_all()

    if len(all_plugins) < EXPECTED_PLUGIN_COUNT:
        msg = f"Expected at least {EXPECTED_PLUGIN_COUNT} plugins, got {len(all_plugins)}"
        raise AssertionError(msg)


def test_list_names() -> None:
    """List names of all registered plugins.

    Raises
    ------
    AssertionError
        If names are missing.
    """
    registry = GraphPluginRegistry()
    plugins = [_make_test_plugin(PluginConfig(name=f"list_names_{i}")) for i in range(2)]
    for p in plugins:
        registry.register(p)

    names = registry.list_names()

    if "list_names_0" not in names:
        msg = "Expected 'list_names_0' in names"
        raise AssertionError(msg)
    if "list_names_1" not in names:
        msg = "Expected 'list_names_1' in names"
        raise AssertionError(msg)


def test_list_by_kind() -> None:
    """List plugins by kind.

    Raises
    ------
    AssertionError
        If filter fails.
    """
    registry = GraphPluginRegistry()
    builder = _make_test_plugin(PluginConfig(name="builder_plugin", kind="builder"))
    metric = _make_test_plugin(PluginConfig(name="metric_plugin", kind="metric"))
    registry.register(builder)
    registry.register(metric)

    builders = registry.list_by_kind("builder")
    metrics = registry.list_by_kind("metric")

    builder_names = [p.metadata.name for p in builders]
    if "builder_plugin" not in builder_names:
        msg = "Expected builder_plugin in builders"
        raise AssertionError(msg)

    metric_names = [p.metadata.name for p in metrics]
    if "metric_plugin" not in metric_names:
        msg = "Expected metric_plugin in metrics"
        raise AssertionError(msg)


def test_list_by_stage() -> None:
    """List plugins by stage.

    Raises
    ------
    AssertionError
        If filter fails.
    """
    registry = GraphPluginRegistry()
    goid = _make_test_plugin(PluginConfig(name="goid_plugin", stage="goid"))
    core = _make_test_plugin(PluginConfig(name="core_plugin", stage="core"))
    registry.register(goid)
    registry.register(core)

    goid_plugins = registry.list_by_stage("goid")
    goid_names = [p.metadata.name for p in goid_plugins]

    if "goid_plugin" not in goid_names:
        msg = "Expected goid_plugin in goid stage plugins"
        raise AssertionError(msg)


def test_list_providing() -> None:
    """List plugins providing a capability.

    Raises
    ------
    AssertionError
        If filter fails.
    """
    registry = GraphPluginRegistry()
    plugin = _make_test_plugin(PluginConfig(name="capability_provider", provides=("capability_a",)))
    registry.register(plugin)

    providers = registry.list_providing("capability_a")

    if len(providers) != 1:
        msg = f"Expected 1 provider, got {len(providers)}"
        raise AssertionError(msg)
    if providers[0].metadata.name != "capability_provider":
        msg = f"Expected 'capability_provider', got '{providers[0].metadata.name}'"
        raise AssertionError(msg)


def test_list_by_table() -> None:
    """List plugins producing a table.

    Raises
    ------
    AssertionError
        If filter fails.
    """
    registry = GraphPluginRegistry()
    plugin = _make_test_plugin(
        PluginConfig(name="table_producer", produces_tables=("graph.test_table",))
    )
    registry.register(plugin)

    producers = registry.list_by_table("graph.test_table")

    if len(producers) != 1:
        msg = f"Expected 1 producer, got {len(producers)}"
        raise AssertionError(msg)


# =============================================================================
# Plan Building Tests
# =============================================================================


def test_plan_simple() -> None:
    """Build a simple plan without dependencies.

    Raises
    ------
    AssertionError
        If plan is not built correctly.
    """
    registry = GraphPluginRegistry()
    plugin = _make_test_plugin(PluginConfig(name="simple_plan_plugin"))
    registry.register(plugin)

    plan = registry.plan(plugin_names=["simple_plan_plugin"])

    if len(plan.plugins) != 1:
        msg = f"Expected 1 plugin in plan, got {len(plan.plugins)}"
        raise AssertionError(msg)
    if plan.plugins[0].metadata.name != "simple_plan_plugin":
        msg = f"Expected 'simple_plan_plugin', got '{plan.plugins[0].metadata.name}'"
        raise AssertionError(msg)


def test_plan_with_dependencies() -> None:
    """Build a plan respecting dependencies.

    Raises
    ------
    AssertionError
        If dependency order is wrong.
    """
    registry = GraphPluginRegistry()
    plugin_a = _make_test_plugin(PluginConfig(name="dep_plan_a"))
    plugin_b = _make_test_plugin(PluginConfig(name="dep_plan_b", depends_on=("dep_plan_a",)))
    registry.register(plugin_a)
    registry.register(plugin_b)

    plan = registry.plan(plugin_names=["dep_plan_a", "dep_plan_b"])

    names = plan.ordered_names
    if names.index("dep_plan_a") >= names.index("dep_plan_b"):
        msg = f"Expected dep_plan_a before dep_plan_b, got {names}"
        raise AssertionError(msg)


def test_plan_capability_based_dependency() -> None:
    """Build a plan with capability-based dependencies.

    Raises
    ------
    AssertionError
        If capability dependency is not resolved.
    """
    registry = GraphPluginRegistry()
    provider = _make_test_plugin(PluginConfig(name="cap_provider", provides=("cap_x",)))
    consumer = _make_test_plugin(PluginConfig(name="cap_consumer", requires=("cap_x",)))
    registry.register(provider)
    registry.register(consumer)

    plan = registry.plan(plugin_names=["cap_provider", "cap_consumer"])

    names = plan.ordered_names
    if names.index("cap_provider") >= names.index("cap_consumer"):
        msg = f"Expected cap_provider before cap_consumer, got {names}"
        raise AssertionError(msg)


def test_plan_with_disabled_plugins() -> None:
    """Plan with disabled plugins excludes them.

    Raises
    ------
    AssertionError
        If disabled plugins are included.
    """
    registry = GraphPluginRegistry()
    plugin_a = _make_test_plugin(PluginConfig(name="enabled_plugin"))
    plugin_b = _make_test_plugin(PluginConfig(name="disabled_plugin"))
    registry.register(plugin_a)
    registry.register(plugin_b)

    plan = registry.plan(
        plugin_names=["enabled_plugin", "disabled_plugin"],
        disabled=["disabled_plugin"],
    )

    names = plan.ordered_names
    if "disabled_plugin" in names:
        msg = "Expected disabled_plugin to be excluded"
        raise AssertionError(msg)
    if len(plan.skipped_plugins) != 1:
        msg = f"Expected 1 skipped plugin, got {len(plan.skipped_plugins)}"
        raise AssertionError(msg)


def test_plan_cycle_detection() -> None:
    """Plan with dependency cycle raises ValueError."""
    registry = GraphPluginRegistry()
    plugin_a = _make_test_plugin(PluginConfig(name="cycle_a", depends_on=("cycle_b",)))
    plugin_b = _make_test_plugin(PluginConfig(name="cycle_b", depends_on=("cycle_a",)))
    registry.register(plugin_a)
    registry.register(plugin_b)

    with pytest.raises(ValueError, match="cycle"):
        registry.plan(plugin_names=["cycle_a", "cycle_b"])


def test_plan_missing_dependency_raises() -> None:
    """Plan with missing dependency raises ValueError."""
    registry = GraphPluginRegistry()
    plugin = _make_test_plugin(PluginConfig(name="missing_dep", depends_on=("nonexistent",)))
    registry.register(plugin)

    with pytest.raises(ValueError, match="depends on"):
        registry.plan(plugin_names=["missing_dep"])


def test_plan_missing_capability_raises() -> None:
    """Plan with missing capability raises ValueError."""
    registry = GraphPluginRegistry()
    plugin = _make_test_plugin(PluginConfig(name="missing_cap", requires=("nonexistent_cap",)))
    registry.register(plugin)

    with pytest.raises(ValueError, match="requires capability"):
        registry.plan(plugin_names=["missing_cap"])


def test_plan_ambiguous_capability_raises() -> None:
    """Plan with ambiguous capability providers raises ValueError."""
    registry = GraphPluginRegistry()
    provider1 = _make_test_plugin(PluginConfig(name="ambig_provider1", provides=("ambig_cap",)))
    provider2 = _make_test_plugin(PluginConfig(name="ambig_provider2", provides=("ambig_cap",)))
    consumer = _make_test_plugin(PluginConfig(name="ambig_consumer", requires=("ambig_cap",)))
    registry.register(provider1)
    registry.register(provider2)
    registry.register(consumer)

    with pytest.raises(ValueError, match="multiple providers"):
        registry.plan(plugin_names=["ambig_provider1", "ambig_provider2", "ambig_consumer"])


# =============================================================================
# Global Functions Tests
# =============================================================================


def test_get_graph_registry_singleton() -> None:
    """get_graph_registry returns singleton.

    Raises
    ------
    AssertionError
        If not a singleton.
    """
    reg1 = get_graph_registry()
    reg2 = get_graph_registry()

    if reg1 is not reg2:
        msg = "Expected singleton registry"
        raise AssertionError(msg)


def test_register_and_unregister_global() -> None:
    """Register and unregister with global functions.

    Raises
    ------
    AssertionError
        If registration or unregistration fails.
    """
    plugin = _make_test_plugin(PluginConfig(name="global_test_plugin"))
    registry = get_graph_registry()

    # Ensure clean state
    registry.unregister("global_test_plugin")

    register_graph_plugin(plugin)

    if not registry.contains("global_test_plugin"):
        msg = "Expected plugin to be registered globally"
        raise AssertionError(msg)

    unregister_graph_plugin("global_test_plugin")

    if registry.contains("global_test_plugin"):
        msg = "Expected plugin to be unregistered globally"
        raise AssertionError(msg)


def test_list_graph_plugins() -> None:
    """list_graph_plugins returns all plugins.

    Raises
    ------
    TypeError
        If return type is wrong.
    """
    # Should not raise
    plugins = list_graph_plugins()
    # Just verify it returns a tuple
    if not isinstance(plugins, tuple):
        msg = f"Expected tuple, got {type(plugins)}"
        raise TypeError(msg)


# =============================================================================
# Dependency Graph Tests
# =============================================================================


def test_dependency_graph() -> None:
    """Get dependency graph from registry.

    Raises
    ------
    AssertionError
        If dependency graph is wrong.
    """
    registry = GraphPluginRegistry()
    plugin_a = _make_test_plugin(PluginConfig(name="dep_graph_a"))
    plugin_b = _make_test_plugin(PluginConfig(name="dep_graph_b", depends_on=("dep_graph_a",)))
    registry.register(plugin_a)
    registry.register(plugin_b)

    dep_graph = registry.dependency_graph()

    if "dep_graph_b" not in dep_graph:
        msg = "Expected dep_graph_b in dependency graph"
        raise AssertionError(msg)
    if "dep_graph_a" not in dep_graph["dep_graph_b"]:
        msg = "Expected dep_graph_a as dependency of dep_graph_b"
        raise AssertionError(msg)


def test_metadata_for() -> None:
    """Get metadata for a plugin.

    Raises
    ------
    AssertionError
        If metadata is wrong.
    """
    registry = GraphPluginRegistry()
    plugin = _make_test_plugin(PluginConfig(name="metadata_test", kind="metric", stage="core"))
    registry.register(plugin)

    meta = registry.metadata_for("metadata_test")

    if meta.name != "metadata_test":
        msg = f"Expected name 'metadata_test', got '{meta.name}'"
        raise AssertionError(msg)
    if meta.kind != "metric":
        msg = f"Expected kind 'metric', got '{meta.kind}'"
        raise AssertionError(msg)
    if meta.stage != "core":
        msg = f"Expected stage 'core', got '{meta.stage}'"
        raise AssertionError(msg)


# =============================================================================
# Index Maintenance Tests
# =============================================================================


def test_capability_index_updated_on_register() -> None:
    """Capability index is updated when plugin is registered.

    Raises
    ------
    AssertionError
        If index is not updated.
    """
    registry = GraphPluginRegistry()
    plugin = _make_test_plugin(PluginConfig(name="cap_index_test", provides=("test_capability",)))
    registry.register(plugin)

    providers = registry.list_providing("test_capability")

    if len(providers) != 1:
        msg = f"Expected 1 provider, got {len(providers)}"
        raise AssertionError(msg)


def test_capability_index_updated_on_unregister() -> None:
    """Capability index is updated when plugin is unregistered.

    Raises
    ------
    AssertionError
        If index is not cleaned up.
    """
    registry = GraphPluginRegistry()
    plugin = _make_test_plugin(PluginConfig(name="cap_unreg_test", provides=("unreg_capability",)))
    registry.register(plugin)
    registry.unregister("cap_unreg_test")

    providers = registry.list_providing("unreg_capability")

    if len(providers) != 0:
        msg = f"Expected 0 providers after unregister, got {len(providers)}"
        raise AssertionError(msg)


def test_table_index_maintained() -> None:
    """Table index is maintained on register/unregister.

    Raises
    ------
    AssertionError
        If table index is not maintained.
    """
    registry = GraphPluginRegistry()
    plugin = _make_test_plugin(
        PluginConfig(name="table_index_test", produces_tables=("graph.index_table",))
    )
    registry.register(plugin)

    producers = registry.list_by_table("graph.index_table")
    if len(producers) != 1:
        msg = f"Expected 1 producer, got {len(producers)}"
        raise AssertionError(msg)

    registry.unregister("table_index_test")

    producers = registry.list_by_table("graph.index_table")
    if len(producers) != 0:
        msg = f"Expected 0 producers after unregister, got {len(producers)}"
        raise AssertionError(msg)
