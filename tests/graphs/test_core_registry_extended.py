"""Extended tests for graph plugin registry.

This module provides additional test coverage for the registry module
from `codeintel.graphs.core.registry`, including:

- Plugin registration and unregistration
- Duplicate plugin name handling
- Plugin lookup and query methods
- Dependency resolution and validation
- Topological sorting with cycle detection
- Capability-based dependency resolution
- Plan building with skip tracking
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields, replace
from typing import Final

import pytest

from codeintel.core.plugins.types.result import PluginResult
from codeintel.graphs.core.context import GraphPluginExecutionContext
from codeintel.graphs.core.protocol import (
    GraphPluginKind,
    GraphPluginMetadata,
    GraphPluginProtocol,
    GraphPluginSkip,
    GraphPluginStage,
)
from codeintel.graphs.core.registry import (
    DependencyPolicy,
    GraphPluginRegistry,
    PlanningOptions,
    SelectionPolicy,
    get_graph_registry,
    register_graph_plugin,
    unregister_graph_plugin,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_length,
    expect_true,
)
from tests._helpers.fakes.graph_plugins import FakeGraphPlugin

# Constants
TEST_PLUGIN_PREFIX: Final = "_test_registry_"


@dataclass
class PluginConfig:
    """Configuration for constructing registry test plugins."""

    kind: GraphPluginKind = "builder"
    stage: GraphPluginStage = "goid"
    depends_on: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    produces_tables: tuple[str, ...] = ()


PLUGIN_CONFIG_FIELDS: Final = {field.name for field in fields(PluginConfig)}


def _resolve_plugin_config(
    config: PluginConfig | None, overrides: dict[str, object]
) -> PluginConfig:
    """Merge a base plugin config with validated overrides.

    Parameters
    ----------
    config
        Base plugin configuration or None for defaults.
    overrides
        Override values keyed by PluginConfig field names.

    Returns
    -------
    PluginConfig
        Combined plugin configuration.

    Raises
    ------
    ValueError
        If overrides contain unsupported keys.
    """
    unknown_keys = set(overrides) - PLUGIN_CONFIG_FIELDS
    if unknown_keys:
        message = f"Unsupported plugin config overrides: {sorted(unknown_keys)}"
        raise ValueError(message)
    base_config = config or PluginConfig()
    if not overrides:
        return base_config
    return replace(base_config, **overrides)


# Test Helpers


def _make_test_plugin(
    name: str, *, config: PluginConfig | None = None, **overrides: object
) -> GraphPluginProtocol:
    """Create a configurable test plugin.

    Parameters
    ----------
    name
        Plugin name.
    config
        Base configuration for the plugin.
    **overrides
        Overrides for plugin configuration fields.

    Returns
    -------
    GraphPluginProtocol
        Test plugin instance.
    """
    plugin_config = _resolve_plugin_config(config, overrides)

    def execute(_ctx: GraphPluginExecutionContext) -> PluginResult:
        return PluginResult.ok()

    metadata = GraphPluginMetadata(
        name=f"{TEST_PLUGIN_PREFIX}{name}",
        description=f"Test plugin {name}",
        kind=plugin_config.kind,
        stage=plugin_config.stage,
        depends_on=plugin_config.depends_on,
        requires=plugin_config.requires,
        provides=plugin_config.provides,
        produces_tables=plugin_config.produces_tables,
    )

    return FakeGraphPlugin(_metadata=metadata, _execute_fn=execute)


@pytest.fixture
def fresh_registry() -> GraphPluginRegistry:
    """Create a fresh registry instance for testing.

    Returns
    -------
    GraphPluginRegistry
        Fresh registry instance (not the global singleton).
    """
    return GraphPluginRegistry()


@pytest.fixture(autouse=True)
def cleanup_test_plugins() -> Iterator[None]:
    """Clean up test plugins from global registry after tests.

    Yields
    ------
    None
        Runs test, then cleans up.
    """
    yield
    # Clean up any test plugins from global registry
    registry = get_graph_registry()
    for name in registry.list_names():
        if name.startswith(TEST_PLUGIN_PREFIX):
            registry.unregister(name)


# Tests: Plugin Registration


def test_register_plugin_basic(fresh_registry: GraphPluginRegistry) -> None:
    """Register a plugin successfully."""
    plugin = _make_test_plugin("basic")

    fresh_registry.register(plugin)

    expect_true(fresh_registry.contains(plugin.metadata.name))


def test_register_plugin_duplicate_raises(fresh_registry: GraphPluginRegistry) -> None:
    """Registering duplicate plugin name raises ValueError."""
    plugin1 = _make_test_plugin("duplicate")
    plugin2 = _make_test_plugin("duplicate")

    fresh_registry.register(plugin1)

    with pytest.raises(ValueError, match="Duplicate plugin name"):
        fresh_registry.register(plugin2)


def test_register_plugin_indexes_by_capability(fresh_registry: GraphPluginRegistry) -> None:
    """Registered plugin is indexed by capabilities."""
    plugin = _make_test_plugin("cap", provides=("test_capability",))

    fresh_registry.register(plugin)

    providers = fresh_registry.list_providing("test_capability")
    expect_in(plugin, providers)


def test_register_plugin_indexes_by_kind(fresh_registry: GraphPluginRegistry) -> None:
    """Registered plugin is indexed by kind."""
    plugin = _make_test_plugin("kind", kind="metric")

    fresh_registry.register(plugin)

    plugins_by_kind = fresh_registry.list_by_kind("metric")
    expect_in(plugin, plugins_by_kind)


def test_register_plugin_indexes_by_stage(fresh_registry: GraphPluginRegistry) -> None:
    """Registered plugin is indexed by stage."""
    plugin = _make_test_plugin("stage", stage="core")

    fresh_registry.register(plugin)

    plugins_by_stage = fresh_registry.list_by_stage("core")
    expect_in(plugin, plugins_by_stage)


def test_register_plugin_indexes_by_table(fresh_registry: GraphPluginRegistry) -> None:
    """Registered plugin is indexed by produced tables."""
    plugin = _make_test_plugin("table", produces_tables=("analytics.test_table",))

    fresh_registry.register(plugin)

    plugins_by_table = fresh_registry.list_by_table("analytics.test_table")
    expect_in(plugin, plugins_by_table)


# Tests: Plugin Unregistration


def test_unregister_plugin_removes(fresh_registry: GraphPluginRegistry) -> None:
    """Unregister removes plugin from registry."""
    plugin = _make_test_plugin("unregister")
    fresh_registry.register(plugin)

    fresh_registry.unregister(plugin.metadata.name)

    expect_false(fresh_registry.contains(plugin.metadata.name))


def test_unregister_removes_from_indexes(fresh_registry: GraphPluginRegistry) -> None:
    """Unregister removes plugin from all indexes."""
    plugin = _make_test_plugin(
        "unregister_idx",
        kind="metric",
        stage="core",
        provides=("test_cap",),
        produces_tables=("test_table",),
    )
    fresh_registry.register(plugin)

    fresh_registry.unregister(plugin.metadata.name)

    expect_true(plugin not in fresh_registry.list_by_kind("metric"))
    expect_true(plugin not in fresh_registry.list_by_stage("core"))
    expect_true(plugin not in fresh_registry.list_providing("test_cap"))
    expect_true(plugin not in fresh_registry.list_by_table("test_table"))


def test_unregister_nonexistent_silent(fresh_registry: GraphPluginRegistry) -> None:
    """Unregister nonexistent plugin does nothing."""
    # Should not raise
    fresh_registry.unregister("nonexistent_plugin")


# Tests: Plugin Lookup


def test_get_plugin_returns_registered(fresh_registry: GraphPluginRegistry) -> None:
    """Get returns registered plugin."""
    plugin = _make_test_plugin("get")
    fresh_registry.register(plugin)

    retrieved = fresh_registry.get(plugin.metadata.name)

    expect_true(retrieved is plugin)


def test_get_plugin_unknown_raises(fresh_registry: GraphPluginRegistry) -> None:
    """Get raises KeyError for unknown plugin."""
    with pytest.raises(KeyError, match="Unknown plugin"):
        fresh_registry.get("nonexistent")


def test_contains_returns_true_for_registered(fresh_registry: GraphPluginRegistry) -> None:
    """Contains returns True for registered plugin."""
    plugin = _make_test_plugin("contains")
    fresh_registry.register(plugin)

    expect_true(fresh_registry.contains(plugin.metadata.name))


def test_contains_returns_false_for_unknown(fresh_registry: GraphPluginRegistry) -> None:
    """Contains returns False for unknown plugin."""
    expect_false(fresh_registry.contains("nonexistent"))


# Tests: List Methods


def test_list_all_returns_registered(fresh_registry: GraphPluginRegistry) -> None:
    """List all returns all registered plugins."""
    plugins = [
        _make_test_plugin("list1"),
        _make_test_plugin("list2"),
        _make_test_plugin("list3"),
    ]

    for plugin in plugins:
        fresh_registry.register(plugin)

    all_plugins = fresh_registry.list_all()

    for plugin in plugins:
        expect_in(plugin, all_plugins)


def test_list_names_returns_names(fresh_registry: GraphPluginRegistry) -> None:
    """List names returns plugin names."""
    plugins = [
        _make_test_plugin("name1"),
        _make_test_plugin("name2"),
    ]

    for plugin in plugins:
        fresh_registry.register(plugin)

    names = fresh_registry.list_names()

    for plugin in plugins:
        expect_in(plugin.metadata.name, names)


def test_list_providing_empty_for_unknown_capability(
    fresh_registry: GraphPluginRegistry,
) -> None:
    """List providing returns empty for unknown capability."""
    result = fresh_registry.list_providing("unknown_capability")
    expect_equal(result, ())


def test_list_by_kind_empty_for_unknown_kind(
    fresh_registry: GraphPluginRegistry,
) -> None:
    """List by kind returns empty for unknown kind."""
    result = fresh_registry.list_by_kind("unknown_kind")
    expect_equal(result, ())


def test_list_by_stage_empty_for_unknown_stage(
    fresh_registry: GraphPluginRegistry,
) -> None:
    """List by stage returns empty for unknown stage."""
    result = fresh_registry.list_by_stage("unknown_stage")
    expect_equal(result, ())


# Tests: Dependency Resolution


def test_resolve_dependencies_explicit(fresh_registry: GraphPluginRegistry) -> None:
    """Dependencies resolved via explicit depends_on."""
    dep_name = f"{TEST_PLUGIN_PREFIX}dependency"
    dep_plugin = _make_test_plugin("dependency")
    main_plugin = _make_test_plugin("main", depends_on=(dep_name,))

    fresh_registry.register(dep_plugin)
    fresh_registry.register(main_plugin)

    plan = fresh_registry.plan(plugin_names=[dep_name, main_plugin.metadata.name])

    # Dependency should come before main
    plugin_names = [p.metadata.name for p in plan.plugins]
    expect_true(plugin_names.index(dep_name) < plugin_names.index(main_plugin.metadata.name))


def test_resolve_dependencies_missing_raises(fresh_registry: GraphPluginRegistry) -> None:
    """Missing dependency raises ValueError."""
    main_plugin = _make_test_plugin("missing_dep", depends_on=(f"{TEST_PLUGIN_PREFIX}nonexistent",))
    fresh_registry.register(main_plugin)

    with pytest.raises(ValueError, match=r"depends on.*not registered"):
        fresh_registry.plan(plugin_names=[main_plugin.metadata.name])


@pytest.mark.parametrize(
    ("selection_policy", "raises"),
    [
        (SelectionPolicy.LENIENT, False),
        (SelectionPolicy.STRICT, True),
    ],
)
def test_unknown_plugin_selection_policy(
    fresh_registry: GraphPluginRegistry, *, selection_policy: SelectionPolicy, raises: bool
) -> None:
    """Unknown requested plugins are skipped in lenient mode and raise in strict mode."""
    plan_opts = PlanningOptions(
        selection_policy=selection_policy,
        requested_required=selection_policy is SelectionPolicy.STRICT,
    )
    if raises:
        with pytest.raises(ValueError, match="is not registered"):
            fresh_registry.plan(plugin_names=["unknown_plugin"], plan_options=plan_opts)
        return

    plan = fresh_registry.plan(plugin_names=["unknown_plugin"], plan_options=plan_opts)
    skipped = {skip.name: skip.reason for skip in plan.skipped_plugins}
    expect_equal(skipped.get("unknown_plugin"), "missing_graph")
    expect_true("unknown_plugin" in plan.dep_graph)


@pytest.mark.parametrize(
    ("dependency_policy", "allow_missing", "raises"),
    [
        (DependencyPolicy.SKIP, False, False),
        (DependencyPolicy.STRICT, False, True),
        (DependencyPolicy.SKIP, True, False),
    ],
)
def test_missing_dependency_policy_controls_skip(
    fresh_registry: GraphPluginRegistry,
    *,
    dependency_policy: DependencyPolicy,
    allow_missing: bool,
    raises: bool,
) -> None:
    """Missing dependencies can be skipped or raised depending on policy."""
    dep_name = f"{TEST_PLUGIN_PREFIX}missing_dep"
    main_plugin = _make_test_plugin("needs_missing", depends_on=(dep_name,))
    fresh_registry.register(main_plugin)

    plan_opts = PlanningOptions(
        dependency_policy=dependency_policy,
        allow_missing_dependencies=allow_missing,
    )
    if raises:
        with pytest.raises(ValueError, match="not registered"):
            fresh_registry.plan(plugin_names=[main_plugin.metadata.name], plan_options=plan_opts)
        return

    plan = fresh_registry.plan(plugin_names=[main_plugin.metadata.name], plan_options=plan_opts)
    skipped = {skip.name: skip.reason for skip in plan.skipped_plugins}
    expect_equal(skipped.get(dep_name), "missing_dependency")
    expect_true(dep_name in plan.dep_graph)


def test_resolve_dependencies_by_capability(fresh_registry: GraphPluginRegistry) -> None:
    """Dependencies resolved via requires capability."""
    provider = _make_test_plugin("provider", provides=("test_data",))
    consumer = _make_test_plugin("consumer", requires=("test_data",))

    fresh_registry.register(provider)
    fresh_registry.register(consumer)

    plan = fresh_registry.plan(plugin_names=[provider.metadata.name, consumer.metadata.name])

    plugin_names = [p.metadata.name for p in plan.plugins]
    expect_true(
        plugin_names.index(provider.metadata.name) < plugin_names.index(consumer.metadata.name)
    )


def test_resolve_dependencies_missing_capability_raises(
    fresh_registry: GraphPluginRegistry,
) -> None:
    """Missing required capability raises ValueError."""
    consumer = _make_test_plugin("cap_consumer", requires=("missing_cap",))
    fresh_registry.register(consumer)

    with pytest.raises(ValueError, match=r"requires capability.*no provider"):
        fresh_registry.plan(plugin_names=[consumer.metadata.name])


def test_resolve_dependencies_ambiguous_capability_raises(
    fresh_registry: GraphPluginRegistry,
) -> None:
    """Ambiguous capability provider raises ValueError."""
    provider1 = _make_test_plugin("provider1", provides=("shared_cap",))
    provider2 = _make_test_plugin("provider2", provides=("shared_cap",))
    consumer = _make_test_plugin("ambig_consumer", requires=("shared_cap",))

    fresh_registry.register(provider1)
    fresh_registry.register(provider2)
    fresh_registry.register(consumer)

    with pytest.raises(ValueError, match=r"multiple providers.*disambiguate"):
        fresh_registry.plan(
            plugin_names=[
                provider1.metadata.name,
                provider2.metadata.name,
                consumer.metadata.name,
            ]
        )


# Tests: Topological Sorting


def test_topological_sort_orders_dependencies(fresh_registry: GraphPluginRegistry) -> None:
    """Topological sort orders plugins by dependencies."""
    p1_name = f"{TEST_PLUGIN_PREFIX}sort1"
    p2_name = f"{TEST_PLUGIN_PREFIX}sort2"
    p3_name = f"{TEST_PLUGIN_PREFIX}sort3"

    p1 = _make_test_plugin("sort1")  # No deps
    p2 = _make_test_plugin("sort2", depends_on=(p1_name,))  # Depends on p1
    p3 = _make_test_plugin("sort3", depends_on=(p2_name,))  # Depends on p2

    fresh_registry.register(p1)
    fresh_registry.register(p2)
    fresh_registry.register(p3)

    plan = fresh_registry.plan(plugin_names=[p3_name, p1_name, p2_name])

    names = [p.metadata.name for p in plan.plugins]
    expect_true(names.index(p1_name) < names.index(p2_name))
    expect_true(names.index(p2_name) < names.index(p3_name))


def test_topological_sort_cycle_detection(fresh_registry: GraphPluginRegistry) -> None:
    """Topological sort detects dependency cycles."""
    p1_name = f"{TEST_PLUGIN_PREFIX}cycle1"
    p2_name = f"{TEST_PLUGIN_PREFIX}cycle2"

    # Create a cycle: p1 -> p2 -> p1
    p1 = _make_test_plugin("cycle1", depends_on=(p2_name,))
    p2 = _make_test_plugin("cycle2", depends_on=(p1_name,))

    fresh_registry.register(p1)
    fresh_registry.register(p2)

    with pytest.raises(ValueError, match="Dependency cycle detected"):
        fresh_registry.plan(plugin_names=[p1_name, p2_name])


# Tests: Plan Building


def test_plan_includes_plan_id(fresh_registry: GraphPluginRegistry) -> None:
    """Plan includes a unique plan ID."""
    plugin = _make_test_plugin("plan_id")
    fresh_registry.register(plugin)

    plan = fresh_registry.plan(plugin_names=[plugin.metadata.name])

    expect_true(bool(plan.plan_id))
    expect_true(len(plan.plan_id) > 0)


def test_plan_includes_dep_graph(fresh_registry: GraphPluginRegistry) -> None:
    """Plan includes dependency graph mapping."""
    p1_name = f"{TEST_PLUGIN_PREFIX}dep1"
    p2_name = f"{TEST_PLUGIN_PREFIX}dep2"

    p1 = _make_test_plugin("dep1")
    p2 = _make_test_plugin("dep2", depends_on=(p1_name,))

    fresh_registry.register(p1)
    fresh_registry.register(p2)

    plan = fresh_registry.plan(plugin_names=[p1_name, p2_name])

    expect_in(p2_name, plan.dep_graph)
    expect_in(p1_name, plan.dep_graph[p2_name])


def test_plan_tracks_skipped_plugins(fresh_registry: GraphPluginRegistry) -> None:
    """Plan tracks skipped plugins with reasons."""
    plugin = _make_test_plugin("skipped_disabled")
    fresh_registry.register(plugin)

    plan = fresh_registry.plan(
        plugin_names=[plugin.metadata.name],
        disabled=[plugin.metadata.name],
    )

    skipped_names = [s.name for s in plan.skipped_plugins]
    expect_in(plugin.metadata.name, skipped_names)

    skipped = next(s for s in plan.skipped_plugins if s.name == plugin.metadata.name)
    expect_equal(skipped.reason, "disabled")


def test_plan_skips_unknown_plugins(fresh_registry: GraphPluginRegistry) -> None:
    """Plan skips unknown plugins with missing_dependency reason."""
    plan = fresh_registry.plan(
        plugin_names=["nonexistent_plugin"],
        plan_options=PlanningOptions(requested_required=False),
    )

    skipped = plan.skipped_plugins
    expect_length(skipped, 1)
    expect_equal(skipped[0].name, "nonexistent_plugin")
    expect_equal(skipped[0].reason, "missing_graph")


def test_plan_duplicate_plugin_raises(fresh_registry: GraphPluginRegistry) -> None:
    """Plan raises for duplicate plugin names."""
    plugin = _make_test_plugin("dup_plan")
    fresh_registry.register(plugin)

    with pytest.raises(ValueError, match="listed more than once"):
        fresh_registry.plan(plugin_names=[plugin.metadata.name, plugin.metadata.name])


def test_plan_with_enabled_overrides_defaults(fresh_registry: GraphPluginRegistry) -> None:
    """Plan with enabled parameter overrides defaults."""
    p1 = _make_test_plugin("enabled1")
    p2 = _make_test_plugin("enabled2")

    fresh_registry.register(p1)
    fresh_registry.register(p2)

    plan = fresh_registry.plan(
        enabled=[p1.metadata.name],
        defaults=[p2.metadata.name],  # This should be ignored
    )

    plugin_names = [p.metadata.name for p in plan.plugins]
    expect_in(p1.metadata.name, plugin_names)
    expect_true(p2.metadata.name not in plugin_names)


# Tests: Global Registry Functions


def test_get_graph_registry_returns_singleton() -> None:
    """get_graph_registry returns singleton instance."""
    reg1 = get_graph_registry()
    reg2 = get_graph_registry()

    expect_true(reg1 is reg2)


def test_register_graph_plugin_uses_global() -> None:
    """register_graph_plugin adds to global registry."""
    plugin = _make_test_plugin("global_register")

    register_graph_plugin(plugin)

    expect_true(get_graph_registry().contains(plugin.metadata.name))


def test_unregister_graph_plugin_uses_global() -> None:
    """unregister_graph_plugin removes from global registry."""
    plugin = _make_test_plugin("global_unregister")
    register_graph_plugin(plugin)

    unregister_graph_plugin(plugin.metadata.name)

    expect_false(get_graph_registry().contains(plugin.metadata.name))


# Tests: Metadata Access


def test_metadata_for_returns_metadata(fresh_registry: GraphPluginRegistry) -> None:
    """metadata_for returns plugin metadata."""
    plugin = _make_test_plugin("metadata", kind="metric", stage="core", provides=("test_cap",))
    fresh_registry.register(plugin)

    meta = fresh_registry.metadata_for(plugin.metadata.name)

    expect_equal(meta.name, plugin.metadata.name)
    expect_equal(meta.kind, "metric")
    expect_equal(meta.stage, "core")
    expect_in("test_cap", meta.provides)


def test_dependency_graph_returns_deps(fresh_registry: GraphPluginRegistry) -> None:
    """dependency_graph returns mapping of dependencies."""
    p1_name = f"{TEST_PLUGIN_PREFIX}dep_graph1"
    p1 = _make_test_plugin("dep_graph1")
    p2 = _make_test_plugin("dep_graph2", depends_on=(p1_name,))

    fresh_registry.register(p1)
    fresh_registry.register(p2)

    dep_graph = fresh_registry.dependency_graph()

    expect_in(p1_name, dep_graph)
    expect_in(p2.metadata.name, dep_graph)
    expect_in(p1_name, dep_graph[p2.metadata.name])


# Tests: GraphPluginPlan Attributes


def test_graph_plugin_plan_plugins_tuple(fresh_registry: GraphPluginRegistry) -> None:
    """GraphPluginPlan.plugins is a tuple."""
    plugin = _make_test_plugin("plan_tuple")
    fresh_registry.register(plugin)

    plan = fresh_registry.plan(plugin_names=[plugin.metadata.name])

    expect_is_instance(plan.plugins, tuple)


def test_graph_plugin_plan_skipped_tuple(fresh_registry: GraphPluginRegistry) -> None:
    """GraphPluginPlan.skipped_plugins is a tuple."""
    plan = fresh_registry.plan(
        plugin_names=["unknown"],
        plan_options=PlanningOptions(
            selection_policy=SelectionPolicy.LENIENT,
            requested_required=False,
        ),
    )

    expect_is_instance(plan.skipped_plugins, tuple)


def test_graph_plugin_skip_has_name_and_reason() -> None:
    """GraphPluginSkip has name and reason attributes."""
    skip = GraphPluginSkip(name="skipped_plugin", reason="disabled")

    expect_equal(skip.name, "skipped_plugin")
    expect_equal(skip.reason, "disabled")
