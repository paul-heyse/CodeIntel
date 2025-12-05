"""Test plugin registry infrastructure from codeintel.core.plugins.registry.

This module tests:
- PluginSkip and PluginPlan dataclasses
- BasePluginRegistry registration/unregistration
- Index building (_by_capability, _by_stage, _by_kind, _by_table)
- Lookup methods (get, contains, list_all, list_by_stage, etc.)
- _resolve_selection() with enabled/disabled sets
- _resolve_dependencies() capability-based resolution
- _topological_sort() with cycle detection
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, TypedDict, Unpack

import pytest

from codeintel.core.plugins.registry.base import (
    BasePluginRegistry,
    PluginPlan,
    PluginSkip,
    RegistrablePlugin,
)
from codeintel.core.plugins.types.protocol import PluginKind, PluginMetadata, PluginStage

# Test constants
EXPECTED_PLUGIN_COUNT = 2
EXPECTED_PLAN_PLUGINS = 3
EXPECTED_INDEPENDENT_COUNT = 3


class PluginMetadataOverrides(TypedDict, total=False):
    """Overrides for building plugin metadata."""

    kind: PluginKind
    stage: PluginStage
    provides: tuple[str, ...]
    requires: tuple[str, ...]
    depends_on: tuple[str, ...]
    produces_tables: tuple[str, ...]


# =============================================================================
# Test Plugin Implementation
# =============================================================================


@dataclass
class MockPlugin:
    """Mock plugin for testing registry operations."""

    _metadata: PluginMetadata

    @property
    def metadata(self) -> PluginMetadata:
        """Return the plugin metadata."""
        return self._metadata


def make_plugin(
    name: str,
    **overrides: Unpack[PluginMetadataOverrides],
) -> MockPlugin:
    """Create a mock plugin with the given configuration.

    Parameters
    ----------
    name
        Name of the plugin.
    overrides
        Keyword overrides for plugin metadata fields.

    Returns
    -------
    MockPlugin
        The mock plugin instance.
    """
    return MockPlugin(
        _metadata=PluginMetadata(
            name=name,
            description=f"Mock plugin {name}",
            kind=overrides.get("kind", "analytics"),
            stage=overrides.get("stage", "function"),
            provides=overrides.get("provides", ()),
            requires=overrides.get("requires", ()),
            depends_on=overrides.get("depends_on", ()),
            produces_tables=overrides.get("produces_tables", ()),
        ),
    )


# =============================================================================
# Test Registry Implementation
# =============================================================================


class TestRegistry(BasePluginRegistry[MockPlugin]):
    """Concrete test registry for unit tests."""

    DEFAULT_PLUGINS: ClassVar[tuple[str, ...]] = ("default1", "default2")

    @staticmethod
    def _get_default_plugins() -> Sequence[str]:
        """Return default plugin names.

        Returns
        -------
        Sequence[str]
            Default plugin names for testing.
        """
        return TestRegistry.DEFAULT_PLUGINS

    def _ensure_builtins_loaded(self) -> None:
        """No builtins to load in tests."""
        self._builtins_loaded = True

    def _ensure_entrypoints_loaded(self) -> None:
        """No entrypoints to load in tests."""
        self._entrypoints_loaded = True


@pytest.fixture
def registry() -> TestRegistry:
    """Create a fresh test registry.

    Returns
    -------
    TestRegistry
        Empty registry for testing.
    """
    return TestRegistry()


# =============================================================================
# PluginSkip Tests
# =============================================================================


def test_plugin_skip_construction() -> None:
    """Verify PluginSkip can be constructed."""
    skip = PluginSkip(name="test.plugin", reason="disabled")

    assert skip.name == "test.plugin"
    assert skip.reason == "disabled"


def test_plugin_skip_is_frozen() -> None:
    """Verify PluginSkip is immutable."""
    skip = PluginSkip(name="test", reason="test")

    with pytest.raises(AttributeError):
        skip.name = "modified"  # type: ignore[misc]


def test_plugin_skip_common_reasons() -> None:
    """Verify common skip reasons are valid strings."""
    reasons = ["disabled", "missing_dependency", "config_error", "unsupported"]

    for reason in reasons:
        skip = PluginSkip(name="test", reason=reason)
        assert skip.reason == reason


# =============================================================================
# PluginPlan Tests
# =============================================================================


def test_plugin_plan_empty() -> None:
    """Verify PluginPlan can be created with no plugins."""
    plan: PluginPlan[MockPlugin] = PluginPlan(plugins=())

    assert plan.plugins == ()
    assert plan.skipped == ()
    assert plan.plan_id is not None


def test_plugin_plan_with_plugins() -> None:
    """Verify PluginPlan holds plugins correctly."""
    plugin1 = make_plugin("plugin1")
    plugin2 = make_plugin("plugin2")

    plan: PluginPlan[MockPlugin] = PluginPlan(plugins=(plugin1, plugin2))

    assert len(plan.plugins) == EXPECTED_PLUGIN_COUNT
    assert plan.plugins[0] is plugin1
    assert plan.plugins[1] is plugin2


def test_plugin_plan_with_skipped() -> None:
    """Verify PluginPlan tracks skipped plugins."""
    skip = PluginSkip(name="skipped.plugin", reason="disabled")
    plan: PluginPlan[MockPlugin] = PluginPlan(
        plugins=(),
        skipped=(skip,),
    )

    assert len(plan.skipped) == 1
    assert plan.skipped[0].name == "skipped.plugin"


def test_plugin_plan_ordered_names() -> None:
    """Verify ordered_names returns plugin names in order."""
    plugin1 = make_plugin("first")
    plugin2 = make_plugin("second")
    plugin3 = make_plugin("third")

    plan: PluginPlan[MockPlugin] = PluginPlan(plugins=(plugin1, plugin2, plugin3))

    assert plan.ordered_names == ("first", "second", "third")


def test_plugin_plan_dep_graph() -> None:
    """Verify PluginPlan can include dependency graph."""
    plan: PluginPlan[MockPlugin] = PluginPlan(
        plugins=(),
        dep_graph={"a": ("b", "c"), "b": ()},
    )

    assert plan.dep_graph == {"a": ("b", "c"), "b": ()}


def test_plugin_plan_unique_ids() -> None:
    """Verify each plan gets a unique ID."""
    plan1: PluginPlan[MockPlugin] = PluginPlan(plugins=())
    plan2: PluginPlan[MockPlugin] = PluginPlan(plugins=())

    assert plan1.plan_id != plan2.plan_id


# =============================================================================
# BasePluginRegistry Registration Tests
# =============================================================================


def test_register_plugin(registry: TestRegistry) -> None:
    """Verify register() adds a plugin."""
    plugin = make_plugin("test.plugin")

    registry.register(plugin)

    assert registry.contains("test.plugin")


def test_register_duplicate_raises(registry: TestRegistry) -> None:
    """Verify registering duplicate name raises ValueError."""
    plugin = make_plugin("test.plugin")
    registry.register(plugin)

    with pytest.raises(ValueError, match="Duplicate plugin"):
        registry.register(make_plugin("test.plugin"))


def test_unregister_removes_plugin(registry: TestRegistry) -> None:
    """Verify unregister() removes a plugin."""
    plugin = make_plugin("test.plugin")
    registry.register(plugin)

    registry.unregister("test.plugin")

    assert not registry.contains("test.plugin")


def test_unregister_nonexistent_no_error(registry: TestRegistry) -> None:
    """Verify unregister() doesn't raise for missing plugin."""
    # Should not raise
    registry.unregister("nonexistent")


# =============================================================================
# Index Building Tests
# =============================================================================


def test_index_by_capability(registry: TestRegistry) -> None:
    """Verify plugins are indexed by capability."""
    plugin = make_plugin("provider", provides=("capability.test",))
    registry.register(plugin)

    result = registry.list_providing("capability.test")

    assert len(result) == 1
    assert result[0].metadata.name == "provider"


def test_index_by_stage(registry: TestRegistry) -> None:
    """Verify plugins are indexed by stage."""
    plugin1 = make_plugin("p1", stage="function")
    plugin2 = make_plugin("p2", stage="graph")
    registry.register(plugin1)
    registry.register(plugin2)

    function_plugins = registry.list_by_stage("function")
    graph_plugins = registry.list_by_stage("graph")

    assert len(function_plugins) == 1
    assert function_plugins[0].metadata.name == "p1"
    assert len(graph_plugins) == 1
    assert graph_plugins[0].metadata.name == "p2"


def test_index_by_kind(registry: TestRegistry) -> None:
    """Verify plugins are indexed by kind."""
    plugin1 = make_plugin("p1", kind="analytics")
    plugin2 = make_plugin("p2", kind="builder")
    registry.register(plugin1)
    registry.register(plugin2)

    analytics_plugins = registry.list_by_kind("analytics")
    builder_plugins = registry.list_by_kind("builder")

    assert len(analytics_plugins) == 1
    assert len(builder_plugins) == 1


def test_index_by_table(registry: TestRegistry) -> None:
    """Verify plugins are indexed by produced tables."""
    plugin = make_plugin(
        "producer",
        produces_tables=("analytics.metrics", "analytics.stats"),
    )
    registry.register(plugin)

    metrics_plugins = registry.list_by_table("analytics.metrics")
    stats_plugins = registry.list_by_table("analytics.stats")

    assert len(metrics_plugins) == 1
    assert len(stats_plugins) == 1


def test_unindex_on_unregister(registry: TestRegistry) -> None:
    """Verify unregister removes plugin from indices."""
    plugin = make_plugin(
        "test",
        provides=("cap",),
        produces_tables=("table",),
    )
    registry.register(plugin)
    registry.unregister("test")

    assert len(registry.list_providing("cap")) == 0
    assert len(registry.list_by_table("table")) == 0


# =============================================================================
# Lookup Tests
# =============================================================================


def test_get_returns_plugin(registry: TestRegistry) -> None:
    """Verify get() returns the registered plugin."""
    plugin = make_plugin("test.plugin")
    registry.register(plugin)

    result = registry.get("test.plugin")

    assert result is plugin


def test_get_unknown_raises(registry: TestRegistry) -> None:
    """Verify get() raises KeyError for unknown plugin."""
    with pytest.raises(KeyError, match="Unknown plugin"):
        registry.get("nonexistent")


def test_contains_true(registry: TestRegistry) -> None:
    """Verify contains() returns True for registered plugin."""
    registry.register(make_plugin("test"))

    assert registry.contains("test")


def test_contains_false(registry: TestRegistry) -> None:
    """Verify contains() returns False for unregistered plugin."""
    assert not registry.contains("nonexistent")


def test_list_all(registry: TestRegistry) -> None:
    """Verify list_all() returns all plugins."""
    registry.register(make_plugin("p1"))
    registry.register(make_plugin("p2"))
    registry.register(make_plugin("p3"))

    result = registry.list_all()

    assert len(result) == EXPECTED_PLAN_PLUGINS
    names = {p.metadata.name for p in result}
    assert names == {"p1", "p2", "p3"}


def test_list_names(registry: TestRegistry) -> None:
    """Verify list_names() returns plugin names."""
    registry.register(make_plugin("alpha"))
    registry.register(make_plugin("beta"))

    result = registry.list_names()

    assert set(result) == {"alpha", "beta"}


def test_list_by_stage_empty(registry: TestRegistry) -> None:
    """Verify list_by_stage() returns empty for no matches."""
    result = registry.list_by_stage("nonexistent")

    assert result == ()


def test_list_by_kind_empty(registry: TestRegistry) -> None:
    """Verify list_by_kind() returns empty for no matches."""
    result = registry.list_by_kind("nonexistent")

    assert result == ()


def test_list_providing_empty(registry: TestRegistry) -> None:
    """Verify list_providing() returns empty for no matches."""
    result = registry.list_providing("nonexistent.capability")

    assert result == ()


def test_metadata_for(registry: TestRegistry) -> None:
    """Verify metadata_for() returns plugin metadata."""
    plugin = make_plugin("test", kind="builder")
    registry.register(plugin)

    meta = registry.metadata_for("test")

    assert meta.name == "test"
    assert meta.kind == "builder"


def test_dependency_graph(registry: TestRegistry) -> None:
    """Verify dependency_graph() returns correct mapping."""
    registry.register(make_plugin("a", depends_on=("b", "c")))
    registry.register(make_plugin("b", depends_on=("c",)))
    registry.register(make_plugin("c", depends_on=()))

    graph = registry.dependency_graph()

    assert graph == {
        "a": ("b", "c"),
        "b": ("c",),
        "c": (),
    }


# =============================================================================
# Selection Resolution Tests
# =============================================================================


def test_resolve_selection_with_enabled(registry: TestRegistry) -> None:
    """Verify _resolve_selection uses enabled list."""
    registry.register(make_plugin("a"))
    registry.register(make_plugin("b"))
    registry.register(make_plugin("c"))

    selected, _skipped = registry._resolve_selection(  # noqa: SLF001
        plugin_names=None,
        enabled=["a", "b"],
        disabled=None,
        defaults=["c"],
    )

    assert set(selected.keys()) == {"a", "b"}


def test_resolve_selection_with_disabled(registry: TestRegistry) -> None:
    """Verify _resolve_selection excludes disabled plugins."""
    registry.register(make_plugin("a"))
    registry.register(make_plugin("b"))
    registry.register(make_plugin("c"))

    selected, skipped = registry._resolve_selection(  # noqa: SLF001
        plugin_names=["a", "b", "c"],
        enabled=None,
        disabled=["b"],
        defaults=[],
    )

    assert set(selected.keys()) == {"a", "c"}
    assert len(skipped) == 1
    assert skipped[0].name == "b"
    assert skipped[0].reason == "disabled"


def test_resolve_selection_uses_defaults(registry: TestRegistry) -> None:
    """Verify _resolve_selection uses defaults when no selection."""
    registry.register(make_plugin("default1"))
    registry.register(make_plugin("default2"))

    selected, _ = registry._resolve_selection(  # noqa: SLF001
        plugin_names=None,
        enabled=None,
        disabled=None,
        defaults=["default1", "default2"],
    )

    assert set(selected.keys()) == {"default1", "default2"}


def test_resolve_selection_missing_plugin(registry: TestRegistry) -> None:
    """Verify _resolve_selection skips missing plugins."""
    registry.register(make_plugin("exists"))

    selected, skipped = registry._resolve_selection(  # noqa: SLF001
        plugin_names=["exists", "missing"],
        enabled=None,
        disabled=None,
        defaults=[],
    )

    assert set(selected.keys()) == {"exists"}
    assert len(skipped) == 1
    assert skipped[0].name == "missing"
    assert skipped[0].reason == "missing_dependency"


# =============================================================================
# Dependency Resolution Tests
# =============================================================================


def test_resolve_dependencies_explicit(registry: TestRegistry) -> None:
    """Verify _resolve_dependencies handles explicit depends_on."""
    a = make_plugin("a", depends_on=("b",))
    b = make_plugin("b", depends_on=())

    selected = {"a": a, "b": b}
    deps = registry._resolve_dependencies(selected)  # noqa: SLF001

    assert deps["a"] == {"b"}
    assert deps["b"] == set()


def test_resolve_dependencies_capability_based(registry: TestRegistry) -> None:
    """Verify _resolve_dependencies handles capability requirements."""
    consumer = make_plugin("consumer", requires=("data.source",))
    provider = make_plugin("provider", provides=("data.source",))

    selected = {"consumer": consumer, "provider": provider}
    deps = registry._resolve_dependencies(selected)  # noqa: SLF001

    assert "provider" in deps["consumer"]


def test_resolve_dependencies_missing_capability(registry: TestRegistry) -> None:
    """Verify _resolve_dependencies handles missing capability gracefully."""
    consumer = make_plugin("consumer", requires=("missing.capability",))

    selected = {"consumer": consumer}
    deps = registry._resolve_dependencies(selected)  # noqa: SLF001

    # Should not raise, just logs warning
    assert deps["consumer"] == set()


def test_resolve_dependencies_self_provide(registry: TestRegistry) -> None:
    """Verify _resolve_dependencies doesn't add self as dependency."""
    plugin = make_plugin(
        "self_sufficient",
        provides=("capability",),
        requires=("capability",),
    )

    selected = {"self_sufficient": plugin}
    deps = registry._resolve_dependencies(selected)  # noqa: SLF001

    assert "self_sufficient" not in deps["self_sufficient"]


# =============================================================================
# Topological Sort Tests
# =============================================================================


def test_topological_sort_linear(registry: TestRegistry) -> None:
    """Verify _topological_sort handles linear dependencies."""
    a = make_plugin("a")
    b = make_plugin("b")
    c = make_plugin("c")

    selected = {"a": a, "b": b, "c": c}
    deps = {"a": {"b"}, "b": {"c"}, "c": set()}

    result = registry._topological_sort(selected, deps)  # noqa: SLF001
    names = [p.metadata.name for p in result]

    # c must come before b, b before a
    assert names.index("c") < names.index("b")
    assert names.index("b") < names.index("a")


def test_topological_sort_diamond(registry: TestRegistry) -> None:
    """Verify _topological_sort handles diamond dependencies."""
    # d depends on b and c, which both depend on a
    a = make_plugin("a")
    b = make_plugin("b")
    c = make_plugin("c")
    d = make_plugin("d")

    selected = {"a": a, "b": b, "c": c, "d": d}
    deps = {"a": set(), "b": {"a"}, "c": {"a"}, "d": {"b", "c"}}

    result = registry._topological_sort(selected, deps)  # noqa: SLF001
    names = [p.metadata.name for p in result]

    # a must come first, d must come last
    assert names[0] == "a"
    assert names[-1] == "d"


def test_topological_sort_cycle_detection(registry: TestRegistry) -> None:
    """Verify _topological_sort detects cycles."""
    a = make_plugin("a")
    b = make_plugin("b")

    selected = {"a": a, "b": b}
    deps = {"a": {"b"}, "b": {"a"}}  # Cycle!

    with pytest.raises(ValueError, match=r"[Cc]ycle"):
        registry._topological_sort(selected, deps)  # noqa: SLF001


def test_topological_sort_independent(registry: TestRegistry) -> None:
    """Verify _topological_sort handles independent plugins."""
    a = make_plugin("a")
    b = make_plugin("b")
    c = make_plugin("c")

    selected = {"a": a, "b": b, "c": c}
    deps = {"a": set(), "b": set(), "c": set()}

    result = registry._topological_sort(selected, deps)  # noqa: SLF001

    # All plugins should be in result
    assert len(result) == EXPECTED_INDEPENDENT_COUNT


# =============================================================================
# RegistrablePlugin Protocol Tests
# =============================================================================


def test_mock_plugin_implements_registrable() -> None:
    """Verify MockPlugin implements RegistrablePlugin protocol."""
    plugin = make_plugin("test")

    assert isinstance(plugin, RegistrablePlugin)


def test_non_conforming_not_registrable() -> None:
    """Verify non-conforming classes don't pass protocol check."""

    class NotAPlugin:
        pass

    assert not isinstance(NotAPlugin(), RegistrablePlugin)
