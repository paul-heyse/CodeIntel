"""Tests for the plugin registry."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from codeintel.analytics.core.context import (
    PluginExecutionContext,
    PluginScratch,
)
from codeintel.analytics.core.protocol import PluginResult, PluginStage
from codeintel.analytics.core.registry import (
    FunctionalPlugin,
    PluginRegistry,
    plugin,
)
from codeintel.core.plugins.functional import BaseFunctionalPlugin
from codeintel.analytics.runtime.manifest import AnalyticsScope
from codeintel.config.primitives import SnapshotRef
from tests._helpers.gateway import open_ingestion_gateway

EXPECTED_TWO_PLUGINS = 2
EXPECTED_ROW_COUNT = 42


@pytest.fixture
def registry() -> PluginRegistry:
    """Provide a fresh registry for each test.

    Returns
    -------
    PluginRegistry
        New plugin registry.
    """
    return PluginRegistry()


@pytest.fixture
def make_functional_plugin() -> Callable[..., FunctionalPlugin]:
    """Create FunctionalPlugin instances via the production decorator.

    Returns
    -------
    Callable[..., FunctionalPlugin]
        Factory that accepts plugin metadata parameters.
    """

    def _factory(
        *,
        name: str,
        stage: PluginStage = "function",
        provides: tuple[str, ...] = (),
        requires: tuple[str, ...] = (),
        enabled: bool = True,
    ) -> FunctionalPlugin:
        def _impl(ctx: PluginExecutionContext) -> PluginResult:
            _ = ctx
            return PluginResult.ok(meta={"name": name})

        # Call plugin() directly with the function to get a FunctionalPlugin
        # Cast required because plugin() has union return type for decorator pattern
        return cast(
            "FunctionalPlugin",
            plugin(
                _impl,
                name=name,
                description=f"Functional plugin {name}",
                stage=stage,
                provides=list(provides),
                requires=list(requires),
                enabled_by_default=enabled,
                register=False,
            ),
        )

    return _factory


@pytest.mark.parametrize(
    "plugins_to_register",
    [
        ("test.plugin",),
        ("one.plugin", "two.plugin"),
    ],
)
def test_register_plugins(
    registry: PluginRegistry,
    make_functional_plugin: Callable[..., FunctionalPlugin],
    plugins_to_register: tuple[str, ...],
) -> None:
    """Registering plugins should make them retrievable."""
    for name in plugins_to_register:
        registry.register(make_functional_plugin(name=name))

    for name in plugins_to_register:
        assert registry.get(name).metadata.name == name


def test_register_duplicate_raises(
    registry: PluginRegistry, make_functional_plugin: Callable[..., FunctionalPlugin]
) -> None:
    """Registering a duplicate plugin should raise ValueError."""
    plugin1 = make_functional_plugin(name="test.plugin")
    plugin2 = make_functional_plugin(name="test.plugin")

    registry.register(plugin1)

    with pytest.raises(ValueError, match="Duplicate plugin name"):
        registry.register(plugin2)


def test_get_unknown_plugin_raises(registry: PluginRegistry) -> None:
    """Getting an unknown plugin should raise KeyError."""
    with pytest.raises(KeyError, match="Unknown plugin"):
        registry.get("nonexistent")


def test_list_all_plugins(
    registry: PluginRegistry, make_functional_plugin: Callable[..., FunctionalPlugin]
) -> None:
    """list_all should return every registered plugin."""
    plugin1 = make_functional_plugin(name="plugin1")
    plugin2 = make_functional_plugin(name="plugin2")

    registry.register(plugin1)
    registry.register(plugin2)

    all_plugins = registry.list_all()
    assert len(all_plugins) == EXPECTED_TWO_PLUGINS
    assert plugin1 in all_plugins
    assert plugin2 in all_plugins


def test_list_by_stage(
    registry: PluginRegistry, make_functional_plugin: Callable[..., FunctionalPlugin]
) -> None:
    """list_by_stage should filter by declared stage."""
    func_plugin = make_functional_plugin(name="func.plugin", stage="function")
    graph_plugin = make_functional_plugin(name="graph.plugin", stage="graph")

    registry.register(func_plugin)
    registry.register(graph_plugin)

    func_plugins = registry.list_by_stage("function")
    assert func_plugins == (func_plugin,)


def test_list_by_capability(
    registry: PluginRegistry, make_functional_plugin: Callable[..., FunctionalPlugin]
) -> None:
    """list_providing should find plugins by capability name."""
    plugin1 = make_functional_plugin(name="plugin1", provides=("cap.a", "cap.b"))
    plugin2 = make_functional_plugin(name="plugin2", provides=("cap.a",))
    plugin3 = make_functional_plugin(name="plugin3", provides=("cap.c",))

    registry.register(plugin1)
    registry.register(plugin2)
    registry.register(plugin3)

    cap_a_plugins = registry.list_providing("cap.a")
    assert plugin1 in cap_a_plugins
    assert plugin2 in cap_a_plugins
    assert plugin3 not in cap_a_plugins


def test_unregister_plugin(
    registry: PluginRegistry, make_functional_plugin: Callable[..., FunctionalPlugin]
) -> None:
    """Unregister should remove a plugin by name."""
    test_plugin = make_functional_plugin(name="test.plugin")

    registry.register(test_plugin)
    assert len(registry.list_all()) == 1

    registry.unregister("test.plugin")
    assert len(registry.list_all()) == 0


def test_plan_with_explicit_plugins(
    registry: PluginRegistry, make_functional_plugin: Callable[..., FunctionalPlugin]
) -> None:
    """Planning with explicit plugin list should respect provided names."""
    plugin1 = make_functional_plugin(name="plugin1")
    plugin2 = make_functional_plugin(name="plugin2")

    registry.register(plugin1)
    registry.register(plugin2)

    plan = registry.plan(["plugin1"])
    assert plan.plugins == (plugin1,)


def test_plan_respects_dependencies(
    registry: PluginRegistry, make_functional_plugin: Callable[..., FunctionalPlugin]
) -> None:
    """Dependencies should influence plugin ordering."""
    plugin_a = make_functional_plugin(name="a", provides=("cap.a",))
    plugin_b = make_functional_plugin(name="b", requires=("cap.a",))

    registry.register(plugin_a)
    registry.register(plugin_b)

    plan = registry.plan(["a", "b"])
    names = plan.ordered_names

    assert names.index("a") < names.index("b")


def test_plan_detects_cycle(
    registry: PluginRegistry, make_functional_plugin: Callable[..., FunctionalPlugin]
) -> None:
    """Cycles in dependencies should raise ValueError."""
    plugin_a = make_functional_plugin(name="a", provides=("cap.a",), requires=("cap.b",))
    plugin_b = make_functional_plugin(name="b", provides=("cap.b",), requires=("cap.a",))

    registry.register(plugin_a)
    registry.register(plugin_b)

    with pytest.raises(ValueError, match="cycle detected"):
        registry.plan(["a", "b"])


def test_plan_with_disabled(
    registry: PluginRegistry, make_functional_plugin: Callable[..., FunctionalPlugin]
) -> None:
    """Disabled plugins should be skipped during planning."""
    plugin1 = make_functional_plugin(name="plugin1")
    plugin2 = make_functional_plugin(name="plugin2")

    registry.register(plugin1)
    registry.register(plugin2)

    plan = registry.plan(["plugin1", "plugin2"], disabled=["plugin2"])

    assert plan.plugins == (plugin1,)
    assert len(plan.skipped) == 1
    assert plan.skipped[0].name == "plugin2"


def test_plan_default_enabled_only(
    registry: PluginRegistry, make_functional_plugin: Callable[..., FunctionalPlugin]
) -> None:
    """Plan without explicit names should include only enabled-by-default plugins."""
    enabled = make_functional_plugin(name="enabled", enabled=True)
    disabled = make_functional_plugin(name="disabled", enabled=False)

    registry.register(enabled)
    registry.register(disabled)

    plan = registry.plan()
    assert plan.plugins == (enabled,)


def test_plugin_decorator_simple() -> None:
    """Decorator should wrap functions into FunctionalPlugin without registration."""

    def _impl(ctx: PluginExecutionContext) -> PluginResult:
        _ = ctx
        return PluginResult.ok()

    my_plugin = cast(
        "FunctionalPlugin",
        plugin(
            _impl,
            name="decorated.plugin",
            description="A decorated plugin",
            stage="function",
            register=False,
        ),
    )

    # Use BaseFunctionalPlugin for isinstance check (subscripted generics can't be used)
    assert isinstance(my_plugin, BaseFunctionalPlugin)
    assert my_plugin.metadata.name == "decorated.plugin"
    assert my_plugin.metadata.description == "A decorated plugin"
    assert my_plugin.metadata.stage == "function"


def test_plugin_decorator_with_capabilities() -> None:
    """Decorator should apply capabilities to metadata."""

    def _impl(ctx: PluginExecutionContext) -> PluginResult:
        _ = ctx
        return PluginResult.ok()

    cap_plugin = cast(
        "FunctionalPlugin",
        plugin(
            _impl,
            name="cap.plugin",
            description="Plugin with capabilities",
            stage="graph",
            provides=["cap.output"],
            requires=["cap.input"],
            register=False,
        ),
    )

    meta = cap_plugin.metadata
    assert len(meta.provides) == 1
    assert meta.provides[0] == "cap.output"
    assert len(meta.requires) == 1
    assert meta.requires[0] == "cap.input"


def test_decorated_plugin_executes() -> None:
    """A decorated plugin should execute and return PluginResult."""
    executed = False

    def _impl(ctx: PluginExecutionContext) -> PluginResult:
        nonlocal executed
        _ = ctx
        executed = True
        return PluginResult.ok(row_counts={"test": 42})

    exec_plugin = cast(
        "FunctionalPlugin",
        plugin(
            _impl,
            name="exec.plugin",
            description="Executable",
            stage="function",
            register=False,
        ),
    )

    gateway = open_ingestion_gateway()
    ctx = PluginExecutionContext(
        gateway=gateway,
        snapshot=SnapshotRef(
            repo="test",
            commit="abc123",
            repo_root=Path(tempfile.gettempdir()),
        ),
        run_id="test-run",
        scope=AnalyticsScope(),
        scratch=PluginScratch(),
    )

    result = exec_plugin.execute(ctx)

    assert executed is True
    assert result.success is True
    assert result.row_counts.get("test") == EXPECTED_ROW_COUNT
