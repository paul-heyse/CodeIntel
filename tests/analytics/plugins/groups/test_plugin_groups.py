"""Tests for plugin groups protocol and implementation.

This module tests:
- PluginGroup dataclass
- GroupRegistry for managing groups
- Dependency resolution
- Plugin lookup
"""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.analytics.core.protocol import PluginMetadata
from codeintel.analytics.plugins.groups import (
    ALL_GROUPS,
    FUNCTION_PLUGINS,
    GRAPH_PLUGINS,
    RISK_PLUGINS,
    PluginGroup,
)
from codeintel.analytics.plugins.groups.protocol import GroupRegistry
from tests._helpers import assert_frozen

EXPECTED_GROUPS_COUNT = 3
DEPENDENCY_ORDER_EXPECTED = 2


@dataclass
class MockPlugin:
    """Mock plugin implementing AnalyticsPluginProtocol for testing."""

    name: str

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        PluginMetadata
            Plugin metadata.
        """
        return PluginMetadata(
            name=self.name,
            description=f"Mock plugin {self.name}",
            kind="analytics",
            stage="function",
            version="1.0.0",
        )


class MockPluginRegistry:
    """Mock registry for testing get_plugins.

    Attributes
    ----------
    plugins
        Mapping of plugin names to mock plugins.
    """

    def __init__(self, plugins: dict[str, MockPlugin] | None = None) -> None:
        """Initialize with plugins.

        Parameters
        ----------
        plugins
            Initial plugins to register.
        """
        self.plugins: dict[str, MockPlugin] = plugins or {}

    def get(self, name: str) -> MockPlugin:
        """Get a plugin by name.

        Parameters
        ----------
        name
            Plugin name.

        Returns
        -------
        MockPlugin
            The plugin.

        Raises
        ------
        KeyError
            If plugin not found.
        """
        if name not in self.plugins:
            msg = f"Plugin {name} not found"
            raise KeyError(msg)
        return self.plugins[name]


def test_plugin_group_creation() -> None:
    """Create a PluginGroup with all attributes."""
    group = PluginGroup(
        name="test_group",
        description="Test group description",
        plugins=("plugin.a", "plugin.b"),
        default_order="dependency",
        requires=("other_group",),
        tags=("test", "example"),
        enabled=True,
    )

    assert group.name == "test_group"
    assert group.description == "Test group description"
    assert group.plugins == ("plugin.a", "plugin.b")
    assert group.default_order == "dependency"
    assert group.requires == ("other_group",)
    assert group.tags == ("test", "example")
    assert group.enabled is True


def test_plugin_group_defaults() -> None:
    """PluginGroup has sensible defaults."""
    group = PluginGroup(name="minimal")

    assert not group.description
    assert group.plugins == ()
    assert group.default_order == "dependency"
    assert group.requires == ()
    assert group.tags == ()
    assert group.enabled is True


def test_plugin_group_immutable() -> None:
    """PluginGroup is frozen/immutable."""
    group = PluginGroup(name="test")
    assert_frozen(group, "name", "other")


def test_plugin_group_get_plugin_names() -> None:
    """Get plugin names from group."""
    group = PluginGroup(
        name="test",
        plugins=("plugin.a", "plugin.b", "plugin.c"),
    )

    names = group.get_plugin_names()

    assert names == ("plugin.a", "plugin.b", "plugin.c")


def test_plugin_group_with_plugins() -> None:
    """Add plugins to a group."""
    group = PluginGroup(
        name="original",
        plugins=("plugin.a",),
    )

    new_group = group.with_plugins("plugin.b", "plugin.c")

    assert new_group.name == "original"
    assert new_group.plugins == ("plugin.a", "plugin.b", "plugin.c")
    # Original unchanged
    assert group.plugins == ("plugin.a",)


def test_plugin_group_without_plugins() -> None:
    """Remove plugins from a group."""
    group = PluginGroup(
        name="original",
        plugins=("plugin.a", "plugin.b", "plugin.c"),
    )

    new_group = group.without_plugins("plugin.b")

    assert new_group.name == "original"
    assert new_group.plugins == ("plugin.a", "plugin.c")
    # Original unchanged
    assert group.plugins == ("plugin.a", "plugin.b", "plugin.c")


def test_plugin_group_without_multiple_plugins() -> None:
    """Remove multiple plugins from a group."""
    group = PluginGroup(
        name="test",
        plugins=("a", "b", "c", "d"),
    )

    new_group = group.without_plugins("a", "c")

    assert new_group.plugins == ("b", "d")


def test_plugin_group_get_plugins_found() -> None:
    """Get plugins from registry when they exist."""
    group = PluginGroup(
        name="test",
        plugins=("plugin.a", "plugin.b"),
    )
    registry = MockPluginRegistry(
        {
            "plugin.a": MockPlugin("plugin.a"),
            "plugin.b": MockPlugin("plugin.b"),
        }
    )

    plugins = group.get_plugins(registry)  # type: ignore[arg-type]

    assert len(plugins) == DEPENDENCY_ORDER_EXPECTED
    assert plugins[0].metadata.name == "plugin.a"
    assert plugins[1].metadata.name == "plugin.b"


def test_plugin_group_get_plugins_partial() -> None:
    """Get plugins skips missing ones."""
    group = PluginGroup(
        name="test",
        plugins=("plugin.a", "plugin.missing", "plugin.b"),
    )
    registry = MockPluginRegistry(
        {
            "plugin.a": MockPlugin("plugin.a"),
            "plugin.b": MockPlugin("plugin.b"),
        }
    )

    plugins = group.get_plugins(registry)  # type: ignore[arg-type]

    assert len(plugins) == DEPENDENCY_ORDER_EXPECTED
    assert plugins[0].metadata.name == "plugin.a"
    assert plugins[1].metadata.name == "plugin.b"


def test_plugin_group_get_plugins_none_found() -> None:
    """Get plugins returns empty when none found."""
    group = PluginGroup(
        name="test",
        plugins=("plugin.missing",),
    )
    registry = MockPluginRegistry({})

    plugins = group.get_plugins(registry)  # type: ignore[arg-type]

    assert plugins == []


def test_group_registry_register_and_get() -> None:
    """Register and retrieve groups."""
    registry = GroupRegistry()
    group = PluginGroup(name="test", description="Test group")

    registry.register(group)
    retrieved = registry.get("test")

    assert retrieved is not None
    assert retrieved.name == "test"


def test_group_registry_get_missing() -> None:
    """Get returns None for missing groups."""
    registry = GroupRegistry()

    result = registry.get("nonexistent")

    assert result is None


def test_group_registry_resolve_dependencies_no_deps() -> None:
    """Resolve dependencies with no group dependencies."""
    registry = GroupRegistry()
    registry.register(PluginGroup(name="a"))
    registry.register(PluginGroup(name="b"))

    result = registry.resolve_dependencies(["a", "b"])

    assert result == ["a", "b"]


def test_group_registry_resolve_dependencies_with_deps() -> None:
    """Resolve dependencies orders dependencies first."""
    registry = GroupRegistry()
    registry.register(PluginGroup(name="base"))
    registry.register(PluginGroup(name="derived", requires=("base",)))

    result = registry.resolve_dependencies(["derived"])

    assert result == ["base", "derived"]


def test_group_registry_resolve_dependencies_chain() -> None:
    """Resolve dependency chains correctly."""
    registry = GroupRegistry()
    registry.register(PluginGroup(name="level1"))
    registry.register(PluginGroup(name="level2", requires=("level1",)))
    registry.register(PluginGroup(name="level3", requires=("level2",)))

    result = registry.resolve_dependencies(["level3"])

    assert result == ["level1", "level2", "level3"]


def test_group_registry_resolve_dependencies_deduplicates() -> None:
    """Resolve dependencies doesn't repeat groups."""
    registry = GroupRegistry()
    registry.register(PluginGroup(name="shared"))
    registry.register(PluginGroup(name="a", requires=("shared",)))
    registry.register(PluginGroup(name="b", requires=("shared",)))

    result = registry.resolve_dependencies(["a", "b"])

    # shared should appear only once, before a and b
    assert result.count("shared") == 1
    assert result.index("shared") < result.index("a")


def test_group_registry_resolve_dependencies_missing_group() -> None:
    """Resolve dependencies skips missing groups."""
    registry = GroupRegistry()

    result = registry.resolve_dependencies(["nonexistent"])

    assert result == []


def test_group_registry_get_all_plugins() -> None:
    """Get all plugins from multiple groups."""
    plugin_a = MockPlugin("plugin.a")
    plugin_b = MockPlugin("plugin.b")

    plugin_registry = MockPluginRegistry(
        {
            "plugin.a": plugin_a,
            "plugin.b": plugin_b,
        }
    )

    group_registry = GroupRegistry()
    group_registry.register(PluginGroup(name="group1", plugins=("plugin.a",)))
    group_registry.register(PluginGroup(name="group2", plugins=("plugin.b",)))

    plugins = group_registry.get_all_plugins(
        ["group1", "group2"],
        plugin_registry,  # type: ignore[arg-type]
    )

    assert len(plugins) == DEPENDENCY_ORDER_EXPECTED


def test_group_registry_get_all_plugins_deduplicates() -> None:
    """Get all plugins doesn't repeat same plugin."""
    plugin = MockPlugin("shared.plugin")

    plugin_registry = MockPluginRegistry(
        {
            "shared.plugin": plugin,
        }
    )

    group_registry = GroupRegistry()
    group_registry.register(PluginGroup(name="group1", plugins=("shared.plugin",)))
    group_registry.register(PluginGroup(name="group2", plugins=("shared.plugin",)))

    plugins = group_registry.get_all_plugins(
        ["group1", "group2"],
        plugin_registry,  # type: ignore[arg-type]
    )

    # Plugin should appear only once
    assert len(plugins) == 1


def test_group_registry_get_all_plugins_respects_dependency_order() -> None:
    """Get all plugins respects dependency order."""
    plugin_base = MockPlugin("base.plugin")
    plugin_derived = MockPlugin("derived.plugin")

    plugin_registry = MockPluginRegistry(
        {
            "base.plugin": plugin_base,
            "derived.plugin": plugin_derived,
        }
    )

    group_registry = GroupRegistry()
    group_registry.register(PluginGroup(name="base", plugins=("base.plugin",)))
    group_registry.register(
        PluginGroup(
            name="derived",
            plugins=("derived.plugin",),
            requires=("base",),
        )
    )

    plugins = group_registry.get_all_plugins(
        ["derived"],  # Only request derived, should include base
        plugin_registry,  # type: ignore[arg-type]
    )

    assert len(plugins) == DEPENDENCY_ORDER_EXPECTED
    # Base should come first due to dependency ordering
    assert plugins[0].metadata.name == "base.plugin"
    assert plugins[1].metadata.name == "derived.plugin"


def test_function_plugins_group_defined() -> None:
    """FUNCTION_PLUGINS constant is properly defined."""
    assert FUNCTION_PLUGINS.name == "functions"
    assert len(FUNCTION_PLUGINS.plugins) > 0


def test_graph_plugins_group_defined() -> None:
    """GRAPH_PLUGINS constant is properly defined."""
    assert GRAPH_PLUGINS.name == "graphs"
    assert len(GRAPH_PLUGINS.plugins) > 0


def test_risk_plugins_group_defined() -> None:
    """RISK_PLUGINS constant is properly defined."""
    assert RISK_PLUGINS.name == "risk"
    assert "functions" in RISK_PLUGINS.requires
    assert "graphs" in RISK_PLUGINS.requires


def test_all_groups_contains_expected_groups() -> None:
    """ALL_GROUPS contains the expected groups."""
    assert "functions" in ALL_GROUPS
    assert "graphs" in ALL_GROUPS
    assert "risk" in ALL_GROUPS
    assert len(ALL_GROUPS) == EXPECTED_GROUPS_COUNT


def test_plugin_group_with_declaration_order() -> None:
    """PluginGroup can use declaration order."""
    group = PluginGroup(
        name="ordered",
        plugins=("first", "second"),
        default_order="declaration",
    )

    assert group.default_order == "declaration"


def test_plugin_group_disabled() -> None:
    """PluginGroup can be disabled."""
    group = PluginGroup(
        name="disabled",
        enabled=False,
    )

    assert group.enabled is False


def test_group_registry_empty() -> None:
    """GroupRegistry starts empty."""
    registry = GroupRegistry()

    result = registry.get("anything")

    assert result is None


def test_group_registry_overwrite_group() -> None:
    """Registering a group with same name overwrites."""
    registry = GroupRegistry()
    registry.register(PluginGroup(name="test", description="first"))
    registry.register(PluginGroup(name="test", description="second"))

    result = registry.get("test")

    assert result is not None
    assert result.description == "second"
