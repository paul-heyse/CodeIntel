"""Protocol and implementation for plugin groups.

This module provides `PluginGroup` for bundling related plugins
into logical groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from codeintel.analytics.core.protocol import AnalyticsPluginProtocol
    from codeintel.analytics.core.registry import PluginRegistry


@dataclass(frozen=True)
class PluginGroup:
    """A named group of related plugins.

    Groups allow organizing plugins into logical bundles that can be:
    - Enabled/disabled together
    - Executed in a specific order
    - Depend on other groups

    Attributes
    ----------
    name
        Unique group identifier.
    description
        Human-readable description.
    plugins
        Plugin names in this group.
    default_order
        How to order plugins: "dependency" (topological) or "declaration".
    requires
        Other groups this group depends on.
    tags
        Classification tags.
    enabled
        Whether the group is enabled by default.
    """

    name: str
    description: str = ""
    plugins: tuple[str, ...] = ()
    default_order: Literal["dependency", "declaration"] = "dependency"
    requires: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    enabled: bool = True

    def get_plugins(
        self,
        registry: PluginRegistry,
    ) -> list[AnalyticsPluginProtocol]:
        """Get all plugins in this group from a registry.

        Parameters
        ----------
        registry
            Plugin registry to search.

        Returns
        -------
        list[AnalyticsPluginProtocol]
            Plugins found in the registry.
        """
        found: list[AnalyticsPluginProtocol] = []
        for name in self.plugins:
            try:
                plugin = registry.get(name)
                found.append(plugin)
            except KeyError:
                # Plugin not registered, skip it
                pass
        return found

    def get_plugin_names(self) -> tuple[str, ...]:
        """Return the names of plugins in this group.

        Returns
        -------
        tuple[str, ...]
            Plugin names.
        """
        return self.plugins

    def with_plugins(self, *plugins: str) -> PluginGroup:
        """Return a new group with additional plugins.

        Parameters
        ----------
        *plugins
            Plugin names to add.

        Returns
        -------
        PluginGroup
            New group with additional plugins.
        """
        return PluginGroup(
            name=self.name,
            description=self.description,
            plugins=self.plugins + plugins,
            default_order=self.default_order,
            requires=self.requires,
            tags=self.tags,
            enabled=self.enabled,
        )

    def without_plugins(self, *plugins: str) -> PluginGroup:
        """Return a new group with plugins removed.

        Parameters
        ----------
        *plugins
            Plugin names to remove.

        Returns
        -------
        PluginGroup
            New group without specified plugins.
        """
        to_remove = set(plugins)
        return PluginGroup(
            name=self.name,
            description=self.description,
            plugins=tuple(p for p in self.plugins if p not in to_remove),
            default_order=self.default_order,
            requires=self.requires,
            tags=self.tags,
            enabled=self.enabled,
        )


@dataclass
class GroupRegistry:
    """Registry for plugin groups.

    Provides lookup and dependency resolution for groups.
    """

    _groups: dict[str, PluginGroup] = field(default_factory=dict)

    def register(self, group: PluginGroup) -> None:
        """Register a plugin group.

        Parameters
        ----------
        group
            Group to register.
        """
        self._groups[group.name] = group

    def get(self, name: str) -> PluginGroup | None:
        """Get a group by name.

        Parameters
        ----------
        name
            Group name.

        Returns
        -------
        PluginGroup | None
            The group, or None if not found.
        """
        return self._groups.get(name)

    def resolve_dependencies(self, group_names: list[str]) -> list[str]:
        """Resolve group dependencies to get execution order.

        Parameters
        ----------
        group_names
            Groups to resolve.

        Returns
        -------
        list[str]
            Groups in dependency order (dependencies first).
        """
        result: list[str] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)

            group = self._groups.get(name)
            if group is not None:
                for dep in group.requires:
                    visit(dep)
                result.append(name)

        for name in group_names:
            visit(name)

        return result

    def get_all_plugins(
        self,
        group_names: list[str],
        registry: PluginRegistry,
    ) -> list[AnalyticsPluginProtocol]:
        """Get all plugins from multiple groups.

        Parameters
        ----------
        group_names
            Groups to get plugins from.
        registry
            Plugin registry.

        Returns
        -------
        list[AnalyticsPluginProtocol]
            Plugins from all groups (deduplicated).
        """
        seen: set[str] = set()
        plugins: list[AnalyticsPluginProtocol] = []

        ordered_groups = self.resolve_dependencies(group_names)

        for group_name in ordered_groups:
            group = self._groups.get(group_name)
            if group is None:
                continue

            for plugin in group.get_plugins(registry):
                if plugin.metadata.name not in seen:
                    seen.add(plugin.metadata.name)
                    plugins.append(plugin)

        return plugins


__all__ = [
    "GroupRegistry",
    "PluginGroup",
]
