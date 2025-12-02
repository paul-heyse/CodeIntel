"""Modernized graph plugin registry.

This module provides the registry for graph plugins, supporting
decorator-based registration, dependency resolution, and
topological ordering.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Sequence
from uuid import uuid4

from codeintel.analytics.graphs.core.protocol import (
    DEFAULT_GRAPH_METRIC_PLUGINS,
    GraphMetricPluginPlan,
    GraphMetricPluginSkip,
    GraphPluginProtocol,
)

log = logging.getLogger(__name__)


class GraphPluginRegistry:
    """Central registry for graph metric plugins.

    Provides plugin registration, lookup, dependency resolution,
    and topological ordering.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._plugins: dict[str, GraphPluginProtocol] = {}
        self._by_capability: dict[str, set[str]] = {}
        self._entrypoints_loaded: bool = False

    def register(self, plugin: GraphPluginProtocol) -> None:
        """Register a graph plugin.

        Parameters
        ----------
        plugin
            Plugin instance implementing GraphPluginProtocol.

        Raises
        ------
        ValueError
            If a plugin with the same name is already registered.
        """
        meta = plugin.metadata
        if meta.name in self._plugins:
            message = f"Duplicate graph plugin name: {meta.name}"
            raise ValueError(message)

        self._plugins[meta.name] = plugin

        # Index by capabilities
        for cap in meta.provides:
            self._by_capability.setdefault(cap, set()).add(meta.name)

        log.debug("Registered graph plugin %s (stage=%s)", meta.name, meta.stage)

    def unregister(self, name: str) -> None:
        """Remove a plugin from the registry.

        Parameters
        ----------
        name
            Plugin name to remove.
        """
        plugin = self._plugins.pop(name, None)
        if plugin is None:
            return

        meta = plugin.metadata
        for cap in meta.provides:
            if cap in self._by_capability:
                self._by_capability[cap].discard(name)

    def get(self, name: str) -> GraphPluginProtocol:
        """Return a plugin by name.

        Parameters
        ----------
        name
            Plugin name to look up.

        Returns
        -------
        GraphPluginProtocol
            The registered plugin.

        Raises
        ------
        KeyError
            If no plugin is registered with the given name.
        """
        self._ensure_entrypoints_loaded()
        if name not in self._plugins:
            message = f"Unknown graph plugin: {name}"
            raise KeyError(message)
        return self._plugins[name]

    def list_all(self) -> tuple[GraphPluginProtocol, ...]:
        """Return all registered plugins.

        Returns
        -------
        tuple[GraphPluginProtocol, ...]
            All registered plugins in registration order.
        """
        self._ensure_entrypoints_loaded()
        return tuple(self._plugins.values())

    def list_providing(self, capability: str) -> tuple[GraphPluginProtocol, ...]:
        """Return plugins that provide a specific capability.

        Parameters
        ----------
        capability
            Capability name to search for.

        Returns
        -------
        tuple[GraphPluginProtocol, ...]
            Plugins providing the capability.
        """
        self._ensure_entrypoints_loaded()
        names = self._by_capability.get(capability, set())
        return tuple(self._plugins[name] for name in names if name in self._plugins)

    def plan(
        self,
        plugin_names: Sequence[str] | None = None,
        *,
        enabled: Sequence[str] | None = None,
        disabled: Sequence[str] | None = None,
        defaults: Sequence[str] | None = None,
    ) -> GraphMetricPluginPlan:
        """Build an execution plan with dependency resolution.

        Parameters
        ----------
        plugin_names
            Explicit plugin names to include.
        enabled
            Override list of enabled plugins.
        disabled
            Plugins to exclude from the plan.
        defaults
            Default plugins if no explicit list provided.

        Returns
        -------
        GraphMetricPluginPlan
            Ordered execution plan.
        """
        self._ensure_entrypoints_loaded()

        # Resolve which plugins to include
        selected, skipped = self._resolve_selection(
            plugin_names=plugin_names,
            enabled=enabled,
            disabled=disabled,
            defaults=defaults or DEFAULT_GRAPH_METRIC_PLUGINS,
        )

        # Build dependency graph
        dependencies = self._resolve_dependencies(selected)

        # Topological sort
        ordered = self._topological_sort(selected, dependencies)

        return GraphMetricPluginPlan(
            plugins=tuple(ordered),
            plan_id=uuid4().hex,
            skipped_plugins=skipped,
            dep_graph={name: tuple(sorted(deps)) for name, deps in dependencies.items()},
        )

    def load_from_entrypoints(
        self,
        *,
        group: str = "codeintel.graph_metric_plugins",
        force: bool = False,
    ) -> tuple[GraphPluginProtocol, ...]:
        """Discover and register plugins from entry points.

        Parameters
        ----------
        group
            Entry point group to load from.
        force
            Whether to reload even if already loaded.

        Returns
        -------
        tuple[GraphPluginProtocol, ...]
            Newly loaded plugins.

        Raises
        ------
        TypeError
            If an entry point does not return a valid plugin.
        """
        if self._entrypoints_loaded and not force:
            return ()

        discovered: list[GraphPluginProtocol] = []
        for entry_point in importlib.metadata.entry_points().select(group=group):
            plugin = entry_point.load()
            if not isinstance(plugin, GraphPluginProtocol):
                message = f"Entry point {entry_point.name} did not return GraphPluginProtocol"
                raise TypeError(message)
            self.register(plugin)
            discovered.append(plugin)
            log.info("Discovered graph plugin from entrypoint: %s", plugin.metadata.name)

        self._entrypoints_loaded = True
        return tuple(discovered)

    def _ensure_entrypoints_loaded(self) -> None:
        """Load entry points if not already done."""
        if not self._entrypoints_loaded:
            self.load_from_entrypoints()

    def _resolve_selection(
        self,
        *,
        plugin_names: Sequence[str] | None,
        enabled: Sequence[str] | None,
        disabled: Sequence[str] | None,
        defaults: Sequence[str],
    ) -> tuple[dict[str, GraphPluginProtocol], tuple[GraphMetricPluginSkip, ...]]:
        """Resolve which plugins to include in the plan.

        Returns
        -------
        tuple[dict[str, GraphPluginProtocol], tuple[GraphMetricPluginSkip, ...]]
            Selected plugins keyed by name and plugins that were skipped.

        Raises
        ------
        ValueError
            If a plugin is listed more than once.
        """
        # Determine base selection
        if enabled:
            names = list(enabled)
        elif plugin_names:
            names = list(plugin_names)
        else:
            names = list(defaults)

        disabled_set = set(disabled or ())
        selected: dict[str, GraphPluginProtocol] = {}
        skipped: list[GraphMetricPluginSkip] = []

        for name in names:
            if name in disabled_set:
                skipped.append(GraphMetricPluginSkip(name=name, reason="disabled"))
                continue
            if name in selected:
                message = f"Graph plugin '{name}' listed more than once"
                raise ValueError(message)
            selected[name] = self.get(name)

        return selected, tuple(skipped)

    def _resolve_dependencies(
        self,
        selected: dict[str, GraphPluginProtocol],
    ) -> dict[str, set[str]]:
        """Build dependency graph for selected plugins.

        Returns
        -------
        dict[str, set[str]]
            Mapping of plugin name to its dependency names.

        Raises
        ------
        ValueError
            If a dependency or capability is missing.
        """
        dependencies: dict[str, set[str]] = {name: set() for name in selected}

        # Explicit depends_on
        for name, plugin in selected.items():
            for dep in plugin.metadata.depends_on:
                if dep not in selected:
                    message = (
                        f"Graph plugin '{name}' depends on '{dep}', "
                        "which is not in the selected plugin set"
                    )
                    raise ValueError(message)
                dependencies[name].add(dep)

        # Capability-based dependencies
        provider_index = self._build_provider_index(selected)
        for name, plugin in selected.items():
            for requirement in plugin.metadata.requires:
                providers = provider_index.get(requirement, set())
                if not providers:
                    message = (
                        f"Graph plugin '{name}' requires capability '{requirement}', "
                        "but no provider plugin is selected"
                    )
                    raise ValueError(message)
                if name in providers:
                    continue
                # Check if already in explicit deps
                explicit_deps = dependencies[name]
                if providers.intersection(explicit_deps):
                    continue
                if len(providers) > 1:
                    provider_list = ", ".join(sorted(providers))
                    message = (
                        f"Graph plugin '{name}' requires capability '{requirement}', "
                        f"but multiple providers are available ({provider_list}). "
                        "Add an explicit depends_on entry to disambiguate."
                    )
                    raise ValueError(message)
                dependencies[name].add(next(iter(providers)))

        return dependencies

    @staticmethod
    def _build_provider_index(
        selected: dict[str, GraphPluginProtocol],
    ) -> dict[str, set[str]]:
        """Build index of capability -> provider plugins.

        Returns
        -------
        dict[str, set[str]]
            Mapping of capability name to provider plugin names.
        """
        index: dict[str, set[str]] = {}
        for name, plugin in selected.items():
            for cap in plugin.metadata.provides:
                index.setdefault(cap, set()).add(name)
        return index

    @staticmethod
    def _topological_sort(
        selected: dict[str, GraphPluginProtocol],
        dependencies: dict[str, set[str]],
    ) -> list[GraphPluginProtocol]:
        """Perform topological sort with cycle detection.

        Returns
        -------
        list[GraphPluginProtocol]
            Plugins ordered based on dependencies.
        """
        ordered: list[GraphPluginProtocol] = []
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(name: str) -> None:
            if name in permanent:
                return
            if name in temporary:
                message = f"Dependency cycle detected involving graph plugin: {name}"
                raise ValueError(message)
            temporary.add(name)
            for dep in dependencies.get(name, set()):
                visit(dep)
            temporary.remove(name)
            permanent.add(name)
            ordered.append(selected[name])

        for name in selected:
            visit(name)

        return ordered


# Global registry instance
_GRAPH_REGISTRY: GraphPluginRegistry | None = None


def get_graph_registry() -> GraphPluginRegistry:
    """Return the global graph plugin registry.

    Returns
    -------
    GraphPluginRegistry
        The singleton registry instance.
    """
    global _GRAPH_REGISTRY  # noqa: PLW0603
    if _GRAPH_REGISTRY is None:
        _GRAPH_REGISTRY = GraphPluginRegistry()
    return _GRAPH_REGISTRY


def register_graph_plugin(plugin: object) -> None:
    """Register a plugin with the global registry.

    Parameters
    ----------
    plugin
        Plugin instance to register.
    """
    get_graph_registry().register(plugin)  # type: ignore[arg-type]


def list_graph_plugins() -> tuple[GraphPluginProtocol, ...]:
    """Return all registered graph plugins.

    Returns
    -------
    tuple[GraphPluginProtocol, ...]
        All registered graph plugins.
    """
    return get_graph_registry().list_all()


def plan_graph_plugins(
    plugin_names: Sequence[str] | None = None,
    *,
    enabled: Sequence[str] | None = None,
    disabled: Sequence[str] | None = None,
    defaults: Sequence[str] | None = None,
) -> GraphMetricPluginPlan:
    """Build an execution plan for graph plugins.

    Parameters
    ----------
    plugin_names
        Explicit plugin names to include.
    enabled
        Override list of enabled plugins.
    disabled
        Plugins to exclude.
    defaults
        Default plugins if no explicit list provided.

    Returns
    -------
    GraphMetricPluginPlan
        Ordered execution plan.
    """
    return get_graph_registry().plan(
        plugin_names=plugin_names,
        enabled=enabled,
        disabled=disabled,
        defaults=defaults,
    )


__all__ = [
    "GraphPluginRegistry",
    "get_graph_registry",
    "list_graph_plugins",
    "plan_graph_plugins",
    "register_graph_plugin",
]
