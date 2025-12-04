"""Graph plugin registry with dependency resolution.

This module provides the registry for graph plugins, extending the base
registry infrastructure from codeintel.core.plugins. It supports
decorator-based registration, dependency resolution, topological ordering,
and discovery via Python entry points.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from collections.abc import Sequence
from uuid import uuid4

from codeintel.core.plugins.registry import BasePluginRegistry
from codeintel.core.singleton import SingletonHolder
from codeintel.graphs.core.protocol import (
    DEFAULT_GRAPH_PLUGINS,
    GraphPluginPlan,
    GraphPluginProtocol,
    GraphPluginSkip,
)

log = logging.getLogger(__name__)


class _GraphPluginRegistryHolder(SingletonHolder["GraphPluginRegistry"]):
    """Singleton holder for GraphPluginRegistry.

    Uses the thread-safe SingletonHolder pattern from core.
    """


class GraphPluginRegistry(BasePluginRegistry[GraphPluginProtocol]):
    """Central registry for graph plugins.

    Extends BasePluginRegistry with graph-specific functionality
    including GraphPluginPlan and GraphPluginSkip types.

    For singleton access, use :func:`get_graph_registry` rather than
    instantiating directly. Direct instantiation is useful for testing.
    """

    @staticmethod
    def _get_default_plugins() -> Sequence[str]:
        """Return the default graph plugin names.

        Returns
        -------
        Sequence[str]
            Default graph plugin names.
        """
        return DEFAULT_GRAPH_PLUGINS

    def _ensure_builtins_loaded(self) -> None:
        """Import built-in graph plugins to guarantee registration."""
        if self._builtins_loaded:
            return
        try:
            importlib.import_module("codeintel.graphs.plugins")
        except ImportError as exc:
            log.warning("Failed to import built-in graph plugins: %s", exc)
        self._builtins_loaded = True

    def _ensure_entrypoints_loaded(self) -> None:
        """Load plugins from entry points if not already done."""
        if self._entrypoints_loaded:
            return
        self.load_from_entrypoints()

    def load_from_entrypoints(
        self,
        *,
        group: str = "codeintel.graph_plugins",
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
        """
        if self._entrypoints_loaded and not force:
            return ()

        discovered: list[GraphPluginProtocol] = []
        eps = importlib.metadata.entry_points()
        selected = eps.select(group=group)

        for entry_point in selected:
            candidate: GraphPluginProtocol | None = None
            try:
                loaded = entry_point.load()
                if isinstance(loaded, GraphPluginProtocol):
                    candidate = loaded
                elif isinstance(loaded, type) or (
                    callable(loaded) and not hasattr(loaded, "metadata")
                ):
                    instance = loaded()
                    if isinstance(instance, GraphPluginProtocol):
                        candidate = instance
            except (ImportError, AttributeError, TypeError) as exc:
                log.warning(
                    "Failed to load graph plugin from entrypoint %s: %s",
                    entry_point.name,
                    exc,
                )
                continue

            if not isinstance(candidate, GraphPluginProtocol):
                log.warning("Entry point %s did not return GraphPluginProtocol", entry_point.name)
                continue

            plugin = candidate

            self.register(plugin)
            discovered.append(plugin)
            log.info("Discovered graph plugin from entrypoint: %s", plugin.metadata.name)

        self._entrypoints_loaded = True
        return tuple(discovered)

    def plan(
        self,
        plugin_names: Sequence[str] | None = None,
        *,
        enabled: Sequence[str] | None = None,
        disabled: Sequence[str] | None = None,
        defaults: Sequence[str] | None = None,
    ) -> GraphPluginPlan:
        """Build an execution plan with dependency resolution.

        Override base implementation to return GraphPluginPlan with
        graph-specific skip reasons. May raise ValueError from helper
        methods if plugins are listed more than once or dependencies
        are missing/cyclic.

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
        GraphPluginPlan
            Ordered execution plan with graph-specific metadata.
        """
        self._ensure_loaded()

        # Resolve which plugins to include
        selected, skipped = self._resolve_graph_selection(
            plugin_names=plugin_names,
            enabled=enabled,
            disabled=disabled,
            defaults=defaults or self._get_default_plugins(),
        )

        # Build dependency graph
        dependencies = self._resolve_graph_dependencies(selected)

        # Topological sort (reuse base class static method)
        ordered = self._topological_sort(selected, dependencies)

        return GraphPluginPlan(
            plugins=tuple(ordered),
            plan_id=uuid4().hex,
            skipped_plugins=skipped,
            dep_graph={name: tuple(sorted(deps)) for name, deps in dependencies.items()},
        )

    def _resolve_graph_selection(
        self,
        *,
        plugin_names: Sequence[str] | None,
        enabled: Sequence[str] | None,
        disabled: Sequence[str] | None,
        defaults: Sequence[str],
    ) -> tuple[dict[str, GraphPluginProtocol], tuple[GraphPluginSkip, ...]]:
        """Resolve which plugins to include in the plan.

        Returns
        -------
        tuple[dict[str, GraphPluginProtocol], tuple[GraphPluginSkip, ...]]
            Selected plugins keyed by name and plugins that were skipped.

        Raises
        ------
        ValueError
            If a plugin name appears more than once.
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
        skipped: list[GraphPluginSkip] = []

        for name in names:
            if name in disabled_set:
                skipped.append(GraphPluginSkip(name=name, reason="disabled"))
                continue

            if name in selected:
                message = f"Graph plugin '{name}' listed more than once"
                raise ValueError(message)

            try:
                plugin = self.get(name)
            except KeyError:
                skipped.append(GraphPluginSkip(name=name, reason="missing_dependency"))
                log.warning("Skipping unknown graph plugin: %s", name)
                continue

            selected[name] = plugin

        return selected, tuple(skipped)

    def _resolve_graph_dependencies(
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
            If dependencies are missing or ambiguous.
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


def get_graph_registry() -> GraphPluginRegistry:
    """Return the global graph plugin registry.

    Returns
    -------
    GraphPluginRegistry
        The singleton registry instance.
    """
    return _GraphPluginRegistryHolder.get(GraphPluginRegistry)


def reset_graph_registry() -> None:
    """Reset the global registry for testing.

    This clears the global registry, allowing tests to start fresh.
    """
    _GraphPluginRegistryHolder.reset()


def register_graph_plugin(plugin: GraphPluginProtocol) -> None:
    """Register a plugin with the global registry.

    Parameters
    ----------
    plugin
        Plugin instance to register.
    """
    get_graph_registry().register(plugin)


def unregister_graph_plugin(name: str) -> None:
    """Remove a plugin from the global registry.

    Parameters
    ----------
    name
        Plugin name to remove.
    """
    get_graph_registry().unregister(name)


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
) -> GraphPluginPlan:
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
    GraphPluginPlan
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
    "reset_graph_registry",
    "unregister_graph_plugin",
]
