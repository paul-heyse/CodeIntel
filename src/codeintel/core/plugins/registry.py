"""Unified plugin registry infrastructure.

This module provides base classes and utilities for plugin registries,
encapsulating common logic like dependency resolution, topological sorting,
and entry point discovery.

Both graphs and analytics registries extend these base classes to provide
domain-specific functionality while reusing the core logic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar, runtime_checkable
from uuid import uuid4

from codeintel.core.plugins.protocol import PluginMetadata

log = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Plugin Protocol (minimal for registry purposes)
# =============================================================================


@runtime_checkable
class RegistrablePlugin(Protocol):
    """Minimal protocol for plugins that can be registered.

    This protocol defines the minimum interface needed for a plugin to
    participate in registry operations (registration, planning).
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        PluginMetadata
            Metadata describing the plugin.
        """
        ...


# =============================================================================
# Plan and Skip Dataclasses
# =============================================================================


@dataclass(frozen=True)
class PluginSkip:
    """Skip metadata for plugins excluded from execution.

    Attributes
    ----------
    name
        Plugin name.
    reason
        Reason for skipping (e.g., disabled, missing_dependency, config_error).
    """

    name: str
    reason: str


@dataclass(frozen=True)
class PluginPlan[P: RegistrablePlugin]:
    """Resolved execution plan for a set of plugins.

    Type Parameters
    ---------------
    P
        The plugin protocol type this plan holds (must implement RegistrablePlugin).

    Attributes
    ----------
    plugins
        Ordered tuple of plugins to execute.
    plan_id
        Unique identifier for this plan.
    skipped
        Plugins that were excluded from the plan.
    dep_graph
        Dependency graph (plugin name -> dependencies).
    """

    plugins: tuple[P, ...]
    plan_id: str = field(default_factory=lambda: uuid4().hex)
    skipped: tuple[PluginSkip, ...] = ()
    dep_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def ordered_names(self) -> tuple[str, ...]:
        """Return plugin names in execution order.

        Returns
        -------
        tuple[str, ...]
            Plugin names in execution order.
        """
        return tuple(p.metadata.name for p in self.plugins)


# =============================================================================
# Base Registry Class
# =============================================================================


class BasePluginRegistry[P: RegistrablePlugin](ABC):
    """Abstract base class for plugin registries.

    Provides common logic for plugin registration, lookup, dependency
    resolution, and topological ordering. Subclasses implement domain-specific
    methods and entry point loading.

    Type Parameters
    ---------------
    P
        The plugin protocol type this registry manages.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._plugins: dict[str, P] = {}
        self._by_capability: dict[str, set[str]] = {}
        self._by_stage: dict[str, set[str]] = {}
        self._by_kind: dict[str, set[str]] = {}
        self._by_table: dict[str, set[str]] = {}
        self._entrypoints_loaded: bool = False
        self._builtins_loaded: bool = False

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register(self, plugin: P) -> None:
        """Register a plugin instance.

        Parameters
        ----------
        plugin
            Plugin instance implementing the plugin protocol.

        Raises
        ------
        ValueError
            If a plugin with the same name is already registered.
        """
        meta = plugin.metadata
        if meta.name in self._plugins:
            message = f"Duplicate plugin name: {meta.name}"
            raise ValueError(message)

        self._plugins[meta.name] = plugin
        self._index_plugin(meta)
        log.debug("Registered plugin %s (kind=%s, stage=%s)", meta.name, meta.kind, meta.stage)

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
        self._unindex_plugin(plugin.metadata)

    def _index_plugin(self, meta: PluginMetadata) -> None:
        """Index plugin by capabilities, stage, kind, and tables.

        Parameters
        ----------
        meta
            Plugin metadata to index.
        """
        for cap in meta.provides:
            self._by_capability.setdefault(cap, set()).add(meta.name)
        self._by_stage.setdefault(meta.stage, set()).add(meta.name)
        self._by_kind.setdefault(meta.kind, set()).add(meta.name)
        for table in meta.produces_tables:
            self._by_table.setdefault(table, set()).add(meta.name)

    def _unindex_plugin(self, meta: PluginMetadata) -> None:
        """Remove plugin from indices.

        Parameters
        ----------
        meta
            Plugin metadata to unindex.
        """
        for cap in meta.provides:
            if cap in self._by_capability:
                self._by_capability[cap].discard(meta.name)
        if meta.stage in self._by_stage:
            self._by_stage[meta.stage].discard(meta.name)
        if meta.kind in self._by_kind:
            self._by_kind[meta.kind].discard(meta.name)
        for table in meta.produces_tables:
            if table in self._by_table:
                self._by_table[table].discard(meta.name)

    # -------------------------------------------------------------------------
    # Lookup
    # -------------------------------------------------------------------------

    def get(self, name: str) -> P:
        """Return a plugin by name.

        Parameters
        ----------
        name
            Plugin name to look up.

        Returns
        -------
        P
            The registered plugin.

        Raises
        ------
        KeyError
            If no plugin is registered with the given name.
        """
        self._ensure_loaded()
        if name not in self._plugins:
            message = f"Unknown plugin: {name}"
            raise KeyError(message)
        return self._plugins[name]

    def contains(self, name: str) -> bool:
        """Check if a plugin is registered.

        Parameters
        ----------
        name
            Plugin name to check.

        Returns
        -------
        bool
            True if registered.
        """
        self._ensure_loaded()
        return name in self._plugins

    def list_all(self) -> tuple[P, ...]:
        """Return all registered plugins.

        Returns
        -------
        tuple[P, ...]
            All registered plugins in registration order.
        """
        self._ensure_loaded()
        return tuple(self._plugins.values())

    def list_names(self) -> tuple[str, ...]:
        """Return names of all registered plugins.

        Returns
        -------
        tuple[str, ...]
            Plugin names in registration order.
        """
        self._ensure_loaded()
        return tuple(self._plugins.keys())

    def list_by_stage(self, stage: str) -> tuple[P, ...]:
        """Return plugins belonging to a specific stage.

        Parameters
        ----------
        stage
            Stage name to filter by.

        Returns
        -------
        tuple[P, ...]
            Plugins in the specified stage.
        """
        self._ensure_loaded()
        names = self._by_stage.get(stage, set())
        return tuple(self._plugins[name] for name in names if name in self._plugins)

    def list_by_kind(self, kind: str) -> tuple[P, ...]:
        """Return plugins of a specific kind.

        Parameters
        ----------
        kind
            Plugin kind to filter by.

        Returns
        -------
        tuple[P, ...]
            Plugins of the specified kind.
        """
        self._ensure_loaded()
        names = self._by_kind.get(kind, set())
        return tuple(self._plugins[name] for name in names if name in self._plugins)

    def list_providing(self, capability: str) -> tuple[P, ...]:
        """Return plugins that provide a specific capability.

        Parameters
        ----------
        capability
            Capability name to search for.

        Returns
        -------
        tuple[P, ...]
            Plugins providing the capability.
        """
        self._ensure_loaded()
        names = self._by_capability.get(capability, set())
        return tuple(self._plugins[name] for name in names if name in self._plugins)

    def list_by_table(self, table_key: str) -> tuple[P, ...]:
        """Return plugins that produce a specific table.

        Parameters
        ----------
        table_key
            Table key (e.g., "graph.call_graph_nodes").

        Returns
        -------
        tuple[P, ...]
            Plugins producing the table.
        """
        self._ensure_loaded()
        names = self._by_table.get(table_key, set())
        return tuple(self._plugins[name] for name in names if name in self._plugins)

    def metadata_for(self, name: str) -> PluginMetadata:
        """Return metadata for a plugin.

        Parameters
        ----------
        name
            Plugin name.

        Returns
        -------
        PluginMetadata
            Plugin metadata.
        """
        return self.get(name).metadata

    def dependency_graph(self) -> dict[str, tuple[str, ...]]:
        """Return mapping of plugin name to direct dependencies.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Direct dependency map keyed by plugin name.
        """
        self._ensure_loaded()
        return {name: plugin.metadata.depends_on for name, plugin in self._plugins.items()}

    # -------------------------------------------------------------------------
    # Planning Utilities
    # -------------------------------------------------------------------------
    # Subclasses implement their own plan() method using these utilities.
    # This avoids return type incompatibility issues with domain-specific
    # plan types (e.g., GraphPluginPlan vs PluginPlan).

    @staticmethod
    @abstractmethod
    def _get_default_plugins() -> Sequence[str]:
        """Return the default plugin names for this registry.

        Returns
        -------
        Sequence[str]
            Default plugin names.
        """
        ...

    def _resolve_selection(
        self,
        *,
        plugin_names: Sequence[str] | None,
        enabled: Sequence[str] | None,
        disabled: Sequence[str] | None,
        defaults: Sequence[str],
    ) -> tuple[dict[str, P], tuple[PluginSkip, ...]]:
        """Resolve which plugins to include in the plan.

        Returns
        -------
        tuple[dict[str, P], tuple[PluginSkip, ...]]
            Selected plugins and skipped plugins.
        """
        # Determine base selection
        if enabled:
            names = list(enabled)
        elif plugin_names:
            names = list(plugin_names)
        else:
            names = list(defaults)

        disabled_set = set(disabled or ())
        selected: dict[str, P] = {}
        skipped: list[PluginSkip] = []

        for name in names:
            if name in disabled_set:
                skipped.append(PluginSkip(name=name, reason="disabled"))
                continue
            if name in selected:
                continue
            try:
                plugin = self.get(name)
            except KeyError:
                skipped.append(PluginSkip(name=name, reason="missing_dependency"))
                log.warning("Skipping unknown plugin: %s", name)
                continue
            selected[name] = plugin

        return selected, tuple(skipped)

    def _resolve_dependencies(
        self,
        selected: dict[str, P],
    ) -> dict[str, set[str]]:
        """Build dependency graph for selected plugins.

        Returns
        -------
        dict[str, set[str]]
            Mapping of plugin name to its dependency names.
        """
        dependencies: dict[str, set[str]] = {name: set() for name in selected}

        # Explicit depends_on
        for name, plugin in selected.items():
            for dep in plugin.metadata.depends_on:
                if dep in selected:
                    dependencies[name].add(dep)
                else:
                    log.debug("Skipping unmet dependency %s for plugin %s", dep, name)

        # Capability-based dependencies
        provider_index = self._build_provider_index(selected)
        for name, plugin in selected.items():
            for cap_name in plugin.metadata.requires:
                providers = provider_index.get(cap_name, set())
                if not providers:
                    log.warning(
                        "Plugin %s requires capability %s but no provider is selected",
                        name,
                        cap_name,
                    )
                    continue
                # Add first provider as dependency (unless self-provided)
                for provider in providers:
                    if provider != name:
                        dependencies[name].add(provider)
                        break

        return dependencies

    @staticmethod
    def _build_provider_index(
        selected: Mapping[str, RegistrablePlugin],
    ) -> dict[str, set[str]]:
        """Build index of capability -> provider plugins.

        Parameters
        ----------
        selected
            Selected plugins to index.

        Returns
        -------
        dict[str, set[str]]
            Mapping of capability name to provider plugin names.
        """
        index: dict[str, set[str]] = {}
        for name, plugin in selected.items():
            for cap_name in plugin.metadata.provides:
                index.setdefault(cap_name, set()).add(name)
        return index

    @staticmethod
    def _topological_sort[Q](
        selected: dict[str, Q],
        dependencies: dict[str, set[str]],
    ) -> list[Q]:
        """Perform topological sort with cycle detection.

        Use iterative DFS with explicit stack to detect cycles and order plugins.

        Parameters
        ----------
        selected
            Selected plugins keyed by name.
        dependencies
            Dependency graph.

        Returns
        -------
        list[Q]
            Plugins ordered based on dependencies.

        Raises
        ------
        ValueError
            If a dependency cycle is detected.
        """
        ordered: list[Q] = []
        permanent: set[str] = set()
        temporary: set[str] = set()

        # Iterative DFS with explicit stack
        for start in selected:
            if start in permanent:
                continue

            # Stack entries: (name, deps_list, is_processing)
            stack: list[tuple[str, list[str], bool]] = [
                (start, list(dependencies.get(start, set())), False)
            ]

            while stack:
                name, deps, processing = stack.pop()

                if processing:
                    # Finished processing all deps, mark permanent
                    temporary.discard(name)
                    permanent.add(name)
                    ordered.append(selected[name])
                    continue

                if name in permanent:
                    continue

                if name in temporary:
                    # Cycle detected - raise at function level
                    message = f"Dependency cycle detected involving plugin: {name}"
                    raise ValueError(message)

                temporary.add(name)
                # Push back with processing=True to finalize after deps
                stack.append((name, [], True))

                # Push dependencies to process (filter for unvisited)
                unvisited_deps = [
                    (dep, list(dependencies.get(dep, set())), False)
                    for dep in deps
                    if dep not in permanent
                ]
                stack.extend(unvisited_deps)

        return ordered

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Ensure plugins are loaded (builtins + entry points)."""
        self._ensure_builtins_loaded()
        self._ensure_entrypoints_loaded()

    @abstractmethod
    def _ensure_builtins_loaded(self) -> None:
        """Load built-in plugins if not already done."""
        ...

    @abstractmethod
    def _ensure_entrypoints_loaded(self) -> None:
        """Load plugins from entry points if not already done."""
        ...


__all__ = [
    "BasePluginRegistry",
    "PluginPlan",
    "PluginSkip",
    "RegistrablePlugin",
]
