"""Unified plugin registry infrastructure.

This module provides base classes and utilities for plugin registries,
encapsulating common logic like dependency resolution, topological sorting,
and entry point discovery.

Both graphs and analytics registries extend these base classes to provide
domain-specific functionality while reusing the core logic.
"""

from __future__ import annotations

import importlib.metadata
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeGuard, TypeVar, runtime_checkable

from codeintel.core.execution.ids import new_run_id
from codeintel.core.plugins.registry.sorting import (
    build_provider_index_from_metadata,
    topological_sort,
)
from codeintel.core.plugins.types.protocol import PluginMetadata

log = logging.getLogger(__name__)

P = TypeVar("P", bound="RegistrablePlugin")
P_co = TypeVar("P_co", bound="RegistrablePlugin", covariant=True)


# =============================================================================
# Hooks Protocol
# =============================================================================


class RegistryHooks(Protocol[P_co]):
    """Hook contract for registry-specific behaviors."""

    @property
    def entrypoint_group(self) -> str:
        """Entrypoint group name for discovery."""
        ...

    def load_builtins(self) -> None:
        """Load built-in plugins into the registry context."""
        ...

    def resolve_entrypoint(self, loaded: object) -> P_co | None:
        """Convert a loaded entrypoint object into a plugin instance."""
        ...

    def is_valid_plugin(self, obj: object) -> TypeGuard[P_co]:
        """Validate a plugin instance."""
        ...


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
# Default Hooks
# =============================================================================


class DefaultRegistryHooks(RegistryHooks[P]):
    """Default hooks implementation for generic plugin registries."""

    def __init__(self, entrypoint_group: str = "codeintel.plugins") -> None:
        """Initialize default hooks."""
        self._entrypoint_group = entrypoint_group

    @property
    def entrypoint_group(self) -> str:
        """Return default entry point group name."""
        return self._entrypoint_group

    def load_builtins(self) -> None:
        """Perform no-op builtins load for generic registries."""
        _ = self._entrypoint_group

    def resolve_entrypoint(self, loaded: object) -> P | None:
        """Return the loaded object if it is a valid plugin.

        Returns
        -------
        P | None
            Validated plugin instance or None if invalid.
        """
        if self.is_valid_plugin(loaded):
            return loaded
        return None

    def is_valid_plugin(self, obj: object) -> TypeGuard[P]:
        """Validate plugin using the RegistrablePlugin protocol.

        Returns
        -------
        TypeGuard[P]
            True when the object satisfies the plugin protocol.
        """
        _ = self._entrypoint_group
        return isinstance(obj, RegistrablePlugin)


# =============================================================================
# Registry Entries and Provider Index
# =============================================================================


@dataclass(frozen=True)
class RegistryEntry[P: RegistrablePlugin]:
    """Typed registry entry capturing plugin and metadata."""

    name: str
    plugin: P
    metadata: PluginMetadata


class _ProviderRegistry[P: RegistrablePlugin]:
    """Internal provider registry with indexing."""

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry[P]] = {}
        self._by_capability: dict[str, set[str]] = {}
        self._by_stage: dict[str, set[str]] = {}
        self._by_kind: dict[str, set[str]] = {}
        self._by_table: dict[str, set[str]] = {}

    def register(self, plugin: P) -> RegistryEntry[P]:
        """Register plugin and index metadata.

        Parameters
        ----------
        plugin
            Plugin to register.

        Returns
        -------
        RegistryEntry[P]
            Created registry entry.

        Raises
        ------
        ValueError
            If the plugin name is already registered.
        """
        meta = plugin.metadata
        if meta.name in self._entries:
            message = f"Duplicate plugin name: {meta.name}"
            raise ValueError(message)
        entry = RegistryEntry(name=meta.name, plugin=plugin, metadata=meta)
        self._entries[meta.name] = entry
        self._index(entry)
        return entry

    def unregister(self, name: str) -> RegistryEntry[P] | None:
        """Unregister plugin and remove indexes.

        Parameters
        ----------
        name
            Plugin name to remove.

        Returns
        -------
        RegistryEntry[P] | None
            Removed entry, or None if not present.
        """
        entry = self._entries.pop(name, None)
        if entry is None:
            return None
        self._unindex(entry)
        return entry

    def contains(self, name: str) -> bool:
        """Return True if plugin is registered.

        Returns
        -------
        bool
            True when a plugin with the given name is present.
        """
        return name in self._entries

    def get_entry(self, name: str) -> RegistryEntry[P]:
        """Get registry entry by name.

        Parameters
        ----------
        name
            Plugin name.

        Returns
        -------
        RegistryEntry[P]
            Registry entry containing plugin and metadata.

        Raises
        ------
        KeyError
            If the plugin is not registered.
        """
        if name not in self._entries:
            message = f"Unknown plugin: {name}"
            raise KeyError(message)
        return self._entries[name]

    def list_plugins(self) -> tuple[P, ...]:
        """List all registered plugins.

        Returns
        -------
        tuple[P, ...]
            Plugins in registration order.
        """
        return tuple(entry.plugin for entry in self._entries.values())

    def list_names(self) -> tuple[str, ...]:
        """List registered plugin names.

        Returns
        -------
        tuple[str, ...]
            Plugin names in registration order.
        """
        return tuple(self._entries.keys())

    def list_by_stage(self, stage: str) -> tuple[P, ...]:
        """List plugins for a stage.

        Returns
        -------
        tuple[P, ...]
            Plugins that belong to the stage.
        """
        names = self._by_stage.get(stage, set())
        return tuple(self._entries[name].plugin for name in names if name in self._entries)

    def list_by_kind(self, kind: str) -> tuple[P, ...]:
        """List plugins for a kind.

        Returns
        -------
        tuple[P, ...]
            Plugins of the requested kind.
        """
        names = self._by_kind.get(kind, set())
        return tuple(self._entries[name].plugin for name in names if name in self._entries)

    def list_providing(self, capability: str) -> tuple[P, ...]:
        """List plugins providing a capability.

        Returns
        -------
        tuple[P, ...]
            Plugins that declare the capability.
        """
        names = self._by_capability.get(capability, set())
        return tuple(self._entries[name].plugin for name in names if name in self._entries)

    def list_by_table(self, table_key: str) -> tuple[P, ...]:
        """List plugins producing a table.

        Returns
        -------
        tuple[P, ...]
            Plugins that produce the table key.
        """
        names = self._by_table.get(table_key, set())
        return tuple(self._entries[name].plugin for name in names if name in self._entries)

    def dependency_graph(self) -> dict[str, tuple[str, ...]]:
        """Dependency graph keyed by plugin name.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Map of plugin name to its direct dependencies.
        """
        return {name: entry.metadata.depends_on for name, entry in self._entries.items()}

    def metadata_for(self, name: str) -> PluginMetadata:
        """Metadata for a registered plugin.

        Returns
        -------
        PluginMetadata
            Metadata associated with the plugin.
        """
        return self.get_entry(name).metadata

    def _index(self, entry: RegistryEntry[P]) -> None:
        meta = entry.metadata
        for cap in meta.provides:
            self._by_capability.setdefault(cap, set()).add(meta.name)
        self._by_stage.setdefault(meta.stage, set()).add(meta.name)
        self._by_kind.setdefault(meta.kind, set()).add(meta.name)
        for table in meta.produces_tables:
            self._by_table.setdefault(table, set()).add(meta.name)

    def _unindex(self, entry: RegistryEntry[P]) -> None:
        meta = entry.metadata
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
    plan_id: str = field(default_factory=lambda: new_run_id("plan"))
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

    def __init__(self, hooks: RegistryHooks[P] | None = None) -> None:
        """Initialize an empty registry."""
        self._hooks: RegistryHooks[P] = hooks or DefaultRegistryHooks[P]()
        self._providers: _ProviderRegistry[P] = _ProviderRegistry()
        self._entrypoints_loaded: bool = False
        self._builtins_loaded: bool = False

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register(self, plugin: P) -> RegistryEntry[P]:
        """Register a plugin instance.

        Returns
        -------
        RegistryEntry[P]
            Registry entry created for the plugin.
        """
        entry = self._providers.register(plugin)
        log.debug(
            "Registered plugin %s (kind=%s, stage=%s)",
            entry.name,
            entry.metadata.kind,
            entry.metadata.stage,
        )
        return entry

    def unregister(self, name: str) -> None:
        """Remove a plugin from the registry.

        Parameters
        ----------
        name
            Plugin name to remove.
        """
        self._providers.unregister(name)

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
        try:
            return self._providers.get_entry(name).plugin
        except KeyError as exc:
            raise KeyError(str(exc)) from exc

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
        return self._providers.contains(name)

    def list_all(self) -> tuple[P, ...]:
        """Return all registered plugins.

        Returns
        -------
        tuple[P, ...]
            All registered plugins in registration order.
        """
        self._ensure_loaded()
        return self._providers.list_plugins()

    def list_names(self) -> tuple[str, ...]:
        """Return names of all registered plugins.

        Returns
        -------
        tuple[str, ...]
            Plugin names in registration order.
        """
        self._ensure_loaded()
        return self._providers.list_names()

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
        return self._providers.list_by_stage(stage)

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
        return self._providers.list_by_kind(kind)

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
        return self._providers.list_providing(capability)

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
        return self._providers.list_by_table(table_key)

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
        return self._providers.metadata_for(name)

    def dependency_graph(self) -> dict[str, tuple[str, ...]]:
        """Return mapping of plugin name to direct dependencies.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Direct dependency map keyed by plugin name.
        """
        self._ensure_loaded()
        return self._providers.dependency_graph()

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

    def resolve_selection_debug(
        self,
        *,
        plugin_names: Sequence[str] | None,
        enabled: Sequence[str] | None,
        disabled: Sequence[str] | None,
        defaults: Sequence[str] | None = None,
    ) -> tuple[dict[str, P], tuple[PluginSkip, ...]]:
        """
        Public wrapper for selection resolution used in tests and tooling.

        Returns
        -------
        tuple[dict[str, P], tuple[PluginSkip, ...]]
            Selected plugins and skipped plugin reasons.
        """
        self._ensure_loaded()
        base_defaults = defaults if defaults is not None else self._get_default_plugins()
        return self._resolve_selection(
            plugin_names=plugin_names,
            enabled=enabled,
            disabled=disabled,
            defaults=base_defaults,
        )

    @staticmethod
    def _resolve_dependencies[Q: RegistrablePlugin](
        selected: dict[str, Q],
    ) -> dict[str, set[str]]:
        """Build dependency graph for selected plugins.

        Parameters
        ----------
        selected
            Selected plugins keyed by name.

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

        # Capability-based dependencies using shared utility
        provider_index = build_provider_index_from_metadata(
            selected,
            lambda p: p.metadata.provides,
        )
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
    def resolve_dependencies_debug[Q: RegistrablePlugin](
        selected: dict[str, Q],
    ) -> dict[str, set[str]]:
        """
        Public wrapper for dependency resolution used in tests and tooling.

        Returns
        -------
        dict[str, set[str]]
            Dependency mapping for the selected plugins.
        """
        return BasePluginRegistry._resolve_dependencies(selected)

    @staticmethod
    def _topological_sort[Q](
        selected: dict[str, Q],
        dependencies: dict[str, set[str]],
    ) -> list[Q]:
        """Perform topological sort with cycle detection.

        Delegates to the shared `topological_sort` utility from
        `codeintel.core.plugins.sorting`.

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

        Notes
        -----
        May raise `ValueError` if a dependency cycle is detected.
        """
        return topological_sort(selected, dependencies)

    def topological_sort_debug[Q](
        self,
        selected: dict[str, Q],
        dependencies: Mapping[str, set[str]],
    ) -> tuple[Q, ...]:
        """
        Public wrapper for topological sort used in tests and tooling.

        Returns
        -------
        tuple[Q, ...]
            Plugins ordered respecting dependencies.
        """
        ordered = self._topological_sort(dict(selected), dict(dependencies))
        return tuple(ordered)

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Ensure plugins are loaded (builtins + entry points)."""
        self._ensure_builtins_loaded()
        self._ensure_entrypoints_loaded()

    def _ensure_builtins_loaded(self) -> None:
        """Load built-in plugins if not already done."""
        if self._builtins_loaded:
            return
        self._hooks.load_builtins()
        self._builtins_loaded = True

    def _ensure_entrypoints_loaded(self) -> None:
        """Load plugins from entry points if not already done."""
        if self._entrypoints_loaded:
            return
        self.load_from_entrypoints()

    # -------------------------------------------------------------------------
    # Entry Point Discovery
    # -------------------------------------------------------------------------

    def load_from_entrypoints(
        self,
        *,
        group: str | None = None,
        force: bool = False,
    ) -> tuple[P, ...]:
        """Discover and register plugins from entry points.

        This method loads plugins from Python entry points, resolving each
        entry point through the configured hooks for domain-specific resolution.

        Parameters
        ----------
        group
            Entry point group to load from. Defaults to hooks.entrypoint_group.
        force
            Whether to reload even if already loaded.

        Returns
        -------
        tuple[P, ...]
            Tuple of newly registered plugins.
        """
        if self._entrypoints_loaded and not force:
            return ()

        effective_group = group or self._hooks.entrypoint_group
        discovered: list[P] = []
        eps = importlib.metadata.entry_points()
        selected = eps.select(group=effective_group)

        for entry_point in selected:
            try:
                loaded = entry_point.load()
                plugin = self._hooks.resolve_entrypoint(loaded)
                if plugin is not None:
                    self.register(plugin)
                    discovered.append(plugin)
                    log.info("Discovered plugin from entrypoint: %s", plugin.metadata.name)
                else:
                    log.warning(
                        "Entry point %s did not return a valid plugin",
                        entry_point.name,
                    )
            except (ImportError, AttributeError, TypeError) as exc:
                log.warning(
                    "Failed to load plugin from entrypoint %s: %s",
                    entry_point.name,
                    exc,
                )

        self._entrypoints_loaded = True
        return tuple(discovered)


__all__ = [
    "BasePluginRegistry",
    "DefaultRegistryHooks",
    "PluginPlan",
    "PluginSkip",
    "RegistrablePlugin",
    "RegistryEntry",
    "RegistryHooks",
]
