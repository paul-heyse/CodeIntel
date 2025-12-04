"""Ingestion plugin registry with dependency resolution and entry-point discovery.

This module provides the registry for ingestion plugins, supporting
decorator-based registration, dependency resolution, topological ordering,
and discovery via Python entry points.
"""

from __future__ import annotations

import importlib.metadata
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from codeintel.core.plugins.sorting import (
    build_provider_index_from_metadata,
    topological_sort,
)
from codeintel.ingestion.plugins.ast_extract import AstExtractPlugin
from codeintel.ingestion.plugins.config_plugin import ConfigIngestPlugin
from codeintel.ingestion.plugins.coverage_plugin import CoverageIngestPlugin
from codeintel.ingestion.plugins.cst_extract import CstExtractPlugin
from codeintel.ingestion.plugins.docstrings_plugin import DocstringsIngestPlugin
from codeintel.ingestion.plugins.protocol import (
    DEFAULT_INGEST_PLUGINS,
    IngestPluginPlan,
    IngestPluginProtocol,
    IngestPluginSkip,
    is_ingest_plugin,
)
from codeintel.ingestion.plugins.repo_scan import RepoScanPlugin
from codeintel.ingestion.plugins.scip_plugin import ScipIngestPlugin
from codeintel.ingestion.plugins.tests_plugin import TestsIngestPlugin
from codeintel.ingestion.plugins.typing_plugin import TypingIngestPlugin

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanOptions:
    """Options for plugin plan creation.

    Encapsulates all parameters for the registry's plan() method,
    reducing argument count and improving API clarity.

    Attributes
    ----------
    plugin_names
        Explicit plugin names to include.
    enabled
        Override list of enabled plugins.
    disabled
        Plugins to exclude from the plan.
    defaults
        Default plugins if no explicit list provided.
    check_tools
        Whether to check tool availability.
    available_tools
        List of available tool plugins.
    """

    plugin_names: Sequence[str] | None = None
    enabled: Sequence[str] | None = None
    disabled: Sequence[str] | None = None
    defaults: Sequence[str] | None = None
    check_tools: bool = False
    available_tools: Sequence[str] | None = None


class IngestPluginRegistry:
    """Central registry for ingestion plugins.

    Provides plugin registration, lookup, dependency resolution,
    and topological ordering for execution planning.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._plugins: dict[str, IngestPluginProtocol] = {}
        self._by_capability: dict[str, set[str]] = {}
        self._by_stage: dict[str, set[str]] = {}
        self._by_table: dict[str, set[str]] = {}
        self._entrypoints_loaded: bool = False

    def register(self, plugin: IngestPluginProtocol) -> None:
        """Register an ingestion plugin.

        Parameters
        ----------
        plugin
            Plugin instance implementing IngestPluginProtocol.

        Raises
        ------
        ValueError
            If a plugin with the same name is already registered.
        """
        meta = plugin.metadata
        if meta.name in self._plugins:
            message = f"Duplicate ingest plugin name: {meta.name}"
            raise ValueError(message)

        self._plugins[meta.name] = plugin

        # Index by capabilities
        for cap in meta.provides:
            self._by_capability.setdefault(cap, set()).add(meta.name)

        # Index by stage
        self._by_stage.setdefault(meta.stage, set()).add(meta.name)

        # Index by produced tables
        for table in meta.produces_tables:
            self._by_table.setdefault(table, set()).add(meta.name)

        log.debug(
            "Registered ingest plugin %s (stage=%s, tables=%s)",
            meta.name,
            meta.stage,
            meta.produces_tables,
        )

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

        if meta.stage in self._by_stage:
            self._by_stage[meta.stage].discard(name)

        for table in meta.produces_tables:
            if table in self._by_table:
                self._by_table[table].discard(name)

    def get(self, name: str) -> IngestPluginProtocol:
        """Return a plugin by name.

        Parameters
        ----------
        name
            Plugin name to look up.

        Returns
        -------
        IngestPluginProtocol
            The registered plugin.

        Raises
        ------
        KeyError
            If no plugin is registered with the given name.
        """
        self._ensure_entrypoints_loaded()
        if name not in self._plugins:
            message = f"Unknown ingest plugin: {name}"
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
        self._ensure_entrypoints_loaded()
        return name in self._plugins

    def list_all(self) -> tuple[IngestPluginProtocol, ...]:
        """Return all registered plugins.

        Returns
        -------
        tuple[IngestPluginProtocol, ...]
            All registered plugins in registration order.
        """
        self._ensure_entrypoints_loaded()
        return tuple(self._plugins.values())

    def list_names(self) -> tuple[str, ...]:
        """Return names of all registered plugins.

        Returns
        -------
        tuple[str, ...]
            Plugin names in registration order.
        """
        self._ensure_entrypoints_loaded()
        return tuple(self._plugins.keys())

    def list_providing(self, capability: str) -> tuple[IngestPluginProtocol, ...]:
        """Return plugins that provide a specific capability.

        Parameters
        ----------
        capability
            Capability name to search for.

        Returns
        -------
        tuple[IngestPluginProtocol, ...]
            Plugins providing the capability.
        """
        self._ensure_entrypoints_loaded()
        names = self._by_capability.get(capability, set())
        return tuple(self._plugins[name] for name in names if name in self._plugins)

    def list_by_stage(self, stage: str) -> tuple[IngestPluginProtocol, ...]:
        """Return plugins belonging to a specific stage.

        Parameters
        ----------
        stage
            Stage name to filter by.

        Returns
        -------
        tuple[IngestPluginProtocol, ...]
            Plugins in the specified stage.
        """
        self._ensure_entrypoints_loaded()
        names = self._by_stage.get(stage, set())
        return tuple(self._plugins[name] for name in names if name in self._plugins)

    def list_by_table(self, table_key: str) -> tuple[IngestPluginProtocol, ...]:
        """Return plugins that produce a specific table.

        Parameters
        ----------
        table_key
            Table key (e.g., "core.ast_nodes").

        Returns
        -------
        tuple[IngestPluginProtocol, ...]
            Plugins producing the table.
        """
        self._ensure_entrypoints_loaded()
        names = self._by_table.get(table_key, set())
        return tuple(self._plugins[name] for name in names if name in self._plugins)

    def plan(self, options: PlanOptions | None = None) -> IngestPluginPlan:
        """Build an execution plan with dependency resolution.

        Parameters
        ----------
        options
            Plan options encapsulating plugin selection criteria.
            If None, uses default options.

        Returns
        -------
        IngestPluginPlan
            Ordered execution plan.
        """
        self._ensure_entrypoints_loaded()
        opts = options or PlanOptions()

        # Resolve which plugins to include
        selected, skipped = self._resolve_selection(opts)

        # Build dependency graph
        dependencies = self._resolve_dependencies(selected)

        # Topological sort
        ordered = self._topological_sort(selected, dependencies)

        return IngestPluginPlan(
            plugins=tuple(ordered),
            plan_id=uuid4().hex,
            skipped_plugins=skipped,
            dep_graph={name: tuple(sorted(deps)) for name, deps in dependencies.items()},
        )

    def load_from_entrypoints(
        self,
        *,
        group: str = "codeintel.ingest_plugins",
        force: bool = False,
    ) -> tuple[IngestPluginProtocol, ...]:
        """Discover and register plugins from entry points.

        Parameters
        ----------
        group
            Entry point group to load from.
        force
            Whether to reload even if already loaded.

        Returns
        -------
        tuple[IngestPluginProtocol, ...]
            Newly loaded plugins.
        """
        if self._entrypoints_loaded and not force:
            return ()

        discovered: list[IngestPluginProtocol] = []
        eps = importlib.metadata.entry_points()
        # Python 3.10+ uses select(); group kwarg may be unavailable
        selected = eps.select(group=group)

        for entry_point in selected:
            try:
                loaded = entry_point.load()
                # Support both direct plugin instances and factory functions
                if isinstance(loaded, type) or (
                    callable(loaded) and not hasattr(loaded, "metadata")
                ):
                    candidate = loaded()
                else:
                    candidate = loaded

                # Use TypeGuard for proper type narrowing
                if not is_ingest_plugin(candidate):
                    log.warning(
                        "Entry point %s did not return IngestPluginProtocol",
                        entry_point.name,
                    )
                    continue

                # TypeGuard narrows candidate to IngestPluginProtocol
                self.register(candidate)
                discovered.append(candidate)
                log.info("Discovered ingest plugin from entrypoint: %s", candidate.metadata.name)
            except (ImportError, AttributeError, TypeError) as exc:
                log.warning(
                    "Failed to load ingest plugin from entrypoint %s: %s",
                    entry_point.name,
                    exc,
                )

        self._entrypoints_loaded = True
        return tuple(discovered)

    def _ensure_entrypoints_loaded(self) -> None:
        """Load entry points and class-based plugins if not already done."""
        if not self._entrypoints_loaded:
            self.load_from_entrypoints()
            # Also register the class-based plugins directly
            self._register_builtin_plugins()

    def _register_builtin_plugins(self) -> None:
        """Register the built-in class-based plugins."""
        plugins: list[IngestPluginProtocol] = [
            RepoScanPlugin(),
            AstExtractPlugin(),
            CstExtractPlugin(),
            ScipIngestPlugin(),
            TypingIngestPlugin(),
            CoverageIngestPlugin(),
            TestsIngestPlugin(),
            DocstringsIngestPlugin(),
            ConfigIngestPlugin(),
        ]

        for plugin in plugins:
            if plugin.metadata.name not in self._plugins:
                self.register(plugin)

    def _resolve_selection(
        self,
        options: PlanOptions,
    ) -> tuple[dict[str, IngestPluginProtocol], tuple[IngestPluginSkip, ...]]:
        """Resolve which plugins to include in the plan.

        Parameters
        ----------
        options
            Plan options with selection criteria.

        Returns
        -------
        tuple[dict[str, IngestPluginProtocol], tuple[IngestPluginSkip, ...]]
            Selected plugins keyed by name and plugins that were skipped.

        Raises
        ------
        ValueError
            If a plugin name appears more than once.
        """
        defaults = options.defaults or DEFAULT_INGEST_PLUGINS

        # Determine base selection
        if options.enabled:
            names = list(options.enabled)
        elif options.plugin_names:
            names = list(options.plugin_names)
        else:
            names = list(defaults)

        disabled_set = set(options.disabled or ())
        available_tools_set = set(options.available_tools or ())
        selected: dict[str, IngestPluginProtocol] = {}
        skipped: list[IngestPluginSkip] = []

        for name in names:
            if name in disabled_set:
                skipped.append(IngestPluginSkip(name=name, reason="disabled"))
                continue

            if name in selected:
                message = f"Ingest plugin '{name}' listed more than once"
                raise ValueError(message)

            try:
                plugin = self.get(name)
            except KeyError:
                skipped.append(IngestPluginSkip(name=name, reason="missing_dependency"))
                log.warning("Skipping unknown ingest plugin: %s", name)
                continue

            # Check tool dependencies if requested
            if options.check_tools and plugin.metadata.tool_dependencies:
                missing_tools = set(plugin.metadata.tool_dependencies) - available_tools_set
                if missing_tools:
                    skipped.append(IngestPluginSkip(name=name, reason="missing_tool"))
                    log.info(
                        "Skipping ingest plugin %s due to missing tools: %s",
                        name,
                        missing_tools,
                    )
                    continue

            selected[name] = plugin

        return selected, tuple(skipped)

    @staticmethod
    def _resolve_dependencies(
        selected: dict[str, IngestPluginProtocol],
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
                        f"Ingest plugin '{name}' depends on '{dep}', "
                        "which is not in the selected plugin set"
                    )
                    raise ValueError(message)
                dependencies[name].add(dep)

        # Capability-based dependencies using shared utility
        provider_index = build_provider_index_from_metadata(
            selected,
            lambda p: p.metadata.provides,
        )
        for name, plugin in selected.items():
            for requirement in plugin.metadata.requires:
                providers = provider_index.get(requirement, set())
                if not providers:
                    message = (
                        f"Ingest plugin '{name}' requires capability '{requirement}', "
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
                        f"Ingest plugin '{name}' requires capability '{requirement}', "
                        f"but multiple providers are available ({provider_list}). "
                        "Add an explicit depends_on entry to disambiguate."
                    )
                    raise ValueError(message)
                dependencies[name].add(next(iter(providers)))

        return dependencies

    @staticmethod
    def _build_provider_index(
        selected: dict[str, IngestPluginProtocol],
    ) -> dict[str, set[str]]:
        """Build index of capability -> provider plugins.

        Delegates to the shared utility from `codeintel.core.plugins.sorting`.

        Parameters
        ----------
        selected
            Selected plugins to index.

        Returns
        -------
        dict[str, set[str]]
            Mapping of capability name to provider plugin names.
        """
        return build_provider_index_from_metadata(
            selected,
            lambda p: p.metadata.provides,
        )

    @staticmethod
    def _topological_sort(
        selected: dict[str, IngestPluginProtocol],
        dependencies: dict[str, set[str]],
    ) -> list[IngestPluginProtocol]:
        """Perform topological sort with cycle detection.

        Delegates to the shared utility from `codeintel.core.plugins.sorting`.

        Parameters
        ----------
        selected
            Selected plugins keyed by name.
        dependencies
            Dependency graph.

        Returns
        -------
        list[IngestPluginProtocol]
            Plugins ordered based on dependencies.

        Notes
        -----
        May raise `ValueError` if a dependency cycle is detected.
        """
        return topological_sort(selected, dependencies)

    def dependency_graph(self) -> dict[str, tuple[str, ...]]:
        """Return mapping of plugin name to direct dependencies.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Direct dependency map keyed by plugin name.
        """
        self._ensure_entrypoints_loaded()
        return {name: plugin.metadata.depends_on for name, plugin in self._plugins.items()}

    def metadata_for(self, name: str) -> dict[str, object]:
        """Return metadata dictionary for a plugin.

        Parameters
        ----------
        name
            Plugin name.

        Returns
        -------
        dict[str, object]
            Metadata as a dictionary.
        """
        plugin = self.get(name)
        meta = plugin.metadata
        return {
            "name": meta.name,
            "description": meta.description,
            "stage": meta.stage,
            "severity": meta.severity,
            "enabled_by_default": meta.enabled_by_default,
            "depends_on": meta.depends_on,
            "provides": meta.provides,
            "requires": meta.requires,
            "produces_tables": meta.produces_tables,
            "tool_dependencies": meta.tool_dependencies,
            "supports_incremental": meta.supports_incremental,
            "isolation_kind": meta.isolation_kind,
        }


# Thread-safe singleton holder for the global registry


class _RegistrySingleton:
    """Thread-safe singleton holder for the ingest plugin registry.

    Uses double-checked locking to ensure thread-safe initialization
    while minimizing lock contention after initialization.
    """

    _instance: IngestPluginRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get(cls) -> IngestPluginRegistry:
        """Return the singleton registry instance.

        Thread-safe with double-checked locking.

        Returns
        -------
        IngestPluginRegistry
            The singleton registry instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = IngestPluginRegistry()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton for testing.

        Should only be called in test fixtures.
        """
        with cls._lock:
            cls._instance = None


def get_ingest_registry() -> IngestPluginRegistry:
    """Return the global ingest plugin registry.

    Returns
    -------
    IngestPluginRegistry
        The singleton registry instance.
    """
    return _RegistrySingleton.get()


def register_ingest_plugin(plugin: IngestPluginProtocol) -> None:
    """Register a plugin with the global registry.

    Parameters
    ----------
    plugin
        Plugin instance to register.
    """
    get_ingest_registry().register(plugin)


def list_ingest_plugins() -> tuple[IngestPluginProtocol, ...]:
    """Return all registered ingest plugins.

    Returns
    -------
    tuple[IngestPluginProtocol, ...]
        All registered ingest plugins.
    """
    return get_ingest_registry().list_all()


def plan_ingest_plugins(options: PlanOptions | None = None) -> IngestPluginPlan:
    """Build an execution plan for ingest plugins.

    Parameters
    ----------
    options
        Plan options encapsulating plugin selection criteria.
        If None, uses default options.

    Returns
    -------
    IngestPluginPlan
        Ordered execution plan.
    """
    return get_ingest_registry().plan(options)


def register_class_based_plugins() -> tuple[IngestPluginProtocol, ...]:
    """Register all class-based plugins with the global registry.

    Register instances of the new class-based plugins. This function
    can be called to switch from functional plugins to class-based plugins.

    Returns
    -------
    tuple[IngestPluginProtocol, ...]
        The registered plugin instances.

    Notes
    -----
    This function creates new instances of each plugin class. If you need
    to customize plugin instances, create them manually and use
    `register_ingest_plugin()` instead.

    The class-based plugins provide the same functionality as the
    functional plugins in builtin.py but with a cleaner architecture
    that supports traits, middleware, and resource providers.
    """
    registry = get_ingest_registry()

    # Create plugin instances
    plugins: list[IngestPluginProtocol] = [
        RepoScanPlugin(),
        AstExtractPlugin(),
        CstExtractPlugin(),
        ScipIngestPlugin(),
        TypingIngestPlugin(),
        CoverageIngestPlugin(),
        TestsIngestPlugin(),
        DocstringsIngestPlugin(),
        ConfigIngestPlugin(),
    ]

    registered: list[IngestPluginProtocol] = []
    for plugin in plugins:
        try:
            registry.register(plugin)
            registered.append(plugin)
            log.info("Registered class-based plugin: %s", plugin.metadata.name)
        except ValueError:
            # Plugin already registered (likely functional version)
            log.debug(
                "Skipping class-based plugin %s (already registered)",
                plugin.metadata.name,
            )

    return tuple(registered)


__all__ = [
    "IngestPluginRegistry",
    "PlanOptions",
    "get_ingest_registry",
    "list_ingest_plugins",
    "plan_ingest_plugins",
    "register_class_based_plugins",
    "register_ingest_plugin",
]
