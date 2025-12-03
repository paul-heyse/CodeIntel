"""Unified plugin registry for analytics plugins.

This module provides a single, centralized registry for all analytics plugins.
It supports both decorator-based registration and explicit registration,
as well as entry point discovery.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeVar, overload
from uuid import uuid4

from codeintel.analytics.core.plugin_protocol import (
    AnalyticsPluginProtocol,
    PluginCapability,
    PluginInputSpec,
    PluginMetadata,
    PluginOutputSpec,
    PluginResourceHints,
    PluginResult,
    PluginSeverity,
    PluginStage,
    ValidationResult,
)
from codeintel.core.singleton import SingletonHolder

if TYPE_CHECKING:
    from codeintel.analytics.core.execution_context import PluginExecutionContext

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class PluginSkip:
    """Skip metadata for plugins excluded from execution."""

    name: str
    reason: Literal["disabled", "missing_dependency", "config_error"]


@dataclass(frozen=True)
class PluginPlan:
    """Resolved execution plan for a set of plugins.

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

    plugins: tuple[AnalyticsPluginProtocol, ...]
    plan_id: str = field(default_factory=lambda: uuid4().hex)
    skipped: tuple[PluginSkip, ...] = ()
    dep_graph: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def ordered_names(self) -> tuple[str, ...]:
        """Return plugin names in execution order."""
        return tuple(p.metadata.name for p in self.plugins)


class PluginRegistry:
    """Central registry for analytics plugins.

    The registry provides:
    - Plugin registration (decorator, explicit, entry point)
    - Plugin lookup by name
    - Dependency resolution and topological ordering
    - Capability-based discovery
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._plugins: dict[str, AnalyticsPluginProtocol] = {}
        self._by_capability: dict[str, set[str]] = {}
        self._by_stage: dict[PluginStage, set[str]] = {}
        self._entrypoints_loaded: bool = False

    def register(self, plugin: AnalyticsPluginProtocol) -> None:
        """Register a plugin instance.

        Parameters
        ----------
        plugin
            Plugin instance implementing AnalyticsPluginProtocol.

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

        # Index by capabilities
        for cap in meta.capabilities_provided:
            self._by_capability.setdefault(cap.name, set()).add(meta.name)

        # Index by stage
        self._by_stage.setdefault(meta.stage, set()).add(meta.name)

        log.debug("Registered plugin %s (stage=%s)", meta.name, meta.stage)

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
        for cap in meta.capabilities_provided:
            if cap.name in self._by_capability:
                self._by_capability[cap.name].discard(name)

        if meta.stage in self._by_stage:
            self._by_stage[meta.stage].discard(name)

    def get(self, name: str) -> AnalyticsPluginProtocol:
        """Return a plugin by name.

        Parameters
        ----------
        name
            Plugin name to look up.

        Returns
        -------
        AnalyticsPluginProtocol
            The registered plugin.

        Raises
        ------
        KeyError
            If no plugin is registered with the given name.
        """
        self._ensure_entrypoints_loaded()
        if name not in self._plugins:
            message = f"Unknown plugin: {name}"
            raise KeyError(message)
        return self._plugins[name]

    def list_all(self) -> tuple[AnalyticsPluginProtocol, ...]:
        """Return all registered plugins.

        Returns
        -------
        tuple[AnalyticsPluginProtocol, ...]
            All registered plugins in registration order.
        """
        self._ensure_entrypoints_loaded()
        return tuple(self._plugins.values())

    def list_by_stage(self, stage: PluginStage) -> tuple[AnalyticsPluginProtocol, ...]:
        """Return plugins for a specific stage.

        Parameters
        ----------
        stage
            Stage to filter by.

        Returns
        -------
        tuple[AnalyticsPluginProtocol, ...]
            Plugins in the requested stage.
        """
        self._ensure_entrypoints_loaded()
        names = self._by_stage.get(stage, set())
        return tuple(self._plugins[name] for name in names if name in self._plugins)

    def list_providing(self, capability: str) -> tuple[AnalyticsPluginProtocol, ...]:
        """Return plugins that provide a specific capability.

        Parameters
        ----------
        capability
            Capability name to search for.

        Returns
        -------
        tuple[AnalyticsPluginProtocol, ...]
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
    ) -> PluginPlan:
        """Build an execution plan with dependency resolution.

        Dependency resolution will fail with a ValueError if cycles or missing
        dependencies are detected.

        Parameters
        ----------
        plugin_names
            Explicit plugin names to include.
        enabled
            Override list of enabled plugins.
        disabled
            Plugins to exclude from the plan.

        Returns
        -------
        PluginPlan
            Ordered execution plan.
        """
        self._ensure_entrypoints_loaded()

        # Resolve which plugins to include
        selected, skipped = self._resolve_selection(
            plugin_names=plugin_names,
            enabled=enabled,
            disabled=disabled,
        )

        # Build dependency graph
        dependencies = PluginRegistry._resolve_dependencies(selected)

        # Topological sort
        ordered = self._topological_sort(selected, dependencies)

        return PluginPlan(
            plugins=tuple(ordered),
            skipped=skipped,
            dep_graph={name: tuple(sorted(deps)) for name, deps in dependencies.items()},
        )

    def load_from_entrypoints(
        self,
        *,
        group: str = "codeintel.analytics.plugins",
        force: bool = False,
    ) -> tuple[AnalyticsPluginProtocol, ...]:
        """Discover and register plugins from entry points.

        Parameters
        ----------
        group
            Entry point group to load from.
        force
            Whether to reload even if already loaded.

        Returns
        -------
        tuple[AnalyticsPluginProtocol, ...]
            Newly loaded plugins.

        Raises
        ------
        TypeError
            If an entry point does not return a valid plugin.
        """
        if self._entrypoints_loaded and not force:
            return ()

        discovered: list[AnalyticsPluginProtocol] = []
        for entry_point in importlib.metadata.entry_points().select(group=group):
            plugin = entry_point.load()
            if not isinstance(plugin, AnalyticsPluginProtocol):
                message = f"Entry point {entry_point.name} did not return AnalyticsPluginProtocol"
                raise TypeError(message)
            self.register(plugin)
            discovered.append(plugin)
            log.info("Discovered plugin from entrypoint: %s", plugin.metadata.name)

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
    ) -> tuple[dict[str, AnalyticsPluginProtocol], tuple[PluginSkip, ...]]:
        """Resolve which plugins to include in the plan.

        Returns
        -------
        tuple[dict[str, AnalyticsPluginProtocol], tuple[PluginSkip, ...]]
            Selected plugins keyed by name and plugins that were skipped.
        """
        # Determine base selection
        if enabled:
            names = list(enabled)
        elif plugin_names:
            names = list(plugin_names)
        else:
            # Default: all enabled-by-default plugins
            names = [name for name, p in self._plugins.items() if p.metadata.enabled_by_default]

        disabled_set = set(disabled or ())
        selected: dict[str, AnalyticsPluginProtocol] = {}
        skipped: list[PluginSkip] = []

        for name in names:
            if name in disabled_set:
                skipped.append(PluginSkip(name=name, reason="disabled"))
                continue
            if name in selected:
                continue
            selected[name] = self.get(name)

        return selected, tuple(skipped)

    @staticmethod
    def _resolve_dependencies(
        selected: dict[str, AnalyticsPluginProtocol],
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
                    log.debug(
                        "Skipping unmet dependency %s for plugin %s",
                        dep,
                        name,
                    )

        # Capability-based dependencies
        capability_providers = PluginRegistry._build_capability_index(selected)
        for name, plugin in selected.items():
            for cap in plugin.metadata.capabilities_required:
                providers = capability_providers.get(cap.name, set())
                if not providers:
                    log.warning(
                        "Plugin %s requires capability %s but no provider is selected",
                        name,
                        cap.name,
                    )
                    continue
                # Add first provider as dependency (unless self-provided)
                for provider in providers:
                    if provider != name:
                        dependencies[name].add(provider)
                        break

        return dependencies

    @staticmethod
    def _build_capability_index(
        selected: dict[str, AnalyticsPluginProtocol],
    ) -> dict[str, set[str]]:
        """Build index of capability -> provider plugins.

        Returns
        -------
        dict[str, set[str]]
            Mapping of capability name to provider plugin names.
        """
        index: dict[str, set[str]] = {}
        for name, plugin in selected.items():
            for cap in plugin.metadata.capabilities_provided:
                index.setdefault(cap.name, set()).add(name)
        return index

    @staticmethod
    def _topological_sort(
        selected: dict[str, AnalyticsPluginProtocol],
        dependencies: dict[str, set[str]],
    ) -> list[AnalyticsPluginProtocol]:
        """Perform topological sort with cycle detection.

        Returns
        -------
        list[AnalyticsPluginProtocol]
            Plugins ordered based on dependencies.
        """
        ordered: list[AnalyticsPluginProtocol] = []
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(name: str) -> None:
            if name in permanent:
                return
            if name in temporary:
                message = f"Dependency cycle detected involving plugin: {name}"
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


# Singleton holder for plugin registry
class _PluginRegistryHolder(SingletonHolder["PluginRegistry"]):
    """Thread-safe singleton holder for PluginRegistry."""


def get_registry() -> PluginRegistry:
    """Return the global plugin registry.

    Returns
    -------
    PluginRegistry
        The singleton registry instance.
    """
    return _PluginRegistryHolder.get(PluginRegistry)


def reset_registry() -> None:
    """Reset the global plugin registry.

    Primarily useful for testing to ensure clean state between tests.
    """
    _PluginRegistryHolder.reset()


def register_plugin(plugin: AnalyticsPluginProtocol) -> None:
    """Register a plugin with the global registry.

    Parameters
    ----------
    plugin
        Plugin instance to register.
    """
    get_registry().register(plugin)


@dataclass
class FunctionalPlugin:
    """Plugin implementation wrapping a callable.

    This class provides a simple way to create plugins from functions
    using the @plugin decorator.
    """

    _metadata: PluginMetadata
    _execute_fn: Callable[[PluginExecutionContext], PluginResult]
    _validate_fn: Callable[[PluginExecutionContext], ValidationResult] | None = None

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        PluginMetadata
            Metadata for the wrapped plugin.
        """
        return self._metadata

    def execute(self, ctx: PluginExecutionContext) -> PluginResult:
        """Execute the wrapped function.

        Returns
        -------
        PluginResult
            Result produced by the underlying callable.
        """
        return self._execute_fn(ctx)

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
        """Validate inputs using the custom validator or default.

        Returns
        -------
        ValidationResult
            Validation result from the custom validator or a default success.
        """
        if self._validate_fn is not None:
            return self._validate_fn(ctx)
        return ValidationResult.success()


@overload
def plugin(
    func: Callable[[PluginExecutionContext], PluginResult],
) -> FunctionalPlugin: ...


@overload
def plugin(
    *,
    name: str,
    description: str,
    stage: PluginStage,
    version: str = "1.0.0",
    enabled_by_default: bool = True,
    severity: PluginSeverity = "fatal",
    inputs: Sequence[PluginInputSpec] = (),
    outputs: Sequence[PluginOutputSpec] = (),
    provides: Sequence[str | PluginCapability] = (),
    requires: Sequence[str | PluginCapability] = (),
    depends_on: Sequence[str] = (),
    resource_hints: PluginResourceHints | None = None,
    requires_isolation: bool = False,
    isolation_kind: Literal["process", "thread"] | None = None,
    tags: Sequence[str] = (),
    register: bool = True,
) -> Callable[[Callable[[PluginExecutionContext], PluginResult]], FunctionalPlugin]: ...


def plugin(  # noqa: PLR0913 - decorator with many params by design
    func: Callable[[PluginExecutionContext], PluginResult] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    stage: PluginStage = "other",
    version: str = "1.0.0",
    enabled_by_default: bool = True,
    severity: PluginSeverity = "fatal",
    inputs: Sequence[PluginInputSpec] = (),
    outputs: Sequence[PluginOutputSpec] = (),
    provides: Sequence[str | PluginCapability] = (),
    requires: Sequence[str | PluginCapability] = (),
    depends_on: Sequence[str] = (),
    resource_hints: PluginResourceHints | None = None,
    requires_isolation: bool = False,
    isolation_kind: Literal["process", "thread"] | None = None,
    tags: Sequence[str] = (),
    register: bool = True,
) -> (
    FunctionalPlugin
    | Callable[[Callable[[PluginExecutionContext], PluginResult]], FunctionalPlugin]
):
    """Create and register plugins.

    Can be used in two ways:

    1. Simple decorator (derives metadata from function):
       @plugin
       def my_plugin(ctx): ...

    2. With explicit metadata:
       @plugin(name="my.plugin", description="...", stage="function")
       def my_plugin(ctx): ...

    Parameters
    ----------
    func
        The function to wrap (when used without arguments).
    name
        Plugin name (defaults to function name with dots).
    description
        Human-readable description (defaults to docstring).
    stage
        Processing stage.
    version
        Plugin version.
    enabled_by_default
        Whether enabled when no explicit list is provided.
    severity
        How failures should be handled.
    inputs
        Required/optional inputs.
    outputs
        Tables/artifacts produced.
    provides
        Capabilities provided (strings converted to PluginCapability).
    requires
        Capabilities required (strings converted to PluginCapability).
    depends_on
        Explicit plugin dependencies.
    resource_hints
        Runtime hints.
    requires_isolation
        Whether process/thread isolation is needed.
    isolation_kind
        Type of isolation.
    tags
        Free-form tags.
    register
        Whether to auto-register with global registry.

    Returns
    -------
    FunctionalPlugin | Callable
        The plugin instance or a decorator.
    """

    def _normalize_capability(cap: str | PluginCapability) -> PluginCapability:
        if isinstance(cap, PluginCapability):
            return cap
        return PluginCapability(name=cap)

    def _make_plugin(
        fn: Callable[[PluginExecutionContext], PluginResult],
    ) -> FunctionalPlugin:
        resolved_name = name or fn.__name__.replace("_", ".")
        resolved_description = description or fn.__doc__ or ""

        meta = PluginMetadata(
            name=resolved_name,
            description=resolved_description.strip(),
            stage=stage,
            version=version,
            enabled_by_default=enabled_by_default,
            severity=severity,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            capabilities_provided=tuple(_normalize_capability(c) for c in provides),
            capabilities_required=tuple(_normalize_capability(c) for c in requires),
            depends_on=tuple(depends_on),
            resource_hints=resource_hints,
            requires_isolation=requires_isolation,
            isolation_kind=isolation_kind,
            tags=tuple(tags),
        )

        plugin_instance = FunctionalPlugin(_metadata=meta, _execute_fn=fn)

        if register:
            get_registry().register(plugin_instance)

        return plugin_instance

    if func is not None:
        # Used as @plugin without arguments
        return _make_plugin(func)

    # Used as @plugin(...) with arguments
    return _make_plugin


__all__ = [
    "FunctionalPlugin",
    "PluginPlan",
    "PluginRegistry",
    "PluginSkip",
    "get_registry",
    "plugin",
    "register_plugin",
]
