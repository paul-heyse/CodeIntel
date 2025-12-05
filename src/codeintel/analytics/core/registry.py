"""Unified plugin registry for analytics plugins.

This module provides a single, centralized registry for all analytics plugins,
extending the base registry infrastructure from codeintel.core.plugins.
It supports both decorator-based registration and explicit registration,
as well as entry point discovery.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Unpack

# Import at runtime for use in type alias (FunctionalPlugin)
from codeintel.analytics.core.context import PluginExecutionContext
from codeintel.analytics.core.protocol import (
    AnalyticsPluginProtocol,
    PluginMetadata,
    PluginResult,
)
from codeintel.core.execution.ids import new_run_id
from codeintel.core.plugins.decorators.functional import BaseFunctionalPlugin
from codeintel.core.plugins.decorators.meta import (
    BasePluginMetaOptions,
    BasePluginMetaOptionsInput,
)
from codeintel.core.plugins.decorators.step import make_plugin_instance
from codeintel.core.plugins.registry.base import BasePluginRegistry, PluginSkip
from codeintel.core.singleton import SingletonHolder

log = logging.getLogger(__name__)


# =============================================================================
# Analytics-specific Plan types (PluginSkip is imported from core)
# =============================================================================


@dataclass(frozen=True)
class PluginPlan:
    """Resolved execution plan for a set of analytics plugins.

    Structurally equivalent to ``codeintel.core.plugins.registry.PluginPlan[P]``
    instantiated with ``P=AnalyticsPluginProtocol``. Kept as a separate class
    for simpler type annotations throughout analytics code.

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
    plan_id: str = field(default_factory=lambda: new_run_id("plan"))
    skipped: tuple[PluginSkip, ...] = ()
    dep_graph: dict[str, tuple[str, ...]] = field(default_factory=dict)

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
# Analytics Plugin Registry
# =============================================================================


class PluginRegistry(BasePluginRegistry[AnalyticsPluginProtocol]):
    """Central registry for analytics plugins.

    Extends BasePluginRegistry with analytics-specific functionality.
    Provides plugin registration (decorator, explicit, entry point),
    plugin lookup by name, dependency resolution and topological ordering,
    and capability-based discovery.
    """

    @staticmethod
    def _get_default_plugins() -> Sequence[str]:
        """Return the default analytics plugin names.

        Analytics uses enabled_by_default from each plugin's metadata
        rather than a static list.

        Returns
        -------
        Sequence[str]
            Empty sequence (analytics uses enabled_by_default instead).
        """
        return ()

    def _ensure_builtins_loaded(self) -> None:
        """Load built-in analytics plugins (currently not used)."""
        self._builtins_loaded = True

    @property
    def _default_entrypoint_group(self) -> str:
        """Return the analytics entry point group.

        Returns
        -------
        str
            Entry point group name for analytics plugins.
        """
        return "codeintel.analytics.plugins"

    def _is_valid_plugin(self, obj: object) -> bool:
        """Check if an object is a valid analytics plugin.

        Parameters
        ----------
        obj
            Object to validate.

        Returns
        -------
        bool
            True if the object implements AnalyticsPluginProtocol.
        """
        # Access self to satisfy PLR6301 while also checking for protocol
        _ = self._entrypoints_loaded  # Ensure registry state is accessible
        return isinstance(obj, AnalyticsPluginProtocol)

    def plan(
        self,
        plugin_names: Sequence[str] | None = None,
        *,
        enabled: Sequence[str] | None = None,
        disabled: Sequence[str] | None = None,
    ) -> PluginPlan:
        """Build an execution plan with dependency resolution.

        Dependency resolution raises ValueError if cycles or missing
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
        self._ensure_loaded()

        # Resolve which plugins to include
        selected, skipped = self._resolve_analytics_selection(
            plugin_names=plugin_names,
            enabled=enabled,
            disabled=disabled,
        )

        # Build dependency graph
        dependencies = self._resolve_analytics_dependencies(selected)

        # Topological sort (reuse base class static method)
        ordered = self._topological_sort(selected, dependencies)

        return PluginPlan(
            plugins=tuple(ordered),
            skipped=skipped,
            dep_graph={name: tuple(sorted(deps)) for name, deps in dependencies.items()},
        )

    def _resolve_analytics_selection(
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
            try:
                selected[name] = self.get(name)
            except KeyError:
                skipped.append(PluginSkip(name=name, reason="missing_dependency"))
                log.warning("Skipping unknown plugin: %s", name)

        return selected, tuple(skipped)

    def _resolve_analytics_dependencies(
        self,
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
                    log.debug("Skipping unmet dependency %s for plugin %s", dep, name)

        # Capability-based dependencies
        capability_providers = self._build_capability_index(selected)
        for name, plugin in selected.items():
            for cap_name in plugin.metadata.requires:
                providers = capability_providers.get(cap_name, set())
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
    def _build_capability_index(
        selected: dict[str, AnalyticsPluginProtocol],
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


# =============================================================================
# Singleton Access
# =============================================================================


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


# =============================================================================
# Decorator Support
# =============================================================================


class PluginMetaOptionsInput(BasePluginMetaOptionsInput, total=False):
    """Typed keyword arguments for PluginMetaOptions.from_kwargs factory.

    Extends BasePluginMetaOptionsInput with any analytics-specific fields.
    Currently identical to base, but can be extended as needed.
    """


@dataclass
class PluginMetaOptions(BasePluginMetaOptions):
    """Options container for analytics plugin metadata.

    Extends BasePluginMetaOptions with analytics-specific defaults.
    Grouping metadata in a single object keeps decorator signatures small
    and makes future evolution easier.
    """

    @staticmethod
    def from_kwargs(**kwargs: Unpack[PluginMetaOptionsInput]) -> PluginMetaOptions:
        """Build options from keyword arguments with validation.

        Delegates validation to BasePluginMetaOptions.validate_option_keys,
        which raises ValueError if unknown keys are provided.

        Parameters
        ----------
        **kwargs
            Keyword arguments matching PluginMetaOptionsInput fields.

        Returns
        -------
        PluginMetaOptions
            Options built from the provided keyword arguments.
        """
        BasePluginMetaOptions.validate_option_keys(
            set(PluginMetaOptionsInput.__annotations__),
            kwargs,
        )
        return PluginMetaOptions(**kwargs)

    def to_metadata(
        self,
        fn: Callable[[PluginExecutionContext], PluginResult],
    ) -> PluginMetadata:
        """Convert options to PluginMetadata using function defaults.

        Parameters
        ----------
        fn
            Plugin callable used for deriving defaults (name/docstring).

        Returns
        -------
        PluginMetadata
            Metadata populated from options with analytics defaults.
        """
        return self.to_base_metadata(fn, default_kind="analytics", default_stage="other")


# Type alias for analytics functional plugin using the base class
FunctionalPlugin = BaseFunctionalPlugin[PluginExecutionContext, PluginMetadata]
"""Analytics plugin implementation wrapping a callable.

This type alias provides analytics-specific typing for the base functional
plugin class. Use with the @plugin decorator.
"""


def plugin(
    func: Callable[[PluginExecutionContext], PluginResult] | None = None,
    *,
    meta: PluginMetaOptions | None = None,
    register: bool = True,
    **kwargs: Unpack[PluginMetaOptionsInput],
) -> (
    FunctionalPlugin
    | Callable[[Callable[[PluginExecutionContext], PluginResult]], FunctionalPlugin]
):
    """Create and register analytics plugins.

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
    meta
        Plugin metadata/options container.
    register
        Whether to auto-register with global registry.
    **kwargs
        Legacy metadata fields; prefer `meta`.

    Returns
    -------
    FunctionalPlugin | Callable
        The plugin instance or a decorator.
    """

    def _make_plugin(
        fn: Callable[[PluginExecutionContext], PluginResult],
    ) -> FunctionalPlugin:
        if meta is not None and kwargs:
            message = "Provide either meta or individual keyword options, not both."
            raise ValueError(message)

        options = meta or PluginMetaOptions.from_kwargs(**kwargs)
        register_fn = get_registry().register if register else None

        return make_plugin_instance(
            fn=fn,
            options=options,
            plugin_factory=lambda m, f: FunctionalPlugin(_metadata=m, _execute_fn=f),
            to_metadata=lambda opts, f: opts.to_metadata(f),
            register_fn=register_fn,
        )

    if func is not None:
        # Used as @plugin without arguments
        return _make_plugin(func)

    # Used as @plugin(...) with arguments
    return _make_plugin


__all__ = [
    "FunctionalPlugin",
    "PluginMetaOptions",
    "PluginPlan",
    "PluginRegistry",
    "PluginSkip",
    "get_registry",
    "plugin",
    "register_plugin",
    "reset_registry",
]
