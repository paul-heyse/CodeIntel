"""Plugin traits for capability-based composition.

This module defines protocol classes (traits) that plugins can implement
to declare specific capabilities. The runtime uses these traits to:
- Automatically prepare contexts with required resources
- Validate plugin requirements
- Enable trait-based plugin discovery
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.analytics.core.execution_context import PluginExecutionContext
    from codeintel.analytics.graphs.contracts import ContractChecker
    from codeintel.analytics.runtime_manifest import AnalyticsScope
    from codeintel.graphs.engine import GraphKind


@runtime_checkable
class GraphAwarePlugin(Protocol):
    """Trait for plugins that require graph runtime.

    Plugins implementing this trait declare which graph types they need,
    allowing the runtime to ensure those graphs are loaded before execution.
    """

    def get_graph_requirements(self) -> tuple[GraphKind, ...]:
        """Return required graph types.

        Returns
        -------
        tuple[GraphKind, ...]
            Graph types this plugin requires.
        """
        ...


@runtime_checkable
class ScopeAwarePlugin(Protocol):
    """Trait for plugins that support scoped execution.

    Plugins implementing this trait can filter their analysis to specific
    paths, modules, or time windows.
    """

    @property
    def supported_scopes(self) -> tuple[Literal["paths", "modules", "time_window"], ...]:
        """Return supported scope types.

        Returns
        -------
        tuple[Literal["paths", "modules", "time_window"], ...]
            Scope types this plugin supports.
        """
        ...

    def filter_by_scope(self, scope: AnalyticsScope) -> bool:
        """Check if the plugin should execute for the given scope.

        Parameters
        ----------
        scope
            Execution scope to check.

        Returns
        -------
        bool
            True if the plugin should execute.
        """
        ...


@runtime_checkable
class ContractValidatedPlugin(Protocol):
    """Trait for plugins with post-execution contracts.

    Plugins implementing this trait declare contract checkers that run
    after execution to validate outputs.
    """

    @property
    def contract_checkers(self) -> tuple[ContractChecker, ...]:
        """Return contract checkers for this plugin.

        Returns
        -------
        tuple[ContractChecker, ...]
            Contract checkers to run after execution.
        """
        ...


@runtime_checkable
class IsolatedPlugin(Protocol):
    """Trait for plugins requiring process or thread isolation.

    Plugins implementing this trait will be executed in a separate
    process or thread to prevent interference with other plugins.
    """

    @property
    def isolation_kind(self) -> Literal["process", "thread"]:
        """Return the isolation type required.

        Returns
        -------
        Literal["process", "thread"]
            Type of isolation needed.
        """
        ...


@runtime_checkable
class CacheAwarePlugin(Protocol):
    """Trait for plugins that participate in caching.

    Plugins implementing this trait declare what cache keys they
    populate and consume, enabling intelligent cache management.
    """

    @property
    def cache_populates(self) -> tuple[str, ...]:
        """Return cache keys this plugin populates.

        Returns
        -------
        tuple[str, ...]
            Cache keys populated by this plugin.
        """
        ...

    @property
    def cache_consumes(self) -> tuple[str, ...]:
        """Return cache keys this plugin consumes.

        Returns
        -------
        tuple[str, ...]
            Cache keys consumed by this plugin.
        """
        ...


@runtime_checkable
class RetryablePlugin(Protocol):
    """Trait for plugins with custom retry behavior.

    Plugins implementing this trait can specify which exceptions
    are retryable and custom retry parameters.
    """

    @property
    def retryable_exceptions(self) -> tuple[type[Exception], ...]:
        """Return exception types that should trigger retry.

        Returns
        -------
        tuple[type[Exception], ...]
            Exception types that are retryable.
        """
        ...

    @property
    def max_retries(self) -> int:
        """Return maximum retry attempts.

        Returns
        -------
        int
            Maximum number of retries.
        """
        ...

    @property
    def retry_backoff_ms(self) -> int:
        """Return backoff time between retries.

        Returns
        -------
        int
            Backoff time in milliseconds.
        """
        ...


@runtime_checkable
class IncrementalPlugin(Protocol):
    """Trait for plugins that support incremental execution.

    Plugins implementing this trait can determine if they need to
    run based on input changes and can produce partial results.
    """

    def compute_input_hash(self, ctx: PluginExecutionContext) -> str:
        """Compute a hash of the plugin's inputs.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        str
            Hash of inputs for change detection.
        """
        ...

    def is_unchanged(
        self,
        ctx: PluginExecutionContext,
        prior_hash: str | None,
    ) -> bool:
        """Check if inputs have changed since last run.

        Parameters
        ----------
        ctx
            Execution context.
        prior_hash
            Hash from prior execution.

        Returns
        -------
        bool
            True if inputs are unchanged.
        """
        ...


@runtime_checkable
class ProgressReportingPlugin(Protocol):
    """Trait for plugins that report execution progress.

    Plugins implementing this trait can provide progress updates
    during long-running operations.
    """

    def set_progress_callback(
        self,
        callback: Callable[[float, str], None],
    ) -> None:
        """Set a callback for progress reporting.

        Parameters
        ----------
        callback
            Callback receiving progress (0-1) and status message.
        """
        ...


@runtime_checkable
class CatalogAwarePlugin(Protocol):
    """Trait for plugins that require function catalog access.

    Plugins implementing this trait need access to the function
    catalog for symbol resolution.
    """

    @property
    def requires_catalog(self) -> bool:
        """Return whether catalog is required.

        Returns
        -------
        bool
            True if catalog is required for execution.
        """
        ...


@runtime_checkable
class AnalyticsContextAwarePlugin(Protocol):
    """Trait for plugins that require full analytics context.

    Plugins implementing this trait need the complete analytics
    context including graphs, ASTs, and function features.
    """

    @property
    def requires_analytics_context(self) -> bool:
        """Return whether analytics context is required.

        Returns
        -------
        bool
            True if analytics context is required.
        """
        ...


# =============================================================================
# Trait Mixins for Implementation
# =============================================================================


class GraphAwareMixin:
    """Mixin providing graph awareness to plugins."""

    _graph_requirements: tuple[GraphKind, ...] = ()

    def get_graph_requirements(self) -> tuple[GraphKind, ...]:
        """Return required graph types.

        Returns
        -------
        tuple[GraphKind, ...]
            Graph kinds that must be available.
        """
        return self._graph_requirements


class ScopeAwareMixin:
    """Mixin providing scope awareness to plugins."""

    _supported_scopes: tuple[Literal["paths", "modules", "time_window"], ...] = ()

    @property
    def supported_scopes(self) -> tuple[Literal["paths", "modules", "time_window"], ...]:
        """Return supported scope types.

        Returns
        -------
        tuple[Literal["paths", "modules", "time_window"], ...]
            Scope identifiers the plugin supports.
        """
        return self._supported_scopes

    def filter_by_scope(self, scope: AnalyticsScope) -> bool:
        """Check if plugin should execute for scope.

        Parameters
        ----------
        scope
            Scope to evaluate for execution eligibility.

        Returns
        -------
        bool
            True if the plugin should run for the provided scope.
        """
        requested: set[str] = set()
        if scope.paths:
            requested.add("paths")
        if scope.modules:
            requested.add("modules")
        if scope.time_window is not None:
            requested.add("time_window")

        if not requested:
            return True
        if not self._supported_scopes:
            return True

        return requested.issubset(self._supported_scopes)


class CacheAwareMixin:
    """Mixin providing cache awareness to plugins."""

    _cache_populates: tuple[str, ...] = ()
    _cache_consumes: tuple[str, ...] = ()

    @property
    def cache_populates(self) -> tuple[str, ...]:
        """Return cache keys populated by this plugin.

        Returns
        -------
        tuple[str, ...]
            Keys this plugin writes into the cache.
        """
        return self._cache_populates

    @property
    def cache_consumes(self) -> tuple[str, ...]:
        """Return cache keys consumed by this plugin.

        Returns
        -------
        tuple[str, ...]
            Keys this plugin expects to read from the cache.
        """
        return self._cache_consumes


class RetryableMixin:
    """Mixin providing retry behavior to plugins."""

    _retryable_exceptions: tuple[type[Exception], ...] = (
        RuntimeError,
        ValueError,
        OSError,
    )
    _max_retries: int = 3
    _retry_backoff_ms: int = 1000

    @property
    def retryable_exceptions(self) -> tuple[type[Exception], ...]:
        """Return retryable exception types."""
        return self._retryable_exceptions

    @property
    def max_retries(self) -> int:
        """Return maximum retry attempts."""
        return self._max_retries

    @property
    def retry_backoff_ms(self) -> int:
        """Return retry backoff in milliseconds."""
        return self._retry_backoff_ms


# =============================================================================
# Trait Detection Utilities
# =============================================================================


def is_graph_aware(plugin: object) -> bool:
    """Check if a plugin implements GraphAwarePlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin is graph-aware.
    """
    return isinstance(plugin, GraphAwarePlugin)


def is_scope_aware(plugin: object) -> bool:
    """Check if a plugin implements ScopeAwarePlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin is scope-aware.
    """
    return isinstance(plugin, ScopeAwarePlugin)


def is_contract_validated(plugin: object) -> bool:
    """Check if a plugin implements ContractValidatedPlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin has contract validation.
    """
    return isinstance(plugin, ContractValidatedPlugin)


def is_isolated(plugin: object) -> bool:
    """Check if a plugin implements IsolatedPlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin requires isolation.
    """
    return isinstance(plugin, IsolatedPlugin)


def is_incremental(plugin: object) -> bool:
    """Check if a plugin implements IncrementalPlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin supports incremental execution.
    """
    return isinstance(plugin, IncrementalPlugin)


def get_plugin_traits(plugin: object) -> tuple[str, ...]:
    """Return names of all traits implemented by a plugin.

    Parameters
    ----------
    plugin
        Plugin to inspect.

    Returns
    -------
    tuple[str, ...]
        Names of implemented traits.
    """
    checks: tuple[tuple[Callable[[object], bool], str], ...] = (
        (is_graph_aware, "GraphAware"),
        (is_scope_aware, "ScopeAware"),
        (is_contract_validated, "ContractValidated"),
        (is_isolated, "Isolated"),
        (is_incremental, "Incremental"),
        (lambda p: isinstance(p, CacheAwarePlugin), "CacheAware"),
        (lambda p: isinstance(p, RetryablePlugin), "Retryable"),
        (lambda p: isinstance(p, ProgressReportingPlugin), "ProgressReporting"),
        (lambda p: isinstance(p, CatalogAwarePlugin), "CatalogAware"),
        (lambda p: isinstance(p, AnalyticsContextAwarePlugin), "AnalyticsContextAware"),
    )
    return tuple(name for predicate, name in checks if predicate(plugin))


__all__ = [
    "AnalyticsContextAwarePlugin",
    "CacheAwareMixin",
    "CacheAwarePlugin",
    "CatalogAwarePlugin",
    "ContractValidatedPlugin",
    "GraphAwareMixin",
    "GraphAwarePlugin",
    "IncrementalPlugin",
    "IsolatedPlugin",
    "ProgressReportingPlugin",
    "RetryableMixin",
    "RetryablePlugin",
    "ScopeAwareMixin",
    "ScopeAwarePlugin",
    "get_plugin_traits",
    "is_contract_validated",
    "is_graph_aware",
    "is_incremental",
    "is_isolated",
    "is_scope_aware",
]
