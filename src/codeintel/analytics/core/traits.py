"""Plugin traits for capability-based composition.

This module defines protocol classes (traits) that plugins can implement
to declare specific capabilities. The runtime uses these traits to:
- Automatically prepare contexts with required resources
- Validate plugin requirements
- Enable trait-based plugin discovery

Domain-agnostic traits are imported from codeintel.core.plugins.traits.
This module extends with analytics-specific traits like GraphAwarePlugin
and ScopeAwarePlugin.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

# Re-export domain-agnostic traits from core
from codeintel.core.plugins.traits import (
    CacheAwareMixin,
    CacheAwarePlugin,
    IsolatedPlugin,
    ProgressReportingMixin,
    ProgressReportingPlugin,
    RetryableMixin,
    RetryablePlugin,
    is_cache_aware,
    is_isolated,
    is_progress_reporting,
    is_retryable,
)

if TYPE_CHECKING:
    from codeintel.analytics.core.context import PluginExecutionContext
    from codeintel.analytics.graphs.contracts import ContractChecker
    from codeintel.analytics.runtime.manifest import AnalyticsScope
    from codeintel.graphs.engine import GraphKind

from codeintel.analytics.core.providers import get_support_provider

# =============================================================================
# Analytics-Specific Protocols
# =============================================================================


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


# =============================================================================
# Analytics-Specific Composition Mixins
# =============================================================================


class WithRowCounts:
    """Mixin that auto-computes row counts for declared output tables.

    Plugins using this mixin should define `output_tables` as a class attribute
    containing the table names to count.

    Class Attributes
    ----------------
    output_tables : tuple[str, ...]
        Tables to count rows for after execution.

    Example
    -------
    >>> class MyPlugin(BasePlugin, WithRowCounts):
    ...     output_tables = ("analytics.my_table",)
    ...
    ...     def compute(self, ctx):
    ...         # Write to table...
    ...         return None  # Row counts computed automatically
    """

    output_tables: tuple[str, ...] = ()

    def compute_row_counts_for_tables(
        self,
        ctx: PluginExecutionContext,
        tables: tuple[str, ...] | None = None,
    ) -> dict[str, int]:
        """Compute row counts for the specified or declared tables.

        Parameters
        ----------
        ctx
            Execution context with gateway access.
        tables
            Override table list (defaults to self.output_tables).

        Returns
        -------
        dict[str, int]
            Mapping of table names to row counts.
        """
        target_tables = tables or self.output_tables
        if not target_tables or ctx.snapshot is None:
            return {}

        provider = get_support_provider()
        return provider.compute_row_counts(ctx.gateway, ctx.snapshot, target_tables)


class WithContractValidation:
    """Mixin that runs output contracts after successful execution.

    Plugins using this mixin declare contracts that are validated after
    execution completes successfully.

    Class Attributes
    ----------------
    validate_contracts : bool
        Whether to run contract validation (default True).

    Properties
    ----------
    output_contracts : tuple[OutputContractSpec, ...]
        Override to provide explicit contracts.

    Example
    -------
    >>> class MyPlugin(BasePlugin, WithContractValidation):
    ...     @property
    ...     def output_contracts(self):
    ...         return (OutputContractSpec(table="analytics.my_table", min_rows=1),)
    """

    validate_contracts: bool = True

    @property
    def output_contracts(self) -> tuple[object, ...]:
        """Return output contracts for validation.

        Override in subclasses to provide specific contracts.

        Returns
        -------
        tuple[OutputContractSpec, ...]
            Contracts to validate after execution.
        """
        return ()

    def run_contract_validation(
        self,
        ctx: PluginExecutionContext,
    ) -> tuple[bool, tuple[str, ...]]:
        """Run contract validation for this plugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        tuple[bool, tuple[str, ...]]
            Success flag and list of error messages.
        """
        if not self.validate_contracts:
            return True, ()

        snapshot = ctx.snapshot
        if snapshot is None:
            return True, ()

        contracts = self.output_contracts
        if not contracts:
            return True, ()

        provider = get_support_provider()
        valid, errors = provider.validate_contracts(ctx.gateway, contracts, snapshot)
        return valid, errors


class WithCaching:
    """Mixin for plugins that cache intermediate results in scratch store.

    Enables plugins to store and retrieve intermediate results across
    plugin executions within the same run.

    Class Attributes
    ----------------
    scratch_key : str
        Key for storing results in scratch (default: plugin class name).

    Example
    -------
    >>> class MyPlugin(BasePlugin, WithCaching):
    ...     scratch_key = "my_plugin_data"
    ...
    ...     def compute(self, ctx):
    ...         # Check if data is cached
    ...         cached = self.get_cached(ctx)
    ...         if cached is not None:
    ...             return cached
    ...
    ...         # Compute and cache
    ...         result = expensive_computation()
    ...         self.cache_result(ctx, result)
    ...         return result
    """

    scratch_key: str = ""

    def _get_scratch_key(self) -> str:
        """Return the scratch key for this plugin.

        Returns
        -------
        str
            Key for scratch store access.
        """
        return self.scratch_key or self.__class__.__name__

    def get_cached[T](self, ctx: PluginExecutionContext, default: T | None = None) -> T | None:
        """Retrieve cached result from scratch store.

        Parameters
        ----------
        ctx
            Execution context with scratch store.
        default
            Value to return if not cached.

        Returns
        -------
        T | None
            Cached value or default.
        """
        return ctx.scratch.consume(self._get_scratch_key(), default)

    def cache_result(self, ctx: PluginExecutionContext, value: object) -> None:
        """Store a result in the scratch store.

        Parameters
        ----------
        ctx
            Execution context with scratch store.
        value
            Value to cache.
        """
        ctx.scratch.declare(self._get_scratch_key(), value)

    def has_cached(self, ctx: PluginExecutionContext) -> bool:
        """Check if a cached result exists.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        bool
            True if cached result exists.
        """
        return ctx.scratch.has(self._get_scratch_key())


class WithDependencyData:
    """Mixin for plugins that consume data from dependent plugins.

    Enables type-safe access to data populated by upstream plugins.

    Example
    -------
    >>> class ConsumerPlugin(BasePlugin, WithDependencyData):
    ...     def compute(self, ctx):
    ...         # Get data from upstream plugin
    ...         metrics = self.get_dependency_data(ctx, "function_metrics")
    ...         if metrics is None:
    ...             return PluginResult.fail("Missing function metrics")
    ...         # Use metrics...
    """

    @staticmethod
    def get_dependency_data[T](
        ctx: PluginExecutionContext,
        key: str,
        default: T | None = None,
    ) -> T | None:
        """Retrieve data populated by a dependent plugin.

        Parameters
        ----------
        ctx
            Execution context.
        key
            Key used by the upstream plugin.
        default
            Default value if not found.

        Returns
        -------
        T | None
            Data from upstream plugin or default.
        """
        return ctx.scratch.consume(key, default)

    @staticmethod
    def set_dependency_data(
        ctx: PluginExecutionContext,
        key: str,
        value: object,
    ) -> None:
        """Store data for downstream plugins.

        Parameters
        ----------
        ctx
            Execution context.
        key
            Key for downstream access.
        value
            Data to store.
        """
        ctx.scratch.declare(key, value)


class WithProgressReporting(ProgressReportingMixin):
    """Mixin for plugins that report execution progress.

    Extends the core ProgressReportingMixin with analytics-specific convenience.

    Example
    -------
    >>> class MyPlugin(BasePlugin, WithProgressReporting):
    ...     def compute(self, ctx):
    ...         for i, item in enumerate(items):
    ...             self.report_progress(i / len(items), f"Processing {item}")
    ...             process(item)
    """


class WithCleanup:
    """Mixin for plugins that need cleanup after execution.

    Enables plugins to register cleanup callbacks that run after the
    entire plugin execution batch completes.

    Example
    -------
    >>> class MyPlugin(BasePlugin, WithCleanup):
    ...     def compute(self, ctx):
    ...         temp_file = create_temp_file()
    ...         self.register_cleanup(ctx, lambda: temp_file.unlink())
    ...         # Use temp_file...
    """

    @staticmethod
    def register_cleanup(
        ctx: PluginExecutionContext,
        callback: Callable[[], None],
    ) -> None:
        """Register a cleanup callback.

        Parameters
        ----------
        ctx
            Execution context.
        callback
            Cleanup function to call after run completes.
        """
        ctx.scratch.register_cleanup(callback)


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
    )
    return tuple(name for predicate, name in checks if predicate(plugin))


__all__ = [
    # Re-exported from core (domain-agnostic)
    "CacheAwareMixin",
    "CacheAwarePlugin",
    # Analytics-specific protocols
    "CatalogAwarePlugin",
    "ContractValidatedPlugin",
    # Analytics-specific mixins
    "GraphAwareMixin",
    "GraphAwarePlugin",
    "IncrementalPlugin",
    "IsolatedPlugin",
    "ProgressReportingMixin",
    "ProgressReportingPlugin",
    "RetryableMixin",
    "RetryablePlugin",
    "ScopeAwareMixin",
    "ScopeAwarePlugin",
    # Composition mixins
    "WithCaching",
    "WithCleanup",
    "WithContractValidation",
    "WithDependencyData",
    "WithProgressReporting",
    "WithRowCounts",
    # Detection utilities
    "get_plugin_traits",
    "is_cache_aware",
    "is_contract_validated",
    "is_graph_aware",
    "is_incremental",
    "is_isolated",
    "is_progress_reporting",
    "is_retryable",
    "is_scope_aware",
]
