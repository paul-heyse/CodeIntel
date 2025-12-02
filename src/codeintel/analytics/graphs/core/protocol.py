"""Modernized graph plugin protocol.

This module defines the protocol and types for graph plugins, providing
a modernized interface aligned with the new analytics plugin architecture
while preserving graph-specific functionality.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from codeintel.analytics.context import AnalyticsContext
    from codeintel.analytics.graph_runtime import GraphRuntime
    from codeintel.analytics.graphs.contracts import ContractChecker
    from codeintel.config.steps_graphs import GraphRunScope
    from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
    from codeintel.storage.gateway import StorageGateway

GraphPluginStage = Literal[
    "core",
    "cfg",
    "dfg",
    "test",
    "symbol",
    "subsystem",
    "config",
    "stats",
]


@dataclass
class GraphRuntimeScratch:
    """Ephemeral scratch/cache store shared across plugin executions in a run.

    Provides a way for plugins to share intermediate data within a single
    execution run without persisting to the database.
    """

    _store: dict[str, object] = field(default_factory=dict)
    _cleanup: list[Callable[[], None]] = field(default_factory=list)

    def declare(self, key: str, value: object) -> None:
        """Record a value for later consumption.

        Parameters
        ----------
        key
            Identifier for the stored value.
        value
            Value to store.
        """
        self._store[key] = value

    def consume(self, key: str, default: object | None = None) -> object | None:
        """Retrieve a value populated by another plugin.

        Parameters
        ----------
        key
            Identifier of the value to retrieve.
        default
            Value to return if key is not found.

        Returns
        -------
        object | None
            Cached value or provided default.
        """
        return self._store.get(key, default)

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback executed after the run completes.

        Parameters
        ----------
        callback
            Function to call during cleanup.
        """
        self._cleanup.append(callback)

    def cleanup(self) -> None:
        """Execute cleanup callbacks and clear stored values."""
        import logging  # noqa: PLC0415

        log = logging.getLogger(__name__)
        for callback in reversed(self._cleanup):
            try:
                callback()
            except (RuntimeError, OSError, ValueError):
                log.exception("scratch.cleanup_failed")
        self._store.clear()
        self._cleanup.clear()

    def __len__(self) -> int:
        """Return the number of declared cache entries.

        Returns
        -------
        int
            Count of cached entries.
        """
        return len(self._store)

    def keys(self) -> tuple[str, ...]:
        """Return declared cache keys.

        Returns
        -------
        tuple[str, ...]
            Cache key names.
        """
        return tuple(self._store.keys())


@dataclass(frozen=True)
class GraphPluginContext:
    """Execution context for graph plugins.

    Provides access to storage, graph runtime, and shared scratch space.
    """

    gateway: StorageGateway
    runtime: GraphRuntime
    repo: str
    commit: str
    analytics_context: AnalyticsContext | None = None
    catalog_provider: FunctionCatalogProvider | None = None
    options: object | None = None
    plugin_name: str | None = None
    run_id: str | None = None
    scope: GraphRunScope | None = None
    scratch: GraphRuntimeScratch = field(default_factory=GraphRuntimeScratch)


@dataclass(frozen=True)
class GraphPluginResult:
    """Result returned by graph plugin execution.

    Attributes
    ----------
    success
        Whether execution completed successfully.
    row_counts
        Mapping of table names to row counts written.
    input_hash
        Hash of inputs for caching.
    options_hash
        Hash of options for caching.
    error
        Error message if execution failed.
    """

    success: bool = True
    row_counts: dict[str, int] | None = None
    input_hash: str | None = None
    options_hash: str | None = None
    error: str | None = None

    @staticmethod
    def ok(
        *,
        row_counts: dict[str, int] | None = None,
        input_hash: str | None = None,
        options_hash: str | None = None,
    ) -> GraphPluginResult:
        """Create a successful result.

        Parameters
        ----------
        row_counts
            Optional mapping of table names to row counts written.
        input_hash
            Optional hash of inputs.
        options_hash
            Optional hash of options.

        Returns
        -------
        GraphPluginResult
            Result object marked as successful.
        """
        return GraphPluginResult(
            success=True,
            row_counts=row_counts,
            input_hash=input_hash,
            options_hash=options_hash,
        )

    @staticmethod
    def fail(error: str) -> GraphPluginResult:
        """Create a failed result.

        Parameters
        ----------
        error
            Error message describing the failure.

        Returns
        -------
        GraphPluginResult
            Result object marked as failed.
        """
        return GraphPluginResult(success=False, error=error)


@dataclass(frozen=True)
class GraphMetricResourceHints:
    """Optional resource hints used for planning/observability.

    Attributes
    ----------
    max_runtime_ms
        Maximum expected runtime in milliseconds.
    memory_mb_hint
        Expected memory usage in megabytes.
    """

    max_runtime_ms: int | None = None
    memory_mb_hint: int | None = None


@dataclass(frozen=True)
class GraphMetricPluginMetadata:
    """Metadata for a graph metric plugin.

    Captures all declarative information about a graph plugin for
    introspection, documentation, and planning.
    """

    name: str
    description: str
    stage: GraphPluginStage
    severity: Literal["fatal", "soft_fail", "skip_on_error"]
    enabled_by_default: bool
    depends_on: tuple[str, ...]
    provides: tuple[str, ...]
    requires: tuple[str, ...]
    resource_hints: GraphMetricResourceHints | None
    options_model: type[BaseModel] | None
    options_default: object | None
    version_hash: str | None
    contract_checkers: tuple[ContractChecker, ...]
    scope_aware: bool
    supported_scopes: tuple[Literal["paths", "modules", "time_window"], ...]
    requires_isolation: bool
    isolation_kind: Literal["process", "thread"] | None
    config_schema_ref: str | None
    row_count_tables: tuple[str, ...] = ()
    cache_populates: tuple[str, ...] = ()
    cache_consumes: tuple[str, ...] = ()


@runtime_checkable
class GraphPluginProtocol(Protocol):
    """Protocol for graph metric plugins.

    Graph plugins implement this protocol to be registered and executed
    by the graph runtime.
    """

    @property
    def metadata(self) -> GraphMetricPluginMetadata:
        """Return plugin metadata."""
        ...

    def execute(self, ctx: GraphPluginContext) -> GraphPluginResult:
        """Execute the plugin."""
        ...


@dataclass(frozen=True)
class GraphMetricPluginSkip:
    """Skip metadata for planned plugins that will not execute.

    Attributes
    ----------
    name
        Plugin name.
    reason
        Reason for skipping.
    """

    name: str
    reason: Literal["disabled", "missing_dependency", "config_error"]


@dataclass(frozen=True)
class GraphMetricPluginPlan:
    """Resolved execution plan for graph metric plugins.

    Attributes
    ----------
    plugins
        Ordered plugins to execute.
    plan_id
        Unique identifier for this plan.
    skipped_plugins
        Plugins that were skipped.
    dep_graph
        Dependency graph.
    """

    plugins: tuple[GraphPluginProtocol, ...]
    plan_id: str
    skipped_plugins: tuple[GraphMetricPluginSkip, ...] = ()
    dep_graph: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def ordered_names(self) -> tuple[str, ...]:
        """Return plugin names in execution order.

        Returns
        -------
        tuple[str, ...]
            Plugin names in execution order.
        """
        return tuple(plugin.metadata.name for plugin in self.plugins)


@dataclass
class FunctionalGraphPlugin:
    """Plugin implementation wrapping a callable.

    Provides a simple way to create graph plugins from functions.
    """

    _metadata: GraphMetricPluginMetadata
    _execute_fn: Callable[[GraphPluginContext], GraphPluginResult]

    @property
    def metadata(self) -> GraphMetricPluginMetadata:
        """Return plugin metadata."""
        return self._metadata

    def execute(self, ctx: GraphPluginContext) -> GraphPluginResult:
        """Execute the wrapped function.

        Parameters
        ----------
        ctx
            Graph plugin execution context.

        Returns
        -------
        GraphPluginResult
            Result produced by the underlying callable.
        """
        return self._execute_fn(ctx)


def graph_plugin(  # noqa: PLR0913 - decorator with many params by design
    *,
    name: str,
    description: str,
    stage: GraphPluginStage,
    enabled_by_default: bool = True,
    severity: Literal["fatal", "soft_fail", "skip_on_error"] = "fatal",
    depends_on: tuple[str, ...] = (),
    provides: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    resource_hints: GraphMetricResourceHints | None = None,
    options_model: type[BaseModel] | None = None,
    options_default: object | None = None,
    version_hash: str | None = None,
    contract_checkers: tuple[ContractChecker, ...] = (),
    scope_aware: bool = False,
    supported_scopes: tuple[Literal["paths", "modules", "time_window"], ...] = (),
    requires_isolation: bool = False,
    isolation_kind: Literal["process", "thread"] | None = None,
    config_schema_ref: str | None = None,
    row_count_tables: tuple[str, ...] = (),
    cache_populates: tuple[str, ...] = (),
    cache_consumes: tuple[str, ...] = (),
    register: bool = True,
) -> Callable[[Callable[[GraphPluginContext], GraphPluginResult]], FunctionalGraphPlugin]:
    """Decorate a function as a graph plugin.

    Parameters
    ----------
    name
        Plugin name.
    description
        Human-readable description.
    stage
        Processing stage.
    enabled_by_default
        Whether enabled when no explicit list is provided.
    severity
        How failures should be handled.
    depends_on
        Explicit plugin dependencies.
    provides
        Capabilities provided.
    requires
        Capabilities required.
    resource_hints
        Runtime hints.
    options_model
        Pydantic model for options validation.
    options_default
        Default options value.
    version_hash
        Version hash for caching.
    contract_checkers
        Contract validators.
    scope_aware
        Whether plugin supports scoped execution.
    supported_scopes
        Supported scope types.
    requires_isolation
        Whether process/thread isolation is needed.
    isolation_kind
        Type of isolation.
    config_schema_ref
        Reference to config schema.
    row_count_tables
        Tables to count rows from.
    cache_populates
        Cache keys this plugin populates.
    cache_consumes
        Cache keys this plugin consumes.
    register
        Whether to auto-register with global registry.

    Returns
    -------
    Callable
        Decorator that creates a FunctionalGraphPlugin.
    """

    def decorator(
        fn: Callable[[GraphPluginContext], GraphPluginResult],
    ) -> FunctionalGraphPlugin:
        meta = GraphMetricPluginMetadata(
            name=name,
            description=description,
            stage=stage,
            severity=severity,
            enabled_by_default=enabled_by_default,
            depends_on=depends_on,
            provides=provides,
            requires=requires,
            resource_hints=resource_hints,
            options_model=options_model,
            options_default=options_default,
            version_hash=version_hash,
            contract_checkers=contract_checkers,
            scope_aware=scope_aware,
            supported_scopes=supported_scopes,
            requires_isolation=requires_isolation,
            isolation_kind=isolation_kind,
            config_schema_ref=config_schema_ref,
            row_count_tables=row_count_tables,
            cache_populates=cache_populates,
            cache_consumes=cache_consumes,
        )

        plugin_instance = FunctionalGraphPlugin(_metadata=meta, _execute_fn=fn)

        if register:
            from codeintel.analytics.graphs.core.registry import (  # noqa: PLC0415
                register_graph_plugin,
            )

            register_graph_plugin(plugin_instance)

        return plugin_instance

    return decorator


DEFAULT_GRAPH_METRIC_PLUGINS: tuple[str, ...] = (
    "core_graph_metrics",
    "graph_metrics_functions_ext",
    "graph_metrics_modules_ext",
    "test_graph_metrics",
    "cfg_metrics",
    "dfg_metrics",
    "symbol_graph_metrics_modules",
    "symbol_graph_metrics_functions",
    "config_graph_metrics",
    "subsystem_graph_metrics",
    "subsystem_agreement",
    "graph_stats",
)


__all__ = [
    "DEFAULT_GRAPH_METRIC_PLUGINS",
    "FunctionalGraphPlugin",
    "GraphMetricPluginMetadata",
    "GraphMetricPluginPlan",
    "GraphMetricPluginSkip",
    "GraphMetricResourceHints",
    "GraphPluginContext",
    "GraphPluginProtocol",
    "GraphPluginResult",
    "GraphPluginStage",
    "GraphRuntimeScratch",
    "graph_plugin",
]
