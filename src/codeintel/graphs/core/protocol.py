"""Unified graph plugin protocol.

This module defines the protocol and metadata types for graph plugins,
providing a unified interface for both graph builders and graph metric
plugins without any dependency on the analytics subsystem.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel

from codeintel.graphs.core.context import GraphExecutionContext
from codeintel.graphs.core.result import GraphPluginResult
from codeintel.graphs.engine import GraphKind

GraphPluginKind = Literal["builder", "metric", "validation"]

GraphPluginStage = Literal[
    # Builder stages
    "goid",
    "edges",
    "structure",
    # Metric stages
    "core",
    "cfg",
    "dfg",
    "test",
    "symbol",
    "subsystem",
    "config",
    "stats",
    # Validation stage
    "validation",
]

GraphPluginSeverity = Literal["fatal", "soft_fail", "skip_on_error"]

GraphPluginIsolation = Literal["process", "thread", "none"]


@dataclass(frozen=True)
class GraphPluginResourceHints:
    """Optional resource hints used for planning and observability.

    Attributes
    ----------
    max_runtime_ms
        Maximum expected runtime in milliseconds.
    memory_mb_hint
        Expected memory usage in megabytes.
    cpu_intensive
        Whether this plugin is CPU-bound.
    io_intensive
        Whether this plugin is I/O-bound.
    """

    max_runtime_ms: int | None = None
    memory_mb_hint: int | None = None
    cpu_intensive: bool = False
    io_intensive: bool = False


@dataclass(frozen=True)
class GraphPluginMetadata:
    """Metadata for a graph plugin.

    Captures all declarative information about a graph plugin for
    introspection, documentation, dependency resolution, and planning.

    Attributes
    ----------
    name
        Unique plugin identifier (e.g., "callgraph_builder").
    description
        Human-readable description of what the plugin does.
    kind
        Plugin kind: builder, metric, or validation.
    stage
        Processing stage in the graph pipeline.
    severity
        How failures should be handled.
    enabled_by_default
        Whether enabled when no explicit list is provided.
    depends_on
        Explicit plugin dependencies that must run first.
    provides
        Capabilities or artifacts this plugin produces.
    requires
        Capabilities required from other plugins.
    produces_tables
        DuckDB table keys populated by this plugin.
    produces_graphs
        GraphKind values this plugin builds (for builders).
    requires_graphs
        GraphKind values this plugin needs (for metrics).
    resource_hints
        Runtime resource hints for planning.
    supports_incremental
        Whether incremental execution is supported.
    isolation_kind
        Type of isolation needed for execution.
    options_model
        Optional Pydantic model for plugin options validation.
    options_default
        Default options value.
    version_hash
        Version hash for cache invalidation.
    config_schema_ref
        Reference to configuration schema.
    row_count_tables
        Tables to report row counts from.
    cache_populates
        Cache keys this plugin populates.
    cache_consumes
        Cache keys this plugin consumes.
    requires_isolation
        Whether the plugin needs process/thread isolation.
    scope_aware
        Whether the plugin is scope-aware.
    supported_scopes
        Scopes supported when scope-aware.
    contract_checkers
        Contract checker identifiers used by the plugin.
    """

    name: str
    description: str
    kind: GraphPluginKind
    stage: GraphPluginStage
    severity: GraphPluginSeverity = "fatal"
    enabled_by_default: bool = True
    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    produces_tables: tuple[str, ...] = ()
    produces_graphs: tuple[GraphKind, ...] = ()
    requires_graphs: tuple[GraphKind, ...] = ()
    resource_hints: GraphPluginResourceHints | None = None
    supports_incremental: bool = False
    isolation_kind: GraphPluginIsolation = "none"
    options_model: type[BaseModel] | None = None
    options_default: object | None = None
    version_hash: str | None = None
    config_schema_ref: str | None = None
    row_count_tables: tuple[str, ...] = ()
    cache_populates: tuple[str, ...] = ()
    cache_consumes: tuple[str, ...] = ()
    requires_isolation: bool = False
    scope_aware: bool = False
    supported_scopes: tuple[str, ...] = ()
    contract_checkers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize derived flags."""
        if self.requires_isolation or self.isolation_kind != "none":
            object.__setattr__(
                self,
                "requires_isolation",
                self.requires_isolation or self.isolation_kind != "none",
            )


@runtime_checkable
class GraphPluginProtocol(Protocol):
    """Protocol for graph plugins.

    Graph plugins implement this protocol to be registered and executed
    by the graph runtime. This unified protocol supports builders, metrics,
    and validation plugins.
    """

    @property
    def metadata(self) -> GraphPluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        GraphPluginMetadata
            Metadata describing the plugin.
        """
        ...

    def execute(self, ctx: GraphExecutionContext) -> GraphPluginResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Graph plugin execution context.

        Returns
        -------
        GraphPluginResult
            Result of plugin execution.
        """
        ...


@dataclass(frozen=True)
class GraphPluginMetaOptions:
    """Options container for graph plugin metadata."""

    name: str | None = None
    description: str | None = None
    kind: GraphPluginKind | None = None
    stage: GraphPluginStage | None = None
    severity: GraphPluginSeverity = "fatal"
    enabled_by_default: bool = True
    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    produces_tables: tuple[str, ...] = ()
    produces_graphs: tuple[GraphKind, ...] = ()
    requires_graphs: tuple[GraphKind, ...] = ()
    resource_hints: GraphPluginResourceHints | None = None
    supports_incremental: bool = False
    isolation_kind: GraphPluginIsolation = "none"
    options_model: type[BaseModel] | None = None
    options_default: object | None = None
    version_hash: str | None = None
    config_schema_ref: str | None = None
    row_count_tables: tuple[str, ...] = ()
    cache_populates: tuple[str, ...] = ()
    cache_consumes: tuple[str, ...] = ()
    requires_isolation: bool = False
    scope_aware: bool = False
    supported_scopes: tuple[str, ...] = ()
    contract_checkers: tuple[str, ...] = ()

    @staticmethod
    def from_kwargs(**kwargs: object) -> GraphPluginMetaOptions:
        """Build options from legacy kwargs.

        Returns
        -------
        GraphPluginMetaOptions
            Options instance built from provided kwargs.

        Raises
        ------
        ValueError
            If unsupported option keys are provided.
        """
        allowed_keys = {
            "name",
            "description",
            "kind",
            "stage",
            "severity",
            "enabled_by_default",
            "depends_on",
            "provides",
            "requires",
            "produces_tables",
            "produces_graphs",
            "requires_graphs",
            "resource_hints",
            "supports_incremental",
            "isolation_kind",
            "options_model",
            "options_default",
            "version_hash",
            "config_schema_ref",
            "row_count_tables",
            "cache_populates",
            "cache_consumes",
            "requires_isolation",
            "scope_aware",
            "supported_scopes",
            "contract_checkers",
        }
        unknown = set(kwargs) - allowed_keys
        if unknown:
            message = f"Unsupported graph plugin option keys: {', '.join(sorted(unknown))}"
            raise ValueError(message)
        return GraphPluginMetaOptions(**kwargs)  # type: ignore[arg-type]

    def to_metadata(
        self,
        fn: Callable[[GraphExecutionContext], GraphPluginResult],
    ) -> GraphPluginMetadata:
        """Convert options to GraphPluginMetadata using function defaults.

        Parameters
        ----------
        fn
            Plugin callable used for deriving defaults (name/docstring).

        Returns
        -------
        GraphPluginMetadata
            Populated metadata instance.

        Raises
        ------
        ValueError
            If required fields (kind/stage) are missing.
        """
        resolved_name = self.name or fn.__name__
        if self.kind is None or self.stage is None:
            message = "Graph plugin kind and stage must be specified."
            raise ValueError(message)
        return GraphPluginMetadata(
            name=resolved_name,
            description=(self.description or fn.__doc__ or "").strip(),
            kind=self.kind,
            stage=self.stage,
            severity=self.severity,
            enabled_by_default=self.enabled_by_default,
            depends_on=self.depends_on,
            provides=self.provides,
            requires=self.requires,
            produces_tables=self.produces_tables,
            produces_graphs=self.produces_graphs,
            requires_graphs=self.requires_graphs,
            resource_hints=self.resource_hints,
            supports_incremental=self.supports_incremental,
            isolation_kind=self.isolation_kind,
            options_model=self.options_model,
            options_default=self.options_default,
            version_hash=self.version_hash,
            config_schema_ref=self.config_schema_ref,
            row_count_tables=self.row_count_tables,
            cache_populates=self.cache_populates,
            cache_consumes=self.cache_consumes,
            requires_isolation=self.requires_isolation,
            scope_aware=self.scope_aware,
            supported_scopes=self.supported_scopes,
            contract_checkers=self.contract_checkers,
        )


@dataclass(frozen=True)
class GraphPluginSkip:
    """Skip metadata for planned plugins that will not execute.

    Attributes
    ----------
    name
        Plugin name.
    reason
        Reason for skipping.
    """

    name: str
    reason: Literal[
        "disabled",
        "missing_dependency",
        "missing_graph",
        "config_error",
        "incremental_skip",
        "unchanged",
    ]


@dataclass(frozen=True)
class GraphPluginPlan:
    """Resolved execution plan for graph plugins.

    Attributes
    ----------
    plugins
        Ordered plugins to execute.
    plan_id
        Unique identifier for this plan.
    skipped_plugins
        Plugins that were skipped during planning.
    dep_graph
        Dependency graph mapping plugin names to dependencies.
    """

    plugins: tuple[GraphPluginProtocol, ...]
    plan_id: str
    skipped_plugins: tuple[GraphPluginSkip, ...] = ()
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

    _metadata: GraphPluginMetadata
    _execute_fn: Callable[[GraphExecutionContext], GraphPluginResult]

    @property
    def metadata(self) -> GraphPluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        GraphPluginMetadata
            Metadata describing the plugin.
        """
        return self._metadata

    def execute(self, ctx: GraphExecutionContext) -> GraphPluginResult:
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


def graph_plugin(
    *,
    meta: GraphPluginMetaOptions | None = None,
    register: bool = True,
    **kwargs: object,
) -> Callable[[Callable[[GraphExecutionContext], GraphPluginResult]], FunctionalGraphPlugin]:
    """Decorate a function as a graph plugin.

    Parameters
    ----------
    meta
        Graph plugin metadata/options container.
    register
        Whether to auto-register with global registry.
    **kwargs
        Legacy metadata fields (name, kind, stage, etc.); prefer `meta`.

    Returns
    -------
    Callable
        Decorator that creates a FunctionalGraphPlugin.
    """

    def decorator(
        fn: Callable[[GraphExecutionContext], GraphPluginResult],
    ) -> FunctionalGraphPlugin:
        if meta is not None and kwargs:
            message = "Provide either meta or keyword metadata, not both."
            raise ValueError(message)

        options = meta or GraphPluginMetaOptions.from_kwargs(**kwargs)
        metadata = options.to_metadata(fn)

        plugin_instance = FunctionalGraphPlugin(
            _metadata=metadata,
            _execute_fn=fn,
        )

        if register:
            from codeintel.graphs.core.registry import (  # noqa: PLC0415
                register_graph_plugin,
            )

            register_graph_plugin(cast("GraphPluginProtocol", plugin_instance))

        return plugin_instance

    return decorator


# Default plugins for different plugin kinds
DEFAULT_BUILDER_PLUGINS: tuple[str, ...] = (
    "goid_builder",
    "callgraph_builder",
    "import_graph_builder",
    "cfg_dfg_builder",
    "symbol_uses_builder",
)

DEFAULT_METRIC_PLUGINS: tuple[str, ...] = (
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

DEFAULT_VALIDATION_PLUGINS: tuple[str, ...] = ("graph_validation",)

DEFAULT_GRAPH_PLUGINS: tuple[str, ...] = (
    *DEFAULT_BUILDER_PLUGINS,
    *DEFAULT_METRIC_PLUGINS,
    *DEFAULT_VALIDATION_PLUGINS,
)


__all__ = [
    "DEFAULT_BUILDER_PLUGINS",
    "DEFAULT_GRAPH_PLUGINS",
    "DEFAULT_METRIC_PLUGINS",
    "DEFAULT_VALIDATION_PLUGINS",
    "FunctionalGraphPlugin",
    "GraphPluginIsolation",
    "GraphPluginKind",
    "GraphPluginMetaOptions",
    "GraphPluginMetadata",
    "GraphPluginPlan",
    "GraphPluginProtocol",
    "GraphPluginResourceHints",
    "GraphPluginSeverity",
    "GraphPluginSkip",
    "GraphPluginStage",
    "graph_plugin",
]
