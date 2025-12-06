"""Unified graph plugin protocol.

This module provides graph-specific plugin types that extend the unified
plugin infrastructure from codeintel.core.plugins.

The core types are imported and re-exported, while graph-specific types
like `GraphPluginMetadata` extend them with graph-related fields.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypedDict, Unpack, cast, runtime_checkable

from pydantic import BaseModel

from codeintel.core.plugins.decorators.functional import BaseFunctionalPlugin
from codeintel.core.plugins.decorators.meta import BasePluginMetaOptions
from codeintel.core.plugins.decorators.step import make_plugin_instance
from codeintel.core.plugins.types.protocol import (
    PluginCapability,
    PluginInputSpec,
    PluginIsolation,
    PluginKind,
    PluginMetadata,
    PluginOutputSpec,
    PluginResourceHints,
    PluginSeverity,
    PluginStage,
    ValidationResult,
)
from codeintel.core.plugins.types.result import PluginResult, PluginStatus

# Import at runtime for use in type alias (FunctionalGraphPlugin)
from codeintel.graphs.core.context import GraphPluginExecutionContext
from codeintel.graphs.engine import GraphKind

# Graph-specific plugin kinds and stages
GraphPluginKind = Literal["builder", "metric", "validation"]
"""Plugin kinds specific to graph processing."""

GraphPluginStage = Literal[
    "goid",
    "edges",
    "structure",
    "core",
    "cfg",
    "dfg",
    "test",
    "symbol",
    "subsystem",
    "config",
    "stats",
    "validation",
]
"""Plugin stages specific to graph processing."""


@dataclass(frozen=True)
class GraphPluginMetadata(PluginMetadata):
    """Metadata for a graph plugin extending unified PluginMetadata.

    This dataclass adds graph-specific fields for graph building and
    metric computation while inheriting all standard plugin metadata.

    Attributes
    ----------
    produces_graph_kinds
        Typed GraphKind values this plugin builds (for builders).
    requires_graph_kinds
        Typed GraphKind values this plugin needs (for metrics).
    options_model
        Optional Pydantic model for plugin options validation.
    options_default
        Default options value.
    """

    produces_graph_kinds: tuple[GraphKind, ...] = ()
    requires_graph_kinds: tuple[GraphKind, ...] = ()
    options_model: type[BaseModel] | None = None
    options_default: object | None = None

    def __post_init__(self) -> None:
        """Normalize derived flags."""
        # Call parent's __post_init__ for isolation flag normalization
        super().__post_init__()


def create_graph_metadata(
    *,
    name: str,
    description: str,
    kind: GraphPluginKind,
    stage: GraphPluginStage,
    severity: PluginSeverity = "fatal",
    enabled_by_default: bool = True,
    depends_on: tuple[str, ...] = (),
    provides: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    produces_tables: tuple[str, ...] = (),
    produces_graph_kinds: tuple[GraphKind, ...] = (),
    requires_graph_kinds: tuple[GraphKind, ...] = (),
    resource_hints: PluginResourceHints | None = None,
    supports_incremental: bool = False,
    isolation_kind: PluginIsolation = "none",
    options_model: type[BaseModel] | None = None,
    options_default: object | None = None,
    version_hash: str | None = None,
    config_schema_ref: str | None = None,
    row_count_tables: tuple[str, ...] = (),
    cache_populates: tuple[str, ...] = (),
    cache_consumes: tuple[str, ...] = (),
    requires_isolation: bool = False,
    scope_aware: bool = False,
    supported_scopes: tuple[str, ...] = (),
    contract_checkers: tuple[str, ...] = (),
) -> GraphPluginMetadata:
    """Create graph plugin metadata with sensible defaults.

    This factory function provides a convenient way to create metadata
    with typed GraphKind values that automatically populate the string
    produces_graphs and requires_graphs fields.

    Parameters
    ----------
    name
        Unique plugin identifier.
    description
        Human-readable description.
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
    produces_graph_kinds
        GraphKind values this plugin builds (for builders).
    requires_graph_kinds
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

    Returns
    -------
    GraphPluginMetadata
        Graph plugin metadata with all fields populated.
    """
    return GraphPluginMetadata(
        name=name,
        description=description,
        kind=kind,
        stage=stage,
        severity=severity,
        enabled_by_default=enabled_by_default,
        depends_on=depends_on,
        provides=provides,
        requires=requires,
        produces_tables=produces_tables,
        produces_graphs=tuple(str(g) for g in produces_graph_kinds),
        requires_graphs=tuple(str(g) for g in requires_graph_kinds),
        resource_hints=resource_hints,
        supports_incremental=supports_incremental,
        isolation_kind=isolation_kind,
        requires_isolation=requires_isolation,
        scope_aware=scope_aware,
        supported_scopes=supported_scopes,
        version_hash=version_hash,
        config_schema_ref=config_schema_ref,
        row_count_tables=row_count_tables,
        cache_populates=cache_populates,
        cache_consumes=cache_consumes,
        contract_checkers=contract_checkers,
        produces_graph_kinds=produces_graph_kinds,
        requires_graph_kinds=requires_graph_kinds,
        options_model=options_model,
        options_default=options_default,
    )


@runtime_checkable
class GraphPluginProtocol(Protocol):
    """Protocol for graph plugins.

    Graph plugins implement this protocol to be registered and executed
    by the graph runtime. This protocol uses `GraphPluginExecutionContext`
    and `GraphPluginMetadata` for graph-specific functionality.
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

    def execute(self, ctx: GraphPluginExecutionContext) -> PluginResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Graph plugin execution context.

        Returns
        -------
        PluginResult
            Result of plugin execution.
        """
        ...


class GraphPluginMetaOptionsInput(TypedDict, total=False):
    """Typed keyword arguments for GraphPluginMetaOptions.from_kwargs factory."""

    name: str
    description: str
    kind: GraphPluginKind
    stage: GraphPluginStage
    severity: PluginSeverity
    enabled_by_default: bool
    depends_on: tuple[str, ...]
    provides: tuple[str, ...]
    requires: tuple[str, ...]
    produces_tables: tuple[str, ...]
    produces_graph_kinds: tuple[GraphKind, ...]
    requires_graph_kinds: tuple[GraphKind, ...]
    resource_hints: PluginResourceHints | None
    supports_incremental: bool
    isolation_kind: PluginIsolation
    options_model: type[BaseModel] | None
    options_default: object | None
    version_hash: str | None
    config_schema_ref: str | None
    row_count_tables: tuple[str, ...]
    cache_populates: tuple[str, ...]
    cache_consumes: tuple[str, ...]
    requires_isolation: bool
    scope_aware: bool
    supported_scopes: tuple[str, ...]
    contract_checkers: tuple[str, ...]


@dataclass(frozen=True)
class GraphPluginMetaOptions:
    """Options container for graph plugin metadata."""

    name: str | None = None
    description: str | None = None
    kind: GraphPluginKind | None = None
    stage: GraphPluginStage | None = None
    severity: PluginSeverity = "fatal"
    enabled_by_default: bool = True
    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    produces_tables: tuple[str, ...] = ()
    produces_graph_kinds: tuple[GraphKind, ...] = ()
    requires_graph_kinds: tuple[GraphKind, ...] = ()
    resource_hints: PluginResourceHints | None = None
    supports_incremental: bool = False
    isolation_kind: PluginIsolation = "none"
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
    def from_kwargs(**kwargs: Unpack[GraphPluginMetaOptionsInput]) -> GraphPluginMetaOptions:
        """Build options from keyword arguments with validation.

        Delegates validation to BasePluginMetaOptions.validate_option_keys,
        which raises ValueError if unknown keys are provided.

        Parameters
        ----------
        **kwargs
            Keyword arguments matching GraphPluginMetaOptionsInput fields.

        Returns
        -------
        GraphPluginMetaOptions
            Options built from the provided keyword arguments.
        """
        BasePluginMetaOptions.validate_option_keys(
            set(GraphPluginMetaOptionsInput.__annotations__),
            kwargs,
        )
        return GraphPluginMetaOptions(**kwargs)

    def to_metadata(
        self,
        fn: Callable[[GraphPluginExecutionContext], PluginResult],
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
        return create_graph_metadata(
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
            produces_graph_kinds=self.produces_graph_kinds,
            requires_graph_kinds=self.requires_graph_kinds,
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

    Structurally equivalent to ``codeintel.core.plugins.registry.PluginSkip``
    but with graph-specific skip reasons. The core type uses `str` for
    maximum flexibility; this type uses a Literal for domain-specific type safety.

    Attributes
    ----------
    name
        Plugin name.
    reason
        Reason for skipping (graph-specific values).
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


# Type alias for graph functional plugin using the base class
FunctionalGraphPlugin = BaseFunctionalPlugin[GraphPluginExecutionContext, GraphPluginMetadata]
"""Graph plugin implementation wrapping a callable.

This type alias provides graph-specific typing for the base functional
plugin class. Use with the @graph_plugin decorator.
"""


def graph_plugin(
    *,
    meta: GraphPluginMetaOptions | None = None,
    register: bool = True,
    **kwargs: Unpack[GraphPluginMetaOptionsInput],
) -> Callable[[Callable[[GraphPluginExecutionContext], PluginResult]], FunctionalGraphPlugin]:
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
    Callable[[Callable[[GraphPluginExecutionContext], PluginResult]], FunctionalGraphPlugin]
        Decorator that creates a FunctionalGraphPlugin.
    """

    def _get_register_fn() -> Callable[[FunctionalGraphPlugin], None] | None:
        """Get registration function if register is True.

        Returns
        -------
        Callable[[FunctionalGraphPlugin], None] | None
            Registration function or None if register is False.
        """
        if not register:
            return None

        def _register_plugin(plugin_instance: FunctionalGraphPlugin) -> None:
            """Register plugin via registry without importing at decorator scope."""
            registry = importlib.import_module("codeintel.graphs.core.registry")
            register_fn = registry.register_graph_plugin
            register_fn(cast("GraphPluginProtocol", plugin_instance))

        return _register_plugin

    def decorator(
        fn: Callable[[GraphPluginExecutionContext], PluginResult],
    ) -> FunctionalGraphPlugin:
        if meta is not None and kwargs:
            message = "Provide either meta or keyword metadata, not both."
            raise ValueError(message)

        options = meta or GraphPluginMetaOptions.from_kwargs(**kwargs)

        return make_plugin_instance(
            fn=fn,
            options=options,
            plugin_factory=lambda m, f: FunctionalGraphPlugin(_metadata=m, _execute_fn=f),
            to_metadata=lambda opts, f: opts.to_metadata(f),
            register_fn=_get_register_fn(),
        )

    return decorator


# Default plugins for different plugin kinds
# Names must match actual plugin_name values in TargetPlugin implementations
DEFAULT_BUILDER_PLUGINS: tuple[str, ...] = (
    "goid_builder",
    "callgraph",
    "import_graph",
    "cfg_dfg",
    "symbol_uses",
)

DEFAULT_METRIC_PLUGINS: tuple[str, ...] = (
    "graph_metrics.core",
    "graph_metrics.secondary",
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
    "GraphPluginKind",
    "GraphPluginMetaOptions",
    "GraphPluginMetaOptionsInput",
    "GraphPluginMetadata",
    "GraphPluginPlan",
    "GraphPluginProtocol",
    "GraphPluginSkip",
    "GraphPluginStage",
    "PluginCapability",
    "PluginInputSpec",
    "PluginIsolation",
    "PluginKind",
    "PluginMetadata",
    "PluginOutputSpec",
    "PluginResourceHints",
    "PluginResult",
    "PluginSeverity",
    "PluginStage",
    "PluginStatus",
    "ValidationResult",
    "create_graph_metadata",
    "graph_plugin",
]
