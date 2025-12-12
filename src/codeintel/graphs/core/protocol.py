"""Unified graph plugin protocol.

This module provides graph-specific plugin types that extend the unified
plugin infrastructure from codeintel.core.plugins.

The core types are imported and re-exported, while graph-specific types
like `GraphPluginMetadata` extend them with graph-related fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

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

if TYPE_CHECKING:
    from pydantic import BaseModel

    from codeintel.graphs.core.context import GraphPluginExecutionContext
    from codeintel.graphs.engine import GraphKind


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
        super().__post_init__()


@dataclass(frozen=True)
class GraphPluginMetadataConfig:
    """Bundle optional fields used when constructing graph plugin metadata."""

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


def _validate_metadata_config(
    kind: GraphPluginKind,
    stage: GraphPluginStage,
    config: GraphPluginMetadataConfig,
) -> tuple[tuple[str, ...], str | None]:
    """Validate and normalize metadata configuration.

    Returns
    -------
    tuple[tuple[str, ...], str | None]
        Supported scopes if scope awareness is enabled and no errors, along
        with an optional validation error message.
    """
    errors: list[str] = []

    if config.scope_aware and not config.supported_scopes:
        errors.append("Scope-aware plugins must declare supported_scopes.")

    if kind == "builder" and stage != "goid" and not config.produces_graph_kinds:
        errors.append("Builder plugins must declare produces_graph_kinds.")

    if kind == "metric" and not config.requires_graph_kinds:
        errors.append("Metric plugins must declare requires_graph_kinds.")

    if kind == "validation" and config.produces_graph_kinds:
        errors.append("Validation plugins must not declare produces_graph_kinds.")

    if errors:
        return (), "; ".join(errors)

    if not config.scope_aware:
        return (), None
    return config.supported_scopes, None


def create_graph_metadata(
    *,
    name: str,
    description: str,
    kind: GraphPluginKind,
    stage: GraphPluginStage,
    config: GraphPluginMetadataConfig | None = None,
) -> GraphPluginMetadata:
    """Create graph plugin metadata with sensible defaults.

    This factory function provides a convenient way to create metadata
    with typed GraphKind values that automatically populate the string
    produces_graphs and requires_graphs fields while keeping the
    function signature compact via GraphPluginMetadataConfig.

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
    config
        Optional configuration bundle for advanced metadata fields such
        as cache hints, resource hints, and graph kind declarations.

    Raises
    ------
    ValueError
        If the configuration is inconsistent with the plugin kind or
        scope settings.

    Returns
    -------
    GraphPluginMetadata
        Graph plugin metadata with all fields populated.
    """
    metadata_config = config or GraphPluginMetadataConfig()
    supported_scopes, validation_error = _validate_metadata_config(
        kind,
        stage,
        metadata_config,
    )
    if validation_error is not None:
        raise ValueError(validation_error)

    return GraphPluginMetadata(
        name=name,
        description=description,
        kind=kind,
        stage=stage,
        severity=metadata_config.severity,
        enabled_by_default=metadata_config.enabled_by_default,
        depends_on=metadata_config.depends_on,
        provides=metadata_config.provides,
        requires=metadata_config.requires,
        produces_tables=metadata_config.produces_tables,
        produces_graphs=tuple(str(graph) for graph in metadata_config.produces_graph_kinds),
        requires_graphs=tuple(str(graph) for graph in metadata_config.requires_graph_kinds),
        resource_hints=metadata_config.resource_hints,
        supports_incremental=metadata_config.supports_incremental,
        isolation_kind=metadata_config.isolation_kind,
        requires_isolation=metadata_config.requires_isolation,
        scope_aware=metadata_config.scope_aware,
        supported_scopes=supported_scopes,
        version_hash=metadata_config.version_hash,
        config_schema_ref=metadata_config.config_schema_ref,
        row_count_tables=metadata_config.row_count_tables,
        cache_populates=metadata_config.cache_populates,
        cache_consumes=metadata_config.cache_consumes,
        contract_checkers=metadata_config.contract_checkers,
        produces_graph_kinds=metadata_config.produces_graph_kinds,
        requires_graph_kinds=metadata_config.requires_graph_kinds,
        options_model=metadata_config.options_model,
        options_default=metadata_config.options_default,
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
    "GraphPluginKind",
    "GraphPluginMetadata",
    "GraphPluginMetadataConfig",
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
]
