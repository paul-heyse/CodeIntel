"""Unified plugin protocol for graphs and analytics.

This module defines the core protocol and types for all plugins,
providing a single, unified abstraction for both graph and analytics
computations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from codeintel.core.validation.outcome import ValidationOutcome

if TYPE_CHECKING:
    from codeintel.core.plugins.execution.context import PluginExecutionContext
    from codeintel.core.plugins.types.result import PluginResult


PluginKind = Literal["builder", "metric", "validation", "analytics", "tool"]


PluginStage = Literal[
    "goid",
    "edges",
    "structure",
    "core",
    "graph",
    "function",
    "test",
    "subsystem",
    "data_model",
    "data_model_usage",
    "entrypoints",
    "semantic",
    "risk",
    "cfg",
    "dfg",
    "symbol",
    "config",
    "stats",
    "validation",
    "other",
    "pipeline_ingestion",
    "pipeline_graphs",
    "pipeline_analytics",
    "pipeline_export",
]

PluginSeverity = Literal["fatal", "soft_fail", "skip_on_error"]

PluginIsolation = Literal["process", "thread", "none"]

CapabilityKind = Literal["dataset", "artifact", "service", "graph"]

InputSource = Literal["config", "runtime", "prior_plugin"]


@dataclass(frozen=True)
class PluginCapability:
    """Declare what a plugin provides or requires.

    Capabilities enable loose coupling between plugins. A plugin declares
    what it provides (e.g., "analytics.function_types") and what it
    requires (e.g., "core.goids"). The runtime resolves these dependencies
    automatically.

    Attributes
    ----------
    name
        Stable identifier for the capability (e.g., "analytics.function_types").
    kind
        Classification of the capability type.
    """

    name: str
    kind: CapabilityKind = "dataset"


@dataclass(frozen=True)
class PluginInputSpec:
    """Typed input specification for a plugin.

    Describes what configuration or data a plugin requires to execute.

    Attributes
    ----------
    name
        Identifier for this input (used in validation messages).
    type_ref
        String reference to the expected type.
    required
        Whether this input must be provided.
    source
        Where the input comes from.
    default
        Default value if not provided and not required.
    """

    name: str
    type_ref: str
    required: bool = True
    source: InputSource = "config"
    default: object | None = None


@dataclass(frozen=True)
class PluginOutputSpec:
    """Typed output specification for a plugin.

    Describes what tables or artifacts a plugin produces.

    Attributes
    ----------
    name
        Logical name for this output.
    tables
        Database tables this output writes to.
    artifact_type
        Optional artifact type identifier for non-table outputs.
    min_rows
        Minimum expected rows for validation.
    required_columns
        Columns that must be present in the output.
    """

    name: str
    tables: tuple[str, ...] = ()
    artifact_type: str | None = None
    min_rows: int | None = None
    required_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class PluginResourceHints:
    """Runtime resource hints for scheduling and observability.

    Attributes
    ----------
    max_runtime_ms
        Maximum expected runtime in milliseconds.
    max_memory_mb
        Maximum expected memory usage in megabytes.
    cpu_intensive
        Whether this plugin is CPU-bound.
    io_intensive
        Whether this plugin is I/O-bound.
    requires_gpu
        Whether the plugin benefits from GPU acceleration.
    priority
        Scheduling priority (higher = more important).
    """

    max_runtime_ms: int | None = None
    max_memory_mb: int | None = None
    cpu_intensive: bool = False
    io_intensive: bool = False
    requires_gpu: bool = False
    priority: int = 0


@dataclass(frozen=True)
class PluginMetadata:
    """Complete metadata for a plugin.

    This dataclass captures all declarative information about a plugin,
    enabling introspection, documentation generation, planning, and
    dependency resolution.

    Attributes
    ----------
    name
        Stable identifier used in config and logs.
    description
        Human-readable description.
    kind
        Plugin classification (builder, metric, validation, analytics, tool).
    stage
        Processing stage grouping.
    version
        Plugin version (for cache invalidation).
    enabled_by_default
        Whether this plugin runs when no explicit list is provided.
    severity
        How failures should be handled.
    depends_on
        Explicit plugin dependencies (by name).
    provides
        Capabilities this plugin provides to others.
    requires
        Capabilities this plugin needs from others.
    inputs
        Required and optional input specifications.
    outputs
        Tables and artifacts produced.
    produces_tables
        DuckDB table keys populated by this plugin.
    produces_graphs
        Graph kinds this plugin builds (for builders).
    requires_graphs
        Graph kinds this plugin needs (for metrics).
    resource_hints
        Runtime resource hints.
    supports_incremental
        Whether incremental execution is supported.
    isolation_kind
        Type of isolation needed for execution.
    requires_isolation
        Whether the plugin needs process/thread isolation.
    scope_aware
        Whether the plugin is scope-aware.
    supported_scopes
        Scopes supported when scope-aware.
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
    contract_checkers
        Contract checker identifiers used by the plugin.
    tags
        Free-form tags for categorization.
    tool_binary
        Binary name for tool plugins (kind="tool").
    produces_artifacts
        Artifact names produced by tool plugins.
    consumes_configs
        Config fields consumed by tool plugins.
    """

    name: str
    description: str
    kind: PluginKind
    stage: PluginStage
    version: str = "1.0.0"
    enabled_by_default: bool = True
    severity: PluginSeverity = "fatal"
    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    inputs: tuple[PluginInputSpec, ...] = ()
    outputs: tuple[PluginOutputSpec, ...] = ()
    produces_tables: tuple[str, ...] = ()
    produces_graphs: tuple[str, ...] = ()
    requires_graphs: tuple[str, ...] = ()
    resource_hints: PluginResourceHints | None = None
    supports_incremental: bool = False
    isolation_kind: PluginIsolation = "none"
    requires_isolation: bool = False
    scope_aware: bool = False
    supported_scopes: tuple[str, ...] = ()
    version_hash: str | None = None
    config_schema_ref: str | None = None
    row_count_tables: tuple[str, ...] = ()
    cache_populates: tuple[str, ...] = ()
    cache_consumes: tuple[str, ...] = ()
    contract_checkers: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    # Tool plugin fields (optional, used when kind="tool")
    tool_binary: str | None = None
    produces_artifacts: tuple[str, ...] = ()
    consumes_configs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize derived flags."""
        if self.requires_isolation or self.isolation_kind != "none":
            object.__setattr__(
                self,
                "requires_isolation",
                self.requires_isolation or self.isolation_kind != "none",
            )


@runtime_checkable
class PluginProtocol(Protocol):
    """Unified protocol for all plugins.

    Plugins implementing this protocol can be registered and executed
    by the plugin runtime. The protocol provides a clean separation
    between metadata (declarative) and execution (imperative).

    This protocol is used by both graph plugins (builders, metrics,
    validation) and analytics plugins.
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

    def execute(self, ctx: PluginExecutionContext) -> PluginResult:
        """Execute the plugin with the given context.

        Parameters
        ----------
        ctx
            Execution context providing access to storage, config, and runtime.

        Returns
        -------
        PluginResult
            Result of the plugin execution.
        """
        ...

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationOutcome:
        """Validate that required inputs are available.

        Parameters
        ----------
        ctx
            Execution context to validate against.

        Returns
        -------
        ValidationOutcome
            Validation outcome with any errors or warnings.
        """
        ...


__all__ = [
    "CapabilityKind",
    "InputSource",
    "PluginCapability",
    "PluginInputSpec",
    "PluginIsolation",
    "PluginKind",
    "PluginMetadata",
    "PluginOutputSpec",
    "PluginProtocol",
    "PluginResourceHints",
    "PluginSeverity",
    "PluginStage",
    "ValidationOutcome",
]
