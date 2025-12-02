"""Unified plugin protocol for analytics plugins.

This module defines the core protocol and types for analytics plugins,
replacing the legacy dual system of AnalyticsPlugin and GraphMetricPlugin
with a single, unified abstraction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.analytics.core.execution_context import PluginExecutionContext

PluginStage = Literal[
    "graph",
    "function",
    "function_history",
    "test",
    "coverage",
    "subsystem",
    "data_model",
    "data_model_usage",
    "entrypoints",
    "profiles",
    "history",
    "semantic",
    "hotspots",
    "risk",
    "cfg",
    "dfg",
    "symbol",
    "config",
    "stats",
    "other",
]

PluginSeverity = Literal["fatal", "soft_fail", "skip_on_error"]
CapabilityKind = Literal["dataset", "artifact", "service"]
InputSource = Literal["config", "runtime", "prior_plugin"]


@dataclass(frozen=True)
class PluginCapability:
    """Declare what a plugin provides or requires.

    Capabilities enable loose coupling between plugins. A plugin declares
    what it provides (e.g., "analytics.function_metrics") and what it
    requires (e.g., "core.goids"). The runtime resolves these dependencies
    automatically.

    Attributes
    ----------
    name
        Stable identifier for the capability (e.g., "analytics.function_metrics").
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
        String reference to the expected type (e.g., "FunctionAnalyticsStepConfig").
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
        Logical name for this output (e.g., "metrics").
    tables
        Database tables this output writes to.
    artifact_type
        Optional artifact type identifier for non-table outputs.
    min_rows
        Minimum expected rows for validation (None = no minimum).
    required_columns
        Columns that must be present in the output.
    """

    name: str
    tables: tuple[str, ...] = ()
    artifact_type: str | None = None
    min_rows: int | None = None
    required_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationResult:
    """Result of input or output validation.

    Attributes
    ----------
    valid
        Whether validation passed.
    errors
        List of validation error messages.
    warnings
        List of non-fatal warning messages.
    """

    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @staticmethod
    def success() -> ValidationResult:
        """Create a successful validation result.

        Returns
        -------
        ValidationResult
            Validation result marked as successful.
        """
        return ValidationResult(valid=True)

    @staticmethod
    def failure(errors: tuple[str, ...]) -> ValidationResult:
        """Create a failed validation result.

        Parameters
        ----------
        errors
            Validation error messages.

        Returns
        -------
        ValidationResult
            Validation result marked as failed with the provided errors.
        """
        return ValidationResult(valid=False, errors=errors)


@dataclass(frozen=True)
class PluginResult:
    """Result returned by plugin execution.

    Attributes
    ----------
    success
        Whether execution completed successfully.
    row_counts
        Mapping of table names to row counts written.
    artifacts
        Mapping of artifact names to artifact data.
    input_hash
        Hash of inputs for caching.
    options_hash
        Hash of options for caching.
    error
        Error message if execution failed.
    warnings
        Non-fatal warnings from execution.
    meta
        Additional metadata about the execution.
    """

    success: bool = True
    row_counts: Mapping[str, int] = field(default_factory=dict)
    artifacts: Mapping[str, object] = field(default_factory=dict)
    input_hash: str | None = None
    options_hash: str | None = None
    error: str | None = None
    warnings: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def ok(
        *,
        row_counts: Mapping[str, int] | None = None,
        artifacts: Mapping[str, object] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> PluginResult:
        """Create a successful result.

        Parameters
        ----------
        row_counts
            Optional mapping of table names to row counts written.
        artifacts
            Optional mapping of produced artifacts.
        meta
            Optional execution metadata.

        Returns
        -------
        PluginResult
            Result object marked as successful.
        """
        return PluginResult(
            success=True,
            row_counts=row_counts or {},
            artifacts=artifacts or {},
            meta=meta or {},
        )

    @staticmethod
    def fail(error: str, *, warnings: tuple[str, ...] = ()) -> PluginResult:
        """Create a failed result.

        Parameters
        ----------
        error
            Error message describing the failure.
        warnings
            Optional non-fatal warnings collected during execution.

        Returns
        -------
        PluginResult
            Result object marked as failed.
        """
        return PluginResult(success=False, error=error, warnings=warnings)


@dataclass(frozen=True)
class PluginResourceHints:
    """Runtime resource hints for scheduling and observability.

    Attributes
    ----------
    max_runtime_ms
        Maximum expected runtime in milliseconds.
    max_memory_mb
        Maximum expected memory usage in megabytes.
    requires_gpu
        Whether the plugin benefits from GPU acceleration.
    priority
        Scheduling priority (higher = more important).
    """

    max_runtime_ms: int | None = None
    max_memory_mb: int | None = None
    requires_gpu: bool = False
    priority: int = 0


@dataclass(frozen=True)
class PluginMetadata:
    """Complete metadata for an analytics plugin.

    This dataclass captures all declarative information about a plugin,
    enabling introspection, documentation generation, and planning.

    Attributes
    ----------
    name
        Stable identifier used in config and logs.
    description
        Human-readable description.
    stage
        Processing stage grouping.
    version
        Plugin version (for cache invalidation).
    enabled_by_default
        Whether this plugin runs when no explicit list is provided.
    severity
        How failures should be handled.
    inputs
        Required and optional inputs.
    outputs
        Tables and artifacts produced.
    capabilities_provided
        Capabilities this plugin provides to others.
    capabilities_required
        Capabilities this plugin needs from others.
    depends_on
        Explicit plugin dependencies (by name).
    resource_hints
        Runtime resource hints.
    requires_isolation
        Whether the plugin requires process/thread isolation.
    isolation_kind
        Type of isolation required.
    tags
        Free-form tags for categorization.
    """

    name: str
    description: str
    stage: PluginStage
    version: str = "1.0.0"
    enabled_by_default: bool = True
    severity: PluginSeverity = "fatal"
    inputs: tuple[PluginInputSpec, ...] = ()
    outputs: tuple[PluginOutputSpec, ...] = ()
    capabilities_provided: tuple[PluginCapability, ...] = ()
    capabilities_required: tuple[PluginCapability, ...] = ()
    depends_on: tuple[str, ...] = ()
    resource_hints: PluginResourceHints | None = None
    requires_isolation: bool = False
    isolation_kind: Literal["process", "thread"] | None = None
    tags: tuple[str, ...] = ()


@runtime_checkable
class AnalyticsPluginProtocol(Protocol):
    """Unified protocol for all analytics plugins.

    Plugins implementing this protocol can be registered and executed
    by the analytics runtime. The protocol provides a clean separation
    between metadata (declarative) and execution (imperative).
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
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

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
        """Validate that required inputs are available.

        Parameters
        ----------
        ctx
            Execution context to validate against.

        Returns
        -------
        ValidationResult
            Validation outcome with any errors or warnings.
        """
        ...


@dataclass(frozen=True)
class PluginExecutionRecord:
    """Record of a single plugin execution.

    Attributes
    ----------
    plugin_name
        Name of the executed plugin.
    status
        Execution status.
    started_at
        When execution started.
    ended_at
        When execution ended.
    duration_ms
        Execution duration in milliseconds.
    attempts
        Number of execution attempts.
    result
        Plugin result if available.
    error
        Error message if failed.
    """

    plugin_name: str
    status: Literal["succeeded", "failed", "skipped"]
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    attempts: int = 1
    result: PluginResult | None = None
    error: str | None = None


__all__ = [
    "AnalyticsPluginProtocol",
    "CapabilityKind",
    "InputSource",
    "PluginCapability",
    "PluginExecutionRecord",
    "PluginInputSpec",
    "PluginMetadata",
    "PluginOutputSpec",
    "PluginResourceHints",
    "PluginResult",
    "PluginSeverity",
    "PluginStage",
    "ValidationResult",
]
