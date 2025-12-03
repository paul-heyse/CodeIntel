"""Core protocol and types for ingestion plugins.

This module defines the protocol and types for ingestion plugins, providing
a modernized interface aligned with the analytics graph plugin architecture
while preserving ingestion-specific functionality.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeGuard, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from codeintel.ingestion.core.execution_context import IngestExecutionContext

IngestStage = Literal[
    "scan",
    "parse",
    "index",
    "enrich",
    "validate",
]

IngestSeverity = Literal[
    "fatal",
    "soft_fail",
    "skip_on_error",
]

IngestIsolationKind = Literal[
    "process",
    "thread",
    "none",
]


@dataclass(frozen=True)
class IngestResourceHints:
    """Optional resource hints used for planning and observability.

    Attributes
    ----------
    max_runtime_ms
        Maximum expected runtime in milliseconds.
    memory_mb_hint
        Expected memory usage in megabytes.
    cpu_intensive
        Whether this plugin is CPU-bound and benefits from parallelism.
    io_intensive
        Whether this plugin is I/O-bound.
    """

    max_runtime_ms: int | None = None
    memory_mb_hint: int | None = None
    cpu_intensive: bool = False
    io_intensive: bool = False


@dataclass(frozen=True)
class IngestPluginMetadata:
    """Metadata for an ingestion plugin.

    Captures all declarative information about an ingestion plugin for
    introspection, documentation, dependency resolution, and planning.

    Attributes
    ----------
    name
        Unique plugin identifier (e.g., "ast_extract").
    description
        Human-readable description of what the plugin does.
    stage
        Processing stage in the ingestion pipeline.
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
    tool_dependencies
        Tool plugins required (e.g., "pyright", "scip").
    resource_hints
        Runtime resource hints for planning.
    supports_incremental
        Whether incremental ingestion is supported.
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
    config_class
        Step config class to auto-build from context.
    config_mapping
        Custom field mapping for config building (config_field -> context_attr).
    """

    name: str
    description: str
    stage: IngestStage
    severity: IngestSeverity = "fatal"
    enabled_by_default: bool = True
    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    produces_tables: tuple[str, ...] = ()
    tool_dependencies: tuple[str, ...] = ()
    resource_hints: IngestResourceHints | None = None
    supports_incremental: bool = False
    isolation_kind: IngestIsolationKind = "none"
    options_model: type[BaseModel] | None = None
    options_default: object | None = None
    version_hash: str | None = None
    config_schema_ref: str | None = None
    config_class: type | None = None
    config_mapping: Mapping[str, str] | None = None


@dataclass
class IngestRuntimeScratch:
    """Ephemeral scratch/cache store shared across plugin executions in a run.

    Provides a way for plugins to share intermediate data within a single
    execution run without persisting to the database.
    """

    _store: dict[str, object] = field(default_factory=dict)
    _cleanup: list[Callable[[], None]] = field(default_factory=list)

    def declare(self, key: str, value: object) -> None:
        """Record a value for later consumption by other plugins.

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

    def has(self, key: str) -> bool:
        """Check if a key exists in the scratch store.

        Parameters
        ----------
        key
            Identifier to check.

        Returns
        -------
        bool
            True if key exists.
        """
        return key in self._store

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
class IngestPluginResult:
    """Result returned by plugin execution.

    Attributes
    ----------
    success
        Whether execution completed successfully.
    row_counts
        Mapping of table names to row counts written.
    error
        Error message if execution failed.
    error_kind
        Classification of the error type.
    skipped
        Whether the plugin was skipped (e.g., missing tool).
    skip_reason
        Reason for skipping if applicable.
    artifacts
        Mapping of artifact names to paths produced.
    input_hash
        Hash of inputs for caching.
    options_hash
        Hash of options for caching.
    """

    success: bool = True
    row_counts: Mapping[str, int] | None = None
    error: str | None = None
    error_kind: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    artifacts: Mapping[str, Path] | None = None
    input_hash: str | None = None
    options_hash: str | None = None

    @staticmethod
    def ok(
        *,
        row_counts: Mapping[str, int] | None = None,
        artifacts: Mapping[str, Path] | None = None,
        input_hash: str | None = None,
        options_hash: str | None = None,
    ) -> IngestPluginResult:
        """Create a successful result.

        Parameters
        ----------
        row_counts
            Optional mapping of table names to row counts written.
        artifacts
            Optional mapping of artifact names to paths.
        input_hash
            Optional hash of inputs.
        options_hash
            Optional hash of options.

        Returns
        -------
        IngestPluginResult
            Result object marked as successful.
        """
        return IngestPluginResult(
            success=True,
            row_counts=row_counts,
            artifacts=artifacts,
            input_hash=input_hash,
            options_hash=options_hash,
        )

    @staticmethod
    def fail(error: str, *, error_kind: str | None = None) -> IngestPluginResult:
        """Create a failed result.

        Parameters
        ----------
        error
            Error message describing the failure.
        error_kind
            Optional classification of the error type.

        Returns
        -------
        IngestPluginResult
            Result object marked as failed.
        """
        return IngestPluginResult(success=False, error=error, error_kind=error_kind)

    @staticmethod
    def skip(reason: str) -> IngestPluginResult:
        """Create a skipped result.

        Parameters
        ----------
        reason
            Reason for skipping execution.

        Returns
        -------
        IngestPluginResult
            Result object marked as skipped.
        """
        return IngestPluginResult(success=True, skipped=True, skip_reason=reason)


@runtime_checkable
class IngestPluginProtocol(Protocol):
    """Protocol for ingestion plugins.

    Ingestion plugins implement this protocol to be registered and executed
    by the ingestion runtime.
    """

    @property
    def metadata(self) -> IngestPluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        IngestPluginMetadata
            Metadata describing the plugin.
        """
        ...

    def execute(self, ctx: IngestExecutionContext) -> IngestPluginResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Ingestion plugin execution context.

        Returns
        -------
        IngestPluginResult
            Result of plugin execution.
        """
        ...


def is_ingest_plugin(obj: object) -> TypeGuard[IngestPluginProtocol]:
    """Validate an object conforms to IngestPluginProtocol.

    This function performs runtime validation and provides type narrowing
    for the static type checker via TypeGuard. It checks that the object
    has both a metadata property returning IngestPluginMetadata and a
    callable execute method.

    Parameters
    ----------
    obj
        Object to validate.

    Returns
    -------
    TypeGuard[IngestPluginProtocol]
        True if obj conforms to the protocol, enabling type narrowing.

    Examples
    --------
    >>> from codeintel.ingestion.plugins.protocol import is_ingest_plugin
    >>> class MyPlugin:
    ...     @property
    ...     def metadata(self):
    ...         return IngestPluginMetadata(name="test", description="test", stage="parse")
    ...
    ...     def execute(self, ctx):
    ...         return IngestPluginResult.ok()
    >>> is_ingest_plugin(MyPlugin())
    True
    """
    # Check for required attributes
    if not hasattr(obj, "metadata") or not hasattr(obj, "execute"):
        return False

    # Verify execute is callable
    execute_attr = getattr(obj, "execute", None)
    if not callable(execute_attr):
        return False

    # Verify metadata returns an IngestPluginMetadata instance
    meta = getattr(obj, "metadata", None)
    return isinstance(meta, IngestPluginMetadata)


@dataclass(frozen=True)
class IngestPluginSkip:
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
        "missing_tool",
        "config_error",
        "incremental_skip",
    ]


@dataclass(frozen=True)
class IngestPluginPlan:
    """Resolved execution plan for ingestion plugins.

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

    plugins: tuple[IngestPluginProtocol, ...]
    plan_id: str
    skipped_plugins: tuple[IngestPluginSkip, ...] = ()
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


DEFAULT_INGEST_PLUGINS: tuple[str, ...] = (
    "repo_scan",
    "scip_ingest",
    "cst_extract",
    "ast_extract",
    "typing_ingest",
    "coverage_ingest",
    "tests_ingest",
    "docstrings_ingest",
    "config_ingest",
)


__all__ = [
    "DEFAULT_INGEST_PLUGINS",
    "IngestIsolationKind",
    "IngestPluginMetadata",
    "IngestPluginPlan",
    "IngestPluginProtocol",
    "IngestPluginResult",
    "IngestPluginSkip",
    "IngestResourceHints",
    "IngestRuntimeScratch",
    "IngestSeverity",
    "IngestStage",
    "is_ingest_plugin",
]
