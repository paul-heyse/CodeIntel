"""Core protocol and types for ingestion plugins.

This module defines the protocol and types for ingestion plugins, providing
a modernized interface aligned with the analytics graph plugin architecture
while preserving ingestion-specific functionality.

NOTE: Imports inside functions are intentional to avoid circular dependencies.
"""
# ruff: noqa: PLC0415

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.ingestion.ingest_runs import IngestRun, IngestRunSink
    from codeintel.ingestion.source_scanner import ScanProfile
    from codeintel.ingestion.tool_runner import ToolRunner
    from codeintel.ingestion.tool_service import ToolService
    from codeintel.storage.gateway import StorageGateway

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
        Step config class to auto-build from context via harness.
    config_mapping
        Custom field mapping for config building (config_field -> context_attr).
    harness_config
        Harness configuration for automated error handling and row counting.
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
    # Harness integration fields
    config_class: type | None = None
    config_mapping: Mapping[str, str] | None = None
    harness_config: object | None = None  # HarnessConfig, forward ref to avoid circular


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
        import logging

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
class IngestPluginContext:
    """Execution context for ingestion plugins.

    Provides access to storage, configuration, change tracking, and shared
    scratch space for inter-plugin communication.

    Attributes
    ----------
    gateway
        StorageGateway providing DuckDB access.
    snapshot
        Repository snapshot reference.
    paths
        Build paths configuration.
    tools
        Tools configuration.
    code_profile
        Code scanning profile.
    config_profile
        Config scanning profile.
    tool_runner
        Optional shared tool runner.
    tool_service
        Optional shared tool service.
    change_tracker
        Optional change tracker for incremental ingestion.
    ingest_run_sink
        Optional sink for recording run metrics.
    current_ingest_run
        Current run record for metrics.
    scratch
        Shared scratch space for inter-plugin data.
    options
        Plugin-specific options.
    plugin_name
        Name of the executing plugin.
    run_id
        Unique identifier for this execution run.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    tools: ToolsConfig
    code_profile: ScanProfile
    config_profile: ScanProfile
    tool_runner: ToolRunner | None = None
    tool_service: ToolService | None = None
    change_tracker: ChangeTracker | None = None
    ingest_run_sink: IngestRunSink | None = None
    current_ingest_run: IngestRun | None = None
    scratch: IngestRuntimeScratch = field(default_factory=IngestRuntimeScratch)
    options: object | None = None
    plugin_name: str | None = None
    run_id: str | None = None

    @property
    def repo_root(self) -> Path:
        """Repository root for the current snapshot.

        Returns
        -------
        Path
            Absolute path to the repository root.
        """
        return self.snapshot.repo_root

    @property
    def repo(self) -> str:
        """Repository slug for the current snapshot.

        Returns
        -------
        str
            Repository identifier.
        """
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier for the current snapshot.

        Returns
        -------
        str
            Commit hash or identifier.
        """
        return self.snapshot.commit

    @property
    def build_dir(self) -> Path:
        """Build directory derived from execution config.

        Returns
        -------
        Path
            Path to the build directory.
        """
        return self.paths.build_dir

    @property
    def document_output_dir(self) -> Path:
        """Document output directory resolved for the snapshot.

        Returns
        -------
        Path
            Path to the document output directory.
        """
        return self.paths.document_output_dir

    def require_tracker(self) -> ChangeTracker:
        """Return change tracker or raise if missing.

        Use this when a tracker is required for plugin execution.

        Returns
        -------
        ChangeTracker
            The change tracker.

        Raises
        ------
        RuntimeError
            If change tracker is not available.
        """
        if self.change_tracker is not None:
            return self.change_tracker

        # Try to get from scratch (populated by repo_scan)
        tracker = self.scratch.consume("change_tracker")
        if tracker is not None:
            from codeintel.ingestion.change_tracker import ChangeTracker

            if isinstance(tracker, ChangeTracker):
                return tracker

        message = "Change tracker required but not available; run repo_scan first"
        raise RuntimeError(message)

    def tool_service_or_default(self) -> ToolService:
        """Return tool service or construct a default.

        Returns
        -------
        ToolService
            Existing or newly constructed tool service.
        """
        if self.tool_service is not None:
            return self.tool_service

        from codeintel.ingestion.tool_runner import ToolRunner
        from codeintel.ingestion.tool_service import ToolService

        runner = self.tool_runner or ToolRunner(
            cache_dir=self.paths.tool_cache,
            tools_config=self.tools,
        )
        return ToolService(runner, self.tools)

    def count_produced_tables(
        self,
        tables: tuple[str, ...],
    ) -> Mapping[str, int]:
        """Count rows in the specified tables.

        Parameters
        ----------
        tables
            Table names to count.

        Returns
        -------
        Mapping[str, int]
            Mapping of table names to row counts.
        """
        counts: dict[str, int] = {}
        for table in tables:
            try:
                result = self.gateway.con.execute(
                    f"SELECT COUNT(*) FROM {table}",  # noqa: S608
                ).fetchone()
                counts[table] = int(result[0]) if result else 0
            except Exception:  # noqa: BLE001
                counts[table] = 0
        return counts


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

    def execute(self, ctx: IngestPluginContext) -> IngestPluginResult:
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
    "IngestPluginContext",
    "IngestPluginMetadata",
    "IngestPluginPlan",
    "IngestPluginProtocol",
    "IngestPluginResult",
    "IngestPluginSkip",
    "IngestResourceHints",
    "IngestRuntimeScratch",
    "IngestSeverity",
    "IngestStage",
]
