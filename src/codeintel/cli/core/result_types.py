"""Result type definitions for CLI handlers.

Each handler that returns structured data should have a corresponding
result type defined here. Generic result types (ListResult, ActionResult, etc.)
are provided for common patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from codeintel.cli.core.results import result_type


@result_type
@dataclass(frozen=True)
class ListResult[T]:
    """Generic result type for list operations.

    Use this for any command that returns a list of items with a count.
    The @result_type decorator auto-generates to_dict() that handles
    nested serialization.

    Parameters
    ----------
    items
        List of result items.
    count
        Total number of items.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @result_type
    ... @dataclass(frozen=True)
    ... class UserInfo:
    ...     name: str
    ...     active: bool
    >>> result = ListResult(items=[UserInfo("alice", True)], count=1)
    >>> result.to_dict()
    {'items': [{'name': 'alice', 'active': True}], 'count': 1}
    """

    items: list[T]
    count: int

    @classmethod
    def from_items(cls, items: list[T]) -> ListResult[T]:
        """Create ListResult from items, auto-computing count.

        Parameters
        ----------
        items
            List of items.

        Returns
        -------
        ListResult[T]
            Result with items and count.
        """
        return cls(items=items, count=len(items))


@result_type
@dataclass(frozen=True)
class ActionResult:
    """Result type for action commands (create, delete, update).

    Use this for commands that perform an action and report success.

    Parameters
    ----------
    action
        The action performed (e.g., "created", "deleted", "updated").
    success
        Whether the action succeeded.
    affected_count
        Number of items affected.
    message
        Optional success/failure message.

    Examples
    --------
    >>> result = ActionResult(action="created", success=True, affected_count=1)
    >>> result.to_dict()
    {'action': 'created', 'success': True, 'affected_count': 1}
    """

    action: str
    success: bool
    affected_count: int = 0
    message: str | None = None


@result_type
@dataclass(frozen=True)
class StatusResult:
    """Result type for status check commands.

    Use this for commands that report system or component status.

    Parameters
    ----------
    status
        Status value (e.g., "ok", "warning", "error").
    message
        Human-readable status message.
    details
        Optional additional details.

    Examples
    --------
    >>> result = StatusResult(status="ok", message="All systems operational")
    >>> result.to_dict()
    {'status': 'ok', 'message': 'All systems operational'}
    """

    status: str
    message: str
    details: dict[str, object] | None = None


@result_type
@dataclass(frozen=True)
class ExportResult:
    """Result type for export/generate commands.

    Use this for commands that write output to files.

    Parameters
    ----------
    output_path
        Path where output was written.
    item_count
        Number of items exported.
    duration_seconds
        Optional export duration.

    Examples
    --------
    >>> result = ExportResult(output_path="/tmp/export.json", item_count=100)
    >>> result.to_dict()
    {'output_path': '/tmp/export.json', 'item_count': 100}
    """

    output_path: str
    item_count: int
    duration_seconds: float | None = None


@result_type
@dataclass(frozen=True)
class StorageDatabaseExportResult:
    """Result type for storage database export.

    Parameters
    ----------
    db_path
        Database path that was exported.
    output_dir
        Directory containing the exported database.
    duration_seconds
        Optional export duration.
    """

    db_path: str
    output_dir: str
    duration_seconds: float | None = None


@result_type
@dataclass(frozen=True)
class StorageDatabaseImportResult:
    """Result type for storage database import.

    Parameters
    ----------
    db_path
        Database path that received the import.
    input_dir
        Directory containing the exported database to import.
    duration_seconds
        Optional import duration.
    """

    db_path: str
    input_dir: str
    duration_seconds: float | None = None


@dataclass(frozen=True)
class OperationListResult:
    """Result from op list command.

    Parameters
    ----------
    operations
        List of operation metadata dictionaries.
    count
        Total number of operations.
    """

    operations: list[dict[str, str | None]]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"operations": self.operations, "count": self.count}


@dataclass(frozen=True)
class OperationCallResult:
    """Result from op call command.

    Parameters
    ----------
    operation_id
        The operation that was invoked.
    result
        The operation result data.
    """

    operation_id: str
    result: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"operation_id": self.operation_id, "result": self.result}


@dataclass(frozen=True)
class DatasetListResult:
    """Result from dataset list command.

    Parameters
    ----------
    datasets
        List of dataset metadata dictionaries.
    count
        Total number of datasets.
    """

    datasets: list[dict[str, str | None]]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"datasets": self.datasets, "count": self.count}


@dataclass(frozen=True)
class DatasetDescribeResult:
    """Result from dataset describe command.

    Parameters
    ----------
    table_key
        The dataset table key.
    columns
        List of column metadata dictionaries.
    row_count
        Number of rows (if available).
    name
        Dataset name.
    description
        Dataset description.
    owner_package
        Owner package name.
    upstream_dependencies
        List of upstream dependencies.
    """

    table_key: str
    columns: list[dict[str, str | bool]]
    row_count: int | None
    name: str
    description: str | None = None
    owner_package: str | None = None
    upstream_dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result: dict[str, object] = {
            "table_key": self.table_key,
            "name": self.name,
            "columns": self.columns,
        }
        if self.description:
            result["description"] = self.description
        if self.owner_package:
            result["owner_package"] = self.owner_package
        if self.row_count is not None:
            result["row_count"] = self.row_count
        if self.upstream_dependencies:
            result["upstream_dependencies"] = self.upstream_dependencies
        return result


@dataclass(frozen=True)
class DatasetVerifyResult:
    """Result from dataset verify command.

    Parameters
    ----------
    verified
        Whether verification passed.
    issues
        List of issues found (empty if verified).
    """

    verified: bool
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"verified": self.verified, "issues": self.issues}


@dataclass(frozen=True)
class DatasetInfoResult:
    """Result from dataset info command.

    Provides comprehensive schema information for a dataset including
    column definitions, metadata, and JSON schema representation.

    Parameters
    ----------
    name
        Dataset table key (e.g., "analytics.function_metrics").
    columns
        Tuple of column names in the schema.
    metadata
        Dataset metadata as a dictionary.
    json_schema
        JSON Schema representation of the dataset.
    has_pandera_schema
        Whether a Pandera schema is registered for this dataset.
    """

    name: str
    columns: tuple[str, ...]
    metadata: dict[str, object]
    json_schema: dict[str, object]
    has_pandera_schema: bool = True

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "name": self.name,
            "columns": list(self.columns),
            "column_count": len(self.columns),
            "metadata": self.metadata,
            "json_schema": self.json_schema,
            "has_pandera_schema": self.has_pandera_schema,
        }


@dataclass(frozen=True)
class DatasetFlowResult:
    """Result from dataset flow command.

    Shows the producer/consumer graph for a dataset, indicating which
    plugins produce this dataset and which consume it.

    Parameters
    ----------
    table_key
        Dataset table key.
    producers
        List of plugin names that produce this dataset.
    consumers
        List of plugin names that consume this dataset.
    """

    table_key: str
    producers: list[str]
    consumers: list[str]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "table_key": self.table_key,
            "producers": self.producers,
            "consumers": self.consumers,
            "producer_count": len(self.producers),
            "consumer_count": len(self.consumers),
        }


@dataclass(frozen=True)
class DatasetConstraintsResult:
    """Result from dataset constraints command.

    Shows all constraints extracted from the Pandera schema for a dataset.

    Parameters
    ----------
    table_key
        Dataset table key.
    constraints
        List of constraint dictionaries with kind, column, and expression.
    constraint_count
        Total number of constraints.
    inferred_from
        Sources from which constraints were inferred.
    """

    table_key: str
    constraints: list[dict[str, object]]
    constraint_count: int
    inferred_from: list[str]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "table_key": self.table_key,
            "constraints": self.constraints,
            "constraint_count": self.constraint_count,
            "inferred_from": self.inferred_from,
        }


@dataclass(frozen=True)
class BuildStatusResult:
    """Result from build status command.

    Parameters
    ----------
    targets
        List of target status dictionaries.
    stale_count
        Number of stale targets.
    fresh_count
        Number of fresh targets.
    """

    targets: list[dict[str, Any]]
    stale_count: int
    fresh_count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "targets": self.targets,
            "stale_count": self.stale_count,
            "fresh_count": self.fresh_count,
        }


@dataclass(frozen=True)
class BuildRunResult:
    """Result from build run command.

    Parameters
    ----------
    executed
        List of executed target names.
    skipped
        List of skipped target names.
    failed
        List of failed target names.
    duration_seconds
        Total duration in seconds.
    cache
        Optional cache report for the run (hit/miss summary and per-node outcomes).
    """

    executed: list[str]
    skipped: list[str]
    failed: list[str]
    duration_seconds: float
    cache: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result: dict[str, object] = {
            "executed": self.executed,
            "skipped": self.skipped,
            "failed": self.failed,
            "duration_seconds": self.duration_seconds,
        }
        if self.cache is not None:
            result["cache"] = self.cache
        return result


@dataclass(frozen=True)
class BuildPlanResult:
    """Result from build plan command.

    Parameters
    ----------
    requested
        List of requested target names.
    closure
        List of target names in dependency closure.
    entries
        List of plan entry dictionaries with status/reason.
    to_compute
        List of target names that will be computed.
    to_skip
        List of target names that will be skipped.
    blocked
        List of target names that are blocked.
    """

    requested: list[str]
    closure: list[str]
    entries: list[dict[str, object]]
    to_compute: list[str]
    to_skip: list[str]
    blocked: list[str]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "requested": self.requested,
            "closure": self.closure,
            "entries": self.entries,
            "to_compute": self.to_compute,
            "to_skip": self.to_skip,
            "blocked": self.blocked,
            "compute_count": len(self.to_compute),
            "skip_count": len(self.to_skip),
            "blocked_count": len(self.blocked),
        }


@dataclass(frozen=True)
class BuildExplainResult:
    """Result from build explain command.

    Parameters
    ----------
    target
        Target name being explained.
    status
        Plan status (compute, skip, blocked, missing).
    reason
        Reason for the status.
    is_stale
        Whether the target is stale (needs recomputation).
    input_hash_current
        Current computed input hash.
    input_hash_prior
        Prior input hash from manifest.
    changed_deps
        List of dependencies whose hashes changed.
    added_deps
        List of dependencies that were added.
    removed_deps
        List of dependencies that were removed.
    summary
        Human-readable summary of staleness.
    io_surface
        Optional per-target IO surface (reads/writes).
    """

    target: str
    status: str
    reason: str
    is_stale: bool
    input_hash_current: str | None
    input_hash_prior: str | None
    changed_deps: list[str]
    added_deps: list[str]
    removed_deps: list[str]
    summary: str
    io_surface: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "target": self.target,
            "status": self.status,
            "reason": self.reason,
            "is_stale": self.is_stale,
            "input_hash_current": self.input_hash_current,
            "input_hash_prior": self.input_hash_prior,
            "changed_deps": self.changed_deps,
            "added_deps": self.added_deps,
            "removed_deps": self.removed_deps,
            "summary": self.summary,
            **({"io_surface": self.io_surface} if self.io_surface is not None else {}),
        }


@dataclass(frozen=True)
class SubsystemListResult:
    """Result from subsystem list command.

    Parameters
    ----------
    subsystems
        List of subsystem metadata dictionaries.
    count
        Total number of subsystems.
    """

    subsystems: list[dict[str, Any]]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"subsystems": self.subsystems, "count": self.count}


@dataclass(frozen=True)
class ConfigShowResult:
    """Result from config show command.

    Parameters
    ----------
    config
        Configuration dictionary.
    sources
        Map of config keys to their sources.
    """

    config: dict[str, Any]
    sources: dict[str, list[str]]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"config": self.config, "sources": self.sources}


@dataclass(frozen=True)
class DryRunStep:
    """A single step in a dry-run plan.

    Parameters
    ----------
    operation_id
        Operation identifier.
    description
        Human-readable description.
    params
        Parameters for this step.
    is_prereq
        Whether this step is a prerequisite.
    """

    operation_id: str
    description: str
    params: dict[str, Any]
    is_prereq: bool = False

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "operation_id": self.operation_id,
            "description": self.description,
            "params": self.params,
            "is_prereq": self.is_prereq,
        }


@dataclass(frozen=True)
class DryRunResult:
    """Result from dry-run execution.

    Parameters
    ----------
    target_operation
        The target operation that would be executed.
    steps
        List of steps in the execution plan.
    estimated_duration
        Human-readable duration estimate (if available).
    warnings
        List of warnings about the plan.
    """

    target_operation: str
    steps: list[DryRunStep]
    estimated_duration: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result: dict[str, object] = {
            "target_operation": self.target_operation,
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.estimated_duration:
            result["estimated_duration"] = self.estimated_duration
        if self.warnings:
            result["warnings"] = self.warnings
        return result


@dataclass(frozen=True)
class BuildTargetInfo:
    """Information about a build target.

    Parameters
    ----------
    name
        Target name.
    status
        Current status (fresh, stale, error).
    last_run
        Timestamp of last successful run.
    dependencies
        List of dependency target names.
    outputs
        List of output paths.
    """

    name: str
    status: str
    last_run: str | None
    dependencies: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "name": self.name,
            "status": self.status,
            "last_run": self.last_run,
            "dependencies": self.dependencies,
            "outputs": self.outputs,
        }


@dataclass(frozen=True)
class BuildExecutionResult:
    """Result from executing a build.

    Parameters
    ----------
    targets_executed
        List of targets that were executed.
    targets_skipped
        List of targets that were skipped (already fresh).
    targets_failed
        List of targets that failed.
    total_duration_seconds
        Total execution time.
    success
        Whether all targets succeeded.
    """

    targets_executed: list[str]
    targets_skipped: list[str]
    targets_failed: list[str]
    total_duration_seconds: float
    success: bool = True

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "targets_executed": self.targets_executed,
            "targets_skipped": self.targets_skipped,
            "targets_failed": self.targets_failed,
            "total_duration_seconds": self.total_duration_seconds,
            "success": self.success,
        }


@dataclass(frozen=True)
class BuildAssetsResult:
    """Result from build assets command.

    Parameters
    ----------
    assets
        List of asset record dictionaries.
    count
        Total number of assets.
    format
        Output format used (table, json, csv).
    """

    assets: list[dict[str, Any]]
    count: int
    format: str = "table"

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "assets": self.assets,
            "count": self.count,
            "format": self.format,
        }


@dataclass(frozen=True)
class BuildLineageResult:
    """Result from build lineage command."""

    asset: str
    asset_kind: str
    root_version_hash: str
    direction: str
    depth: int
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    format: str = "json"

    def to_dict(self) -> dict[str, object]:
        return {
            "asset": self.asset,
            "asset_kind": self.asset_kind,
            "root_version_hash": self.root_version_hash,
            "direction": self.direction,
            "depth": self.depth,
            "nodes": self.nodes,
            "edges": self.edges,
            "format": self.format,
        }


@dataclass(frozen=True)
class BuildPromoteResult:
    """Result from build promote command."""

    asset: str
    asset_kind: str
    alias: str
    version_hash: str
    note: str | None = None
    format: str = "json"

    def to_dict(self) -> dict[str, object]:
        return {
            "asset": self.asset,
            "asset_kind": self.asset_kind,
            "alias": self.alias,
            "version_hash": self.version_hash,
            "note": self.note,
            "format": self.format,
        }


@dataclass(frozen=True)
class BuildResolveResult:
    """Result from build resolve command."""

    asset: str
    asset_kind: str
    alias: str
    version_hash: str
    format: str = "json"

    def to_dict(self) -> dict[str, object]:
        return {
            "asset": self.asset,
            "asset_kind": self.asset_kind,
            "alias": self.alias,
            "version_hash": self.version_hash,
            "format": self.format,
        }


@dataclass(frozen=True)
class BuildDiffResult:
    """Result from build diff command."""

    asset: str
    asset_kind: str
    from_spec: str
    to_spec: str
    from_version_hash: str
    to_version_hash: str
    diffs: dict[str, Any]
    cached: bool
    format: str = "json"

    def to_dict(self) -> dict[str, object]:
        return {
            "asset": self.asset,
            "asset_kind": self.asset_kind,
            "from": self.from_spec,
            "to": self.to_spec,
            "from_version_hash": self.from_version_hash,
            "to_version_hash": self.to_version_hash,
            "diffs": self.diffs,
            "cached": self.cached,
            "format": self.format,
        }


@dataclass(frozen=True)
class BuildHistoryResult:
    """Result from build history command.

    Parameters
    ----------
    runs
        List of build run records.
    count
        Total number of runs returned.
    targets
        Per-target run records when a specific run_id is queried.
        None when listing multiple runs.
    """

    runs: list[dict[str, Any]]
    count: int
    targets: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result: dict[str, object] = {"runs": self.runs, "count": self.count}
        if self.targets is not None:
            result["targets"] = self.targets
        return result


@dataclass(frozen=True)
class GraphStatsResult:
    """Statistics about a graph.

    Parameters
    ----------
    node_count
        Number of nodes.
    edge_count
        Number of edges.
    density
        Graph density (0.0 to 1.0).
    components
        Number of connected components.
    avg_degree
        Average node degree.
    """

    node_count: int
    edge_count: int
    density: float
    components: int
    avg_degree: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "density": self.density,
            "components": self.components,
            "avg_degree": self.avg_degree,
        }


@dataclass(frozen=True)
class GraphQueryResult:
    """Result from a graph query.

    Parameters
    ----------
    nodes
        Matching nodes with their attributes.
    edges
        Edges between matching nodes.
    query
        The query that was executed.
    """

    nodes: list[dict[str, object]]
    edges: list[dict[str, object]]
    query: str

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "query": self.query,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }


@dataclass(frozen=True)
class DocsStatusResult:
    """Result from docs status command.

    Parameters
    ----------
    generated_count
        Number of generated documentation files.
    pending_count
        Number of pending documentation files.
    stale_count
        Number of stale documentation files.
    last_generated
        Timestamp of last generation.
    """

    generated_count: int
    pending_count: int
    stale_count: int
    last_generated: str | None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "generated_count": self.generated_count,
            "pending_count": self.pending_count,
            "stale_count": self.stale_count,
            "last_generated": self.last_generated,
        }


@dataclass(frozen=True)
class DocsGenerateResult:
    """Result from docs generate command.

    Parameters
    ----------
    generated
        List of generated file paths.
    skipped
        List of skipped file paths.
    errors
        List of error messages.
    """

    generated: list[str]
    skipped: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "generated": self.generated,
            "skipped": self.skipped,
            "errors": self.errors,
            "generated_count": len(self.generated),
            "error_count": len(self.errors),
        }


@dataclass(frozen=True)
class HistoryListResult:
    """Result from history list command.

    Parameters
    ----------
    entries
        List of history entries.
    count
        Total number of entries.
    """

    entries: list[dict[str, Any]]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"entries": self.entries, "count": self.count}


@dataclass(frozen=True)
class HistoryDetailResult:
    """Result from history detail command.

    Parameters
    ----------
    entry_id
        Entry identifier.
    timestamp
        Entry timestamp.
    operation
        Operation name.
    status
        Operation status.
    duration_seconds
        Duration in seconds.
    details
        Additional details.
    """

    entry_id: str
    timestamp: str
    operation: str
    status: str
    duration_seconds: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "details": self.details,
        }


@dataclass(frozen=True)
class IdeStatusResult:
    """Result from ide status command.

    Parameters
    ----------
    connected
        Whether IDE is connected.
    ide_type
        Type of IDE (vscode, cursor, etc).
    workspace_path
        Current workspace path.
    extensions
        List of relevant extensions.
    """

    connected: bool
    ide_type: str | None
    workspace_path: str | None
    extensions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "connected": self.connected,
            "ide_type": self.ide_type,
            "workspace_path": self.workspace_path,
            "extensions": self.extensions,
        }


@dataclass(frozen=True)
class IdeConfigResult:
    """Result from ide config command.

    Parameters
    ----------
    settings
        IDE settings dictionary.
    path
        Settings file path.
    """

    settings: dict[str, Any]
    path: str | None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"settings": self.settings, "path": self.path}


@dataclass(frozen=True)
class SubsystemDetailResult:
    """Result from subsystem detail command.

    Parameters
    ----------
    name
        Subsystem name.
    description
        Subsystem description.
    modules
        List of modules in the subsystem.
    dependencies
        List of dependency subsystems.
    metrics
        Subsystem metrics.
    """

    name: str
    description: str | None
    modules: list[str]
    dependencies: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "name": self.name,
            "description": self.description,
            "modules": self.modules,
            "dependencies": self.dependencies,
            "metrics": self.metrics,
            "module_count": len(self.modules),
        }


@dataclass(frozen=True)
class StorageStatusResult:
    """Result from storage status command.

    Parameters
    ----------
    connected
        Whether storage is connected.
    database_path
        Path to the database.
    table_count
        Number of tables.
    size_bytes
        Database size in bytes.
    """

    connected: bool
    database_path: str | None
    table_count: int
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "connected": self.connected,
            "database_path": self.database_path,
            "table_count": self.table_count,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class StorageQueryResult:
    """Result from storage query command.

    Parameters
    ----------
    rows
        Query result rows.
    columns
        Column names.
    row_count
        Number of rows returned.
    execution_time_ms
        Query execution time in milliseconds.
    """

    rows: list[dict[str, Any]]
    columns: list[str]
    row_count: int
    execution_time_ms: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "rows": self.rows,
            "columns": self.columns,
            "row_count": self.row_count,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass(frozen=True)
class DatasetLintResult:
    """Result from dataset lint command.

    Parameters
    ----------
    passed
        Whether validation passed.
    issue_count
        Number of issues found.
    issues
        List of issue descriptions.
    """

    passed: bool
    issue_count: int
    issues: list[str]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "passed": self.passed,
            "issue_count": self.issue_count,
            "issues": self.issues,
        }


@dataclass(frozen=True)
class DatasetSnapshotResult:
    """Result from dataset snapshot command.

    Parameters
    ----------
    output_path
        Path where snapshot was written.
    datasets_count
        Number of datasets in snapshot.
    """

    output_path: str
    datasets_count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "output_path": self.output_path,
            "datasets_count": self.datasets_count,
        }


@dataclass(frozen=True)
class DatasetDiffResult:
    """Result from dataset diff command.

    Parameters
    ----------
    added
        List of added dataset names.
    removed
        List of removed dataset names.
    changed
        List of changed dataset names.
    has_differences
        Whether any differences were found.
    """

    added: list[str]
    removed: list[str]
    changed: list[str]
    has_differences: bool

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "added": self.added,
            "removed": self.removed,
            "changed": self.changed,
            "has_differences": self.has_differences,
        }


@dataclass(frozen=True)
class ValidateMacrosResult:
    """Result from macro validation command.

    Parameters
    ----------
    status
        Validation status (valid, skipped, invalid).
    missing_ingest
        List of missing ingest macro names.
    present_ingest
        List of present ingest macro names.
    dataset_rows_only
        List of datasets with rows only (no normalized macro).
    reason
        Optional reason for status (e.g., skip reason).
    """

    status: str
    missing_ingest: list[str]
    present_ingest: list[str]
    dataset_rows_only: list[str]
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result: dict[str, object] = {
            "status": self.status,
            "missing_ingest": self.missing_ingest,
            "present_ingest": self.present_ingest,
            "dataset_rows_only": self.dataset_rows_only,
        }
        if self.reason:
            result["reason"] = self.reason
        return result


@dataclass(frozen=True)
class ProfileStorageResult:
    """Result from storage profiling command.

    Parameters
    ----------
    db_path
        Path to the profiled database.
    output_dir
        Directory where profile output was written.
    include_views
        Whether views were included in profiling.
    """

    db_path: str
    output_dir: str
    include_views: bool

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "db_path": self.db_path,
            "output_dir": self.output_dir,
            "include_views": self.include_views,
        }


@dataclass(frozen=True)
class HistoryTimeseriesResult:
    """Result from history timeseries aggregation command.

    Parameters
    ----------
    output_db
        Path to the output database.
    commits_processed
        Number of commits processed.
    entity_kind
        Entity kind used for aggregation.
    """

    output_db: str
    commits_processed: int
    entity_kind: str

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "output_db": self.output_db,
            "commits_processed": self.commits_processed,
            "entity_kind": self.entity_kind,
        }


@dataclass(frozen=True)
class ServeStartResult:
    """Result from server start command.

    Parameters
    ----------
    server_type
        Type of server (http, mcp).
    host
        Server host address.
    port
        Server port number.
    auto_pipeline
        Whether auto-pipeline is enabled.
    repo
        Repository slug.
    commit
        Commit SHA.
    db_path
        Path to database.
    """

    server_type: str
    host: str | None
    port: int | None
    auto_pipeline: bool
    repo: str
    commit: str
    db_path: str

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "server_type": self.server_type,
            "host": self.host,
            "port": self.port,
            "auto_pipeline": self.auto_pipeline,
            "repo": self.repo,
            "commit": self.commit,
            "db_path": self.db_path,
        }


@dataclass(frozen=True)
class HealthCheckResult:
    """Result from health check command.

    Parameters
    ----------
    checks
        List of individual check results.
    overall_status
        Overall status (ok, warn, fail, skip).
    total_duration_ms
        Total time for all checks in milliseconds.
    """

    checks: list[dict[str, object]]
    overall_status: str
    total_duration_ms: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "checks": self.checks,
            "overall_status": self.overall_status,
            "total_duration_ms": self.total_duration_ms,
        }


__all__ = [
    "ActionResult",
    "BuildAssetsResult",
    "BuildExecutionResult",
    "BuildExplainResult",
    "BuildHistoryResult",
    "BuildPlanResult",
    "BuildRunResult",
    "BuildStatusResult",
    "BuildTargetInfo",
    "ConfigShowResult",
    "DatasetConstraintsResult",
    "DatasetDescribeResult",
    "DatasetDiffResult",
    "DatasetFlowResult",
    "DatasetInfoResult",
    "DatasetLintResult",
    "DatasetListResult",
    "DatasetSnapshotResult",
    "DatasetVerifyResult",
    "DocsGenerateResult",
    "DocsStatusResult",
    "DryRunResult",
    "DryRunStep",
    "ExportResult",
    "GraphQueryResult",
    "GraphStatsResult",
    "HealthCheckResult",
    "HistoryDetailResult",
    "HistoryListResult",
    "HistoryTimeseriesResult",
    "IdeConfigResult",
    "IdeStatusResult",
    "ListResult",
    "OperationCallResult",
    "OperationListResult",
    "ProfileStorageResult",
    "ServeStartResult",
    "StatusResult",
    "StorageDatabaseExportResult",
    "StorageDatabaseImportResult",
    "StorageQueryResult",
    "StorageStatusResult",
    "SubsystemDetailResult",
    "SubsystemListResult",
    "ValidateMacrosResult",
]
