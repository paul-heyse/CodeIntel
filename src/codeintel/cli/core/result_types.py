"""Result type definitions for CLI handlers.

Each handler that returns structured data should have a corresponding
result type defined here. Generic result types (ListResult, ActionResult, etc.)
are provided for common patterns.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import msgspec

from codeintel.cli.core.results import ResultBase

if TYPE_CHECKING:
    from codeintel.build.planning.model import PlanTargetEntry
    from codeintel.cli.core.results import SerializableResult

T = TypeVar("T")


class ListResult[T](ResultBase):
    """Generic result type for list operations.

    Use this for any command that returns a list of items with a count.
    The base ResultBase class provides to_dict() that handles nested
    serialization.

    Parameters
    ----------
    items
        List of result items.
    count
        Total number of items.

    Examples
    --------
    >>> class UserInfo(ResultBase):
    ...     name: str
    ...     active: bool
    >>> result = ListResult(items=[UserInfo(name="alice", active=True)], count=1)
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


class JobInfo(ResultBase):
    """Information about a single background job."""

    job_id: str
    operation_id: str
    status: str
    created_at: str | None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class JobOutputResult(ResultBase):
    """Result from retrieving job output."""

    job_id: str
    has_output: bool
    output: dict[str, Any] | None = None


class GraphTargetInfo(ResultBase):
    """Information about a single graph target."""

    name: str
    description: str
    dependencies: list[str]
    tables: list[str]


class GraphTargetsResult(ResultBase):
    """Result from listing graph targets."""

    targets: list[GraphTargetInfo]
    count: int


class GraphPlanStage(ResultBase):
    """A stage in the graph execution plan."""

    stage: int
    targets: list[str]


class GraphPlanResult(ResultBase):
    """Result from planning graph target execution."""

    stages: list[GraphPlanStage]
    total_targets: int


class ActionResult(ResultBase):
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


class StatusResult(ResultBase):
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


class ExportResult(ResultBase):
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


class InferabilityInfo(ResultBase):
    """Result type for schema inferability inventory records.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    status
        Inferability status ("inferable" or "non_inferable").
    target_name
        Target name that produces the table output.
    saver_node
        Saver node name attached to the output.
    sink
        Materialization sink name.
    compute_node
        Compute node name driving the table output.
    reason
        Optional explanation when non-inferable.
    qparams
        Sorted list of q__ inputs referenced by the compute node.
    requires_env
        Whether inference requires BuildEnv inputs.
    requires_catalog
        Whether inference requires catalog inputs.
    """

    table_key: str
    status: str
    target_name: str
    saver_node: str
    sink: str
    compute_node: str | None = None
    reason: str | None = None
    qparams: tuple[str, ...] | None = None
    requires_env: bool | None = None
    requires_catalog: bool | None = None


class StorageDatabaseExportResult(ResultBase):
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


class StorageDatabaseImportResult(ResultBase):
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


class OperationListResult(ResultBase):
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


class OperationCallResult(ResultBase):
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


class DatasetSummary(ResultBase):
    """Summary metadata for dataset listings."""

    name: str
    table_key: str
    description: str | None = None
    owner_package: str | None = None
    capabilities: dict[str, bool] = msgspec.field(default_factory=dict)


class DatasetListResult(ResultBase):
    """Result from dataset list command.

    Parameters
    ----------
    datasets
        List of dataset summaries.
    count
        Total number of datasets.
    """

    datasets: list[DatasetSummary]
    count: int


class DatasetDescribeResult(ResultBase):
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
    upstream_dependencies: list[str] | None = None


class DatasetVerifyResult(ResultBase):
    """Result from dataset verify command.

    Parameters
    ----------
    verified
        Whether verification passed.
    issues
        List of issues found (empty if verified).
    """

    verified: bool
    issues: list[str] = msgspec.field(default_factory=list)


class DatasetInfoResult(ResultBase):
    """Result from dataset info command.

    Provides comprehensive schema information for a dataset including
    column definitions, metadata, and JSON schema representation.

    Parameters
    ----------
    name
        Dataset table key (e.g., "analytics.function_types").
    columns
        Tuple of column names in the schema.
    metadata
        Dataset metadata as a dictionary.
    json_schema
        JSON Schema representation of the dataset.
    has_table_schema
        Whether a table schema is registered for this dataset.
    """

    name: str
    columns: tuple[str, ...]
    column_count: int
    metadata: dict[str, object]
    json_schema: dict[str, object]
    has_table_schema: bool = True


class DatasetFlowResult(ResultBase):
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
    producer_count: int
    consumer_count: int


class DatasetConstraintsResult(ResultBase):
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


class ExportMode(StrEnum):
    """Execution mode for docs export operations."""

    BUILD_SYSTEM = "build_system"
    DIRECT = "direct"
    DRY_RUN = "dry_run"


class DocsExportResult(ResultBase):
    """Result from docs export operation."""

    status: str
    validation: str
    datasets: list[str] | None
    schemas: list[str] | None
    mode: ExportMode
    macro_requirement: str


class DocsValidateResult(ResultBase):
    """Result from docs validation operation."""

    passed: bool
    issues: list[str]


class BuildStatusResult(ResultBase):
    """Result from build status command.

    Parameters
    ----------
    targets
        List of target status dictionaries.
    current_count
        Number of current targets.
    missing_count
        Number of missing targets.
    blocked_count
        Number of blocked targets.
    current
        List of targets that are up to date.
    missing
        List of missing targets.
    blocked
        List of blocked targets, optionally annotated with reasons.
    """

    targets: list[dict[str, Any]]
    current_count: int
    missing_count: int
    blocked_count: int
    current: list[str] = msgspec.field(default_factory=list)
    missing: list[str] = msgspec.field(default_factory=list)
    blocked: list[str] = msgspec.field(default_factory=list)


class BuildBootstrapSuiteResult(ResultBase):
    """Result from build bootstrap index suite command.

    Parameters
    ----------
    suite_manifest_path
        Path to the suite manifest written to disk.
    targets
        Target names executed for the suite.
    dataset_manifest_paths
        Mapping of table key to dataset manifest path.
    missing_targets
        Requested P0 targets missing from the catalog.
    missing_manifests
        Table keys missing dataset manifests after execution.
    """

    suite_manifest_path: str
    targets: list[str]
    dataset_manifest_paths: dict[str, str]
    missing_targets: list[str] = msgspec.field(default_factory=list)
    missing_manifests: list[str] = msgspec.field(default_factory=list)


class BuildRunResult(ResultBase):
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
    target_tags
        Optional tag metadata keyed by target name.
    """

    executed: list[str]
    skipped: list[str]
    failed: list[str]
    duration_seconds: float
    cache: dict[str, object] | None = None
    target_tags: dict[str, dict[str, object]] | None = None


class BuildPlanResult(ResultBase):
    """Result from build plan command.

    Parameters
    ----------
    requested
        List of requested target names.
    closure
        List of target names in dependency closure.
    entries
        List of plan entries with status/reason metadata.
    to_compute
        List of target names that will be computed.
    to_reuse
        List of target names predicted to reuse cached results.
    blocked
        List of target names that are blocked.
    compute_count
        Count of targets marked for computation.
    reuse_count
        Count of targets predicted to reuse cache.
    blocked_count
        Count of targets blocked from execution.
    """

    requested: list[str]
    closure: list[str]
    entries: list[PlanTargetEntry]
    to_compute: list[str]
    to_reuse: list[str]
    blocked: list[str]
    compute_count: int = 0
    reuse_count: int = 0
    blocked_count: int = 0

    def __post_init__(self) -> None:
        """Compute summary counts for plan outputs."""
        msgspec.structs.force_setattr(self, "compute_count", len(self.to_compute))
        msgspec.structs.force_setattr(self, "reuse_count", len(self.to_reuse))
        msgspec.structs.force_setattr(self, "blocked_count", len(self.blocked))


class BuildExplainResult(ResultBase):
    """Result from build explain command.

    Parameters
    ----------
    target
        Target name being explained.
    predicted_action
        Predicted plan action (compute, reuse, blocked).
    block_reasons
        Reasons for blocking, if any.
    dependencies
        List of dependencies for the target.
    reads
        Table keys read by the target.
    writes_tables
        Table keys written by the target.
    writes_artifacts
        Artifact names written by the target.
    cache_hit_ratio
        Cache hit ratio for the target's node cone, if available.
    miss_nodes
        Cache miss node list when node details are enabled.
    summary
        Human-readable summary of the plan entry.
    """

    target: str
    predicted_action: str
    block_reasons: list[str]
    dependencies: list[str]
    reads: list[str]
    writes_tables: list[str]
    writes_artifacts: list[str]
    __include_none_fields__: ClassVar[frozenset[str]] = frozenset({"cache_hit_ratio"})

    cache_hit_ratio: float | None
    miss_nodes: list[str]
    summary: str
    io_surface: dict[str, object] | None = None


class BuildGraphResult(ResultBase):
    """Result type for build graph command."""

    dag_json: str
    node_count: int
    edge_count: int


class SubsystemListResult(ResultBase):
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


class ConfigShowResult(ResultBase):
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


class DryRunStep(ResultBase):
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


class DryRunResult(ResultBase):
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
    warnings: list[str] | None = None

    def __post_init__(self) -> None:
        """Normalize empty warning lists to omitted values."""
        if self.warnings is not None and len(self.warnings) == 0:
            msgspec.structs.force_setattr(self, "warnings", None)


class BuildTargetInfo(ResultBase):
    """Information about a build target.

    Parameters
    ----------
    name
        Target name.
    status
        Current status (current, missing, blocked).
    last_run
        Timestamp of last successful run.
    dependencies
        List of dependency target names.
    outputs
        List of output paths.
    """

    name: str
    status: str
    __include_none_fields__: ClassVar[frozenset[str]] = frozenset({"last_run"})

    last_run: str | None
    dependencies: list[str] = msgspec.field(default_factory=list)
    outputs: list[str] = msgspec.field(default_factory=list)


class BuildExecutionResult(ResultBase):
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


class BuildAssetsResult(ResultBase):
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


class BuildLineageResult(ResultBase):
    """Result from build lineage command."""

    asset: str
    asset_kind: str
    root_version_hash: str
    direction: str
    depth: int
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    format: str = "json"


class BuildPromoteResult(ResultBase):
    """Result from build promote command."""

    __include_none_fields__: ClassVar[frozenset[str]] = frozenset({"note"})

    asset: str
    asset_kind: str
    alias: str
    version_hash: str
    note: str | None = None
    format: str = "json"


class BuildResolveResult(ResultBase):
    """Result from build resolve command."""

    asset: str
    asset_kind: str
    alias: str
    version_hash: str
    format: str = "json"


class BuildDiffResult(ResultBase):
    """Result from build diff command."""

    __result_key_map__: ClassVar[dict[str, str]] = {"from_spec": "from", "to_spec": "to"}

    asset: str
    asset_kind: str
    from_spec: str
    to_spec: str
    from_version_hash: str
    to_version_hash: str
    diffs: dict[str, Any]
    cached: bool
    format: str = "json"


class BuildImpactResult(ResultBase):
    """Result of build impact analysis."""

    source_kind: str
    source_key: str
    source_version: str | None
    impacted_assets: list[dict[str, Any]]
    impacted_targets: list[str]
    format: str = "json"


class BuildHistoryResult(ResultBase):
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

    runs: list[SerializableResult]
    count: int
    targets: list[dict[str, Any]] | None = None


class BuildPublishSnapshotResult(ResultBase):
    """Result from build publish-serving-snapshot command."""

    run_id: str
    published_at: str
    snapshot_manifest_path: str
    snapshot_db_path: str
    semantic_registry_path: str
    schema_manifest_path: str
    buildspec_path: str
    semantic_layer_version: str
    dataset_count: int


class GraphStatsResult(ResultBase):
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


class GraphQueryResult(ResultBase):
    """Result from a graph query.

    Parameters
    ----------
    nodes
        Matching nodes with their attributes.
    edges
        Edges between matching nodes.
    query
        The query that was executed.
    node_count
        Number of nodes returned.
    edge_count
        Number of edges returned.
    """

    nodes: list[dict[str, object]]
    edges: list[dict[str, object]]
    query: str
    node_count: int = 0
    edge_count: int = 0

    def __post_init__(self) -> None:
        """Compute node and edge counts."""
        msgspec.structs.force_setattr(self, "node_count", len(self.nodes))
        msgspec.structs.force_setattr(self, "edge_count", len(self.edges))


class DocsStatusResult(ResultBase):
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
    __include_none_fields__: ClassVar[frozenset[str]] = frozenset({"last_generated"})

    last_generated: str | None


class DocsGenerateResult(ResultBase):
    """Result from docs generate command.

    Parameters
    ----------
    generated
        List of generated file paths.
    skipped
        List of skipped file paths.
    errors
        List of error messages.
    generated_count
        Number of generated files.
    error_count
        Number of errors encountered.
    """

    generated: list[str]
    skipped: list[str]
    errors: list[str]
    generated_count: int = 0
    error_count: int = 0

    def __post_init__(self) -> None:
        """Compute generated and error counts."""
        msgspec.structs.force_setattr(self, "generated_count", len(self.generated))
        msgspec.structs.force_setattr(self, "error_count", len(self.errors))


class HistoryListResult(ResultBase):
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


class HistoryDetailResult(ResultBase):
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
    details: dict[str, Any] = msgspec.field(default_factory=dict)


class IdeStatusResult(ResultBase):
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

    __include_none_fields__: ClassVar[frozenset[str]] = frozenset({"ide_type", "workspace_path"})

    connected: bool
    ide_type: str | None
    workspace_path: str | None
    extensions: list[str] = msgspec.field(default_factory=list)


class IdeConfigResult(ResultBase):
    """Result from ide config command.

    Parameters
    ----------
    settings
        IDE settings dictionary.
    path
        Settings file path.
    """

    __include_none_fields__: ClassVar[frozenset[str]] = frozenset({"path"})

    settings: dict[str, Any]
    path: str | None


class SubsystemDetailResult(ResultBase):
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
    module_count
        Count of modules in the subsystem.
    """

    name: str
    __include_none_fields__: ClassVar[frozenset[str]] = frozenset({"description"})

    description: str | None
    modules: list[str]
    dependencies: list[str]
    metrics: dict[str, Any] = msgspec.field(default_factory=dict)
    module_count: int = 0

    def __post_init__(self) -> None:
        """Compute module count."""
        msgspec.structs.force_setattr(self, "module_count", len(self.modules))


class StorageStatusResult(ResultBase):
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

    __include_none_fields__: ClassVar[frozenset[str]] = frozenset({"database_path"})

    connected: bool
    database_path: str | None
    table_count: int
    size_bytes: int


class StorageQueryResult(ResultBase):
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


class DatasetLintResult(ResultBase):
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


class DatasetSnapshotResult(ResultBase):
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


class DatasetDiffResult(ResultBase):
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


class DatasetParquetMigrationResult(ResultBase):
    """Result from dataset parquet migration command.

    Parameters
    ----------
    dataset_root_dir
        Dataset root directory where snapshots were written.
    snapshot_id
        Snapshot identifier used for the parquet datasets.
    exported
        Table keys written to parquet.
    skipped
        Table keys skipped during migration.
    """

    dataset_root_dir: str
    snapshot_id: str
    exported: list[str]
    skipped: list[str]


class DatasetScaffoldResult(ResultBase):
    """Result from dataset scaffold command.

    Parameters
    ----------
    dataset
        Dataset name or table key scaffolded.
    status
        Scaffold status ("created" or "dry_run").
    registry_check
        Registry check mode ("enabled" or "disabled").
    """

    dataset: str
    status: str
    registry_check: str


class ValidateMacrosResult(ResultBase):
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


class ProfileStorageResult(ResultBase):
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


class CacheLogIngestSummary(ResultBase):
    """Result from ingesting cache JSONL logs into DuckDB.

    Parameters
    ----------
    db_path
        Path to the DuckDB database.
    cache_dir
        Cache directory used for discovery (if provided).
    inserted_events
        Number of events inserted into DuckDB.
    run_ids
        Run identifiers observed in the logs.
    jsonl_files
        JSONL files that were ingested.
    """

    __include_none_fields__: ClassVar[frozenset[str]] = frozenset({"cache_dir"})

    db_path: str
    cache_dir: str | None
    inserted_events: int
    run_ids: list[str]
    jsonl_files: list[str]


class ServeStartResult(ResultBase):
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

    __include_none_fields__: ClassVar[frozenset[str]] = frozenset({"host", "port"})

    server_type: str
    host: str | None
    port: int | None
    auto_pipeline: bool
    repo: str
    commit: str
    db_path: str


class HealthCheckResult(ResultBase):
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


class PluginModuleInfo(ResultBase):
    """Module metadata for a plugin pack."""

    import_path: str
    file_path: str | None
    content_hash: str | None


class PluginPackInfo(ResultBase):
    """Summary metadata for a plugin pack."""

    name: str
    version: str
    enabled: bool
    default_enabled: bool
    modules: list[str]
    requires_codeintel: str
    config_namespace: str | None
    dist_name: str | None
    dist_version: str | None
    capabilities: list[str]


class PluginPackDetail(ResultBase):
    """Detailed metadata for a plugin pack."""

    name: str
    version: str
    enabled: bool
    default_enabled: bool
    modules: list[PluginModuleInfo]
    requires_codeintel: str
    config_namespace: str | None
    dist_name: str | None
    dist_version: str | None
    capabilities: list[str]
    entry_point: str | None


class TargetOriginInfo(ResultBase):
    """Origin metadata for a runtime target."""

    target: str
    domain: str
    anchor_module: str | None
    origin: str | None
    plugin_name: str | None
    tags: dict[str, object] | None = None


class TargetOriginListResult(ResultBase):
    """Result from listing runtime targets with provenance."""

    targets: list[TargetOriginInfo]
    count: int


__all__ = [
    "ActionResult",
    "BuildAssetsResult",
    "BuildBootstrapSuiteResult",
    "BuildExecutionResult",
    "BuildExplainResult",
    "BuildGraphResult",
    "BuildHistoryResult",
    "BuildImpactResult",
    "BuildPlanResult",
    "BuildPublishSnapshotResult",
    "BuildRunResult",
    "BuildStatusResult",
    "BuildTargetInfo",
    "CacheLogIngestSummary",
    "ConfigShowResult",
    "DatasetConstraintsResult",
    "DatasetDescribeResult",
    "DatasetDiffResult",
    "DatasetFlowResult",
    "DatasetInfoResult",
    "DatasetLintResult",
    "DatasetListResult",
    "DatasetParquetMigrationResult",
    "DatasetScaffoldResult",
    "DatasetSnapshotResult",
    "DatasetSummary",
    "DatasetVerifyResult",
    "DocsExportResult",
    "DocsGenerateResult",
    "DocsStatusResult",
    "DocsValidateResult",
    "DryRunResult",
    "DryRunStep",
    "ExportMode",
    "ExportResult",
    "GraphPlanResult",
    "GraphPlanStage",
    "GraphQueryResult",
    "GraphStatsResult",
    "GraphTargetInfo",
    "GraphTargetsResult",
    "HealthCheckResult",
    "HistoryDetailResult",
    "HistoryListResult",
    "IdeConfigResult",
    "IdeStatusResult",
    "JobInfo",
    "JobOutputResult",
    "ListResult",
    "OperationCallResult",
    "OperationListResult",
    "PluginModuleInfo",
    "PluginPackDetail",
    "PluginPackInfo",
    "ProfileStorageResult",
    "ServeStartResult",
    "StatusResult",
    "StorageDatabaseExportResult",
    "StorageDatabaseImportResult",
    "StorageQueryResult",
    "StorageStatusResult",
    "SubsystemDetailResult",
    "SubsystemListResult",
    "TargetOriginInfo",
    "TargetOriginListResult",
    "ValidateMacrosResult",
]
