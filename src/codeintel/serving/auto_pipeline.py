"""Auto-pipeline integration for serving layer.

This module provides automatic prerequisite pipeline execution for serving
operations. When enabled, the serving layer will automatically run the
necessary pipeline stages before executing an operation if the required
data is not already available.

The auto-pipeline feature is opt-in and controlled via environment variable.
This module also provides data-aware prerequisite checking that verifies
actual data existence rather than just run records.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.executor import BuildExecutor, BuildResult, ExecutorEnv
from codeintel.build.operations import get_targets_for_operation
from codeintel.build.plan import PlanGenerator
from codeintel.build.readiness import DatabaseReadinessView
from codeintel.build.registry import get_target_graph
from codeintel.build.resolver import BuildResolver
from codeintel.build.state import StateValidator
from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY
from codeintel.config.models import CliPathsInput, ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.serving.operations.catalog import get_operation
from codeintel.storage.tracking import PipelineRunRecord
from codeintel.storage.validation import table_has_rows_for_snapshot

if TYPE_CHECKING:
    from codeintel.config.datasets import DatasetContract
    from codeintel.config.serving_models import ServingConfig
    from codeintel.serving.mcp.backend import QueryBackend
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.tracking import PipelineRunTracking

LOG = logging.getLogger(__name__)

AUTO_PIPELINE_ENV = "CODEINTEL_AUTO_PIPELINE"
"""Environment variable to enable auto-pipeline."""


def is_auto_pipeline_enabled() -> bool:
    """Check if auto-pipeline is enabled via environment variable.

    Returns
    -------
    bool
        True if auto-pipeline is enabled, False otherwise.
    """
    value = os.environ.get(AUTO_PIPELINE_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def build_paths_for_serving(config: ServingConfig) -> BuildPaths:
    """Reconstruct BuildPaths from ServingConfig.

    Parameters
    ----------
    config
        Serving configuration with repo_root and db_path.

    Returns
    -------
    BuildPaths
        Build paths suitable for pipeline execution.
    """
    repo_root = config.repo_root or Path.cwd()
    db_path = config.db_path or (repo_root / ".codeintel" / "duckdb.db")

    paths_input = CliPathsInput(
        repo_root=repo_root,
        build_dir=repo_root / ".codeintel",
        db_path=db_path,
        document_output_dir=None,
    )
    return paths_input.to_build_paths()


def _has_successful_recent_run(
    runs: PipelineRunTracking,
    *,
    repo: str,
    commit: str,
) -> PipelineRunRecord | None:
    """Check if a recent successful run exists for the given repo/commit.

    Parameters
    ----------
    runs
        Pipeline run tracking instance.
    repo
        Repository slug.
    commit
        Commit SHA.

    Returns
    -------
    PipelineRunRecord | None
        The most recent successful run if found, None otherwise.
    """
    recent_runs = runs.fetch_recent_runs(limit=20)
    for run in recent_runs:
        if (
            run.repo == repo
            and run.commit == commit
            and run.status == "succeeded"
            and run.kind in {"full", "op_prereqs"}
        ):
            return run
    return None


def has_successful_prereq_run(
    runs: PipelineRunTracking,
    *,
    repo: str,
    commit: str,
    op_id: str,
) -> bool:
    """Check if prerequisites have already been run for an operation.

    This function looks for a recent successful pipeline run that would
    have produced the data needed for the specified operation.

    Parameters
    ----------
    runs
        Pipeline run tracking instance.
    repo
        Repository slug.
    commit
        Commit SHA.
    op_id
        Operation identifier.

    Returns
    -------
    bool
        True if prerequisites have already been run, False otherwise.
    """
    _ = op_id  # Future: could check requested_operation or requested_datasets
    existing_run = _has_successful_recent_run(runs, repo=repo, commit=commit)
    return existing_run is not None


# -----------------------------------------------------------------------------
# Data-Aware Prerequisite Checks
# -----------------------------------------------------------------------------


def dataset_has_rows_for_snapshot(
    gateway: StorageGateway,
    contract: DatasetContract,
    *,
    repo: str,
    commit: str,
) -> bool:
    """Check if a dataset has rows for the given repo/commit.

    Delegates to the storage layer for the actual database query.

    Parameters
    ----------
    gateway
        Storage gateway with database connection.
    contract
        Dataset contract with table_key and schema info.
    repo
        Repository slug.
    commit
        Commit SHA.

    Returns
    -------
    bool
        True if at least one row exists, False otherwise.
    """
    return table_has_rows_for_snapshot(gateway.con, contract, repo=repo, commit=commit)


def get_required_table_keys_for_operation(op_id: str) -> frozenset[str]:
    """Get all required table keys for an operation with transitive expansion.

    Uses the build system's target graph to resolve transitive dependencies.

    Parameters
    ----------
    op_id
        Operation identifier.

    Returns
    -------
    frozenset[str]
        Set of all required table keys including transitive dependencies.
    """
    op = get_operation(op_id)
    if op is None:
        return frozenset()

    # Get the targets required for this operation
    op_targets = get_targets_for_operation(op_id)
    if not op_targets.required_targets:
        return frozenset(op.required_datasets)

    # Get all tables from the targets and their transitive dependencies
    graph = get_target_graph()
    table_keys: set[str] = set()

    for target_name in op_targets.required_targets:
        if target_name not in graph:
            continue

        # Get target and all its dependencies
        target = graph.get(target_name)
        deps = graph.transitive_deps(target_name)

        # Add tables from this target
        table_keys.update(target.table_keys)

        # Add tables from all dependencies
        for dep_name in deps:
            dep_target = graph.get(dep_name)
            table_keys.update(dep_target.table_keys)

    return frozenset(table_keys)


def has_required_data_for_operation(
    gateway: StorageGateway,
    op_id: str,
    *,
    repo: str,
    commit: str,
) -> bool:
    """Check if all required data exists for an operation.

    This function performs data-aware checking by verifying that
    the actual data rows exist, not just that pipeline runs completed.

    Parameters
    ----------
    gateway
        Storage gateway with database connection.
    op_id
        Operation identifier.
    repo
        Repository slug.
    commit
        Commit SHA.

    Returns
    -------
    bool
        True if all required data exists, False otherwise.
    """
    expanded_tables = get_required_table_keys_for_operation(op_id)

    if not expanded_tables:
        # No declared datasets - cannot verify data-aware
        return False

    # Check each required dataset
    for table_key in expanded_tables:
        contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
        if contract is None:
            LOG.debug("has_required_data: unknown contract for %s", table_key)
            continue

        if not dataset_has_rows_for_snapshot(gateway, contract, repo=repo, commit=commit):
            LOG.debug("has_required_data: missing data in %s", table_key)
            return False

    return True


def operation_prereqs_satisfied(
    gateway: StorageGateway,
    op_id: str,
    *,
    repo: str,
    commit: str,
    snapshot: SnapshotRef | None = None,
) -> bool:
    """Check if prerequisites are satisfied using build system readiness.

    This function uses the build system's DatabaseReadinessView to check if
    all targets required by the operation are in a 'current' state.

    Parameters
    ----------
    gateway
        Storage gateway with connection and run tracking.
    op_id
        Operation identifier.
    repo
        Repository slug.
    commit
        Commit SHA.
    snapshot
        Optional snapshot reference. If not provided, one will be constructed.

    Returns
    -------
    bool
        True if prerequisites are satisfied, False otherwise.
    """
    # Get required targets for operation
    op_targets = get_targets_for_operation(op_id)
    if not op_targets.required_targets:
        # Operation has no declared requirements - check fallback
        op = get_operation(op_id)
        if op is not None and op.required_datasets:
            return has_required_data_for_operation(
                gateway,
                op_id,
                repo=repo,
                commit=commit,
            )
        # No requirements declared - consider satisfied
        return True

    # Construct snapshot if not provided
    if snapshot is None:
        snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=Path.cwd())

    # Create readiness view
    graph = get_target_graph()
    view = DatabaseReadinessView(graph, gateway, snapshot)

    # Check if all required targets are ready
    for target_name in op_targets.required_targets:
        if target_name not in view:
            LOG.debug("prereqs: unknown target %s for %s", target_name, op_id)
            continue
        readiness = view[target_name]
        if not readiness.is_ready:
            LOG.debug("prereqs: target %s not ready for %s", target_name, op_id)
            return False

    return True


# =============================================================================
# Error Diagnosis
# =============================================================================


@dataclass(frozen=True)
class PrerequisiteError:
    """Structured error for unmet prerequisites.

    Provides actionable information about why an operation cannot run
    and how to fix it.

    Attributes
    ----------
    op_id
        Operation that cannot run.
    missing_targets
        Targets that are not ready.
    bottleneck
        The ultimate blocker target (the root cause).
    fix_command
        CLI command to fix the issue.
    human_message
        Human-readable explanation.
    """

    op_id: str
    missing_targets: tuple[str, ...]
    bottleneck: str | None
    fix_command: str
    human_message: str


def diagnose_prereq_failure(
    gateway: StorageGateway,
    op_id: str,
    snapshot: SnapshotRef,
) -> PrerequisiteError:
    """Diagnose why prerequisites are not satisfied.

    Returns structured error information with an actionable fix command.
    Uses the build system's readiness view to determine the root cause.

    Parameters
    ----------
    gateway
        Storage gateway with connection and database access.
    op_id
        Operation identifier that failed prerequisite check.
    snapshot
        Repository snapshot reference.

    Returns
    -------
    PrerequisiteError
        Structured error with diagnosis and fix instructions.
    """
    graph = get_target_graph()
    view = DatabaseReadinessView(graph, gateway, snapshot)
    op_targets = get_targets_for_operation(op_id)

    missing: list[str] = []
    bottleneck: str | None = None

    for target_name in op_targets.required_targets:
        if target_name not in view:
            continue
        readiness = view[target_name]
        if not readiness.is_ready:
            missing.append(target_name)
            # Track the ultimate bottleneck (root cause)
            if readiness.ultimate_bottleneck:
                bottleneck = readiness.ultimate_bottleneck

    # Use bottleneck or first missing as fix target
    fix_target = bottleneck or (missing[0] if missing else None)
    fix_command = f"codeintel build run {fix_target}" if fix_target else "codeintel build run --all"

    # Build human message
    if missing:
        missing_str = ", ".join(sorted(missing))
        human_message = (
            f"Operation '{op_id}' requires data that hasn't been computed. "
            f"Missing targets: {missing_str}. "
            f"Run: {fix_command}"
        )
    else:
        human_message = (
            f"Operation '{op_id}' cannot run due to missing prerequisites. Run: {fix_command}"
        )

    return PrerequisiteError(
        op_id=op_id,
        missing_targets=tuple(sorted(missing)),
        bottleneck=bottleneck,
        fix_command=fix_command,
        human_message=human_message,
    )


# -----------------------------------------------------------------------------
# Debug Information
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetDebugInfo:
    """Debug information for a single dataset check.

    Attributes
    ----------
    table_key
        Dataset table key.
    name
        Human-readable dataset name.
    has_rows
        Whether the dataset has rows for the repo/commit.
    checked
        Whether this dataset was successfully checked.
    error
        Error message if check failed, None otherwise.
    """

    table_key: str
    name: str
    has_rows: bool
    checked: bool
    error: str | None = None


@dataclass(frozen=True)
class RunDebugInfo:
    """Debug information for a pipeline run.

    Attributes
    ----------
    run_id
        Pipeline run identifier.
    kind
        Run kind (full, op_prereqs, etc.).
    status
        Run status (succeeded, failed, etc.).
    started_at
        When the run started.
    completed_at
        When the run completed, if applicable.
    """

    run_id: str
    kind: str
    status: str
    started_at: datetime | None
    completed_at: datetime | None


@dataclass(frozen=True)
class PrereqDebugInfo:
    """Complete debug information for prerequisite checking.

    Attributes
    ----------
    op_id
        Operation identifier.
    repo
        Repository slug.
    commit
        Commit SHA.
    required_datasets
        Directly required dataset table keys.
    expanded_datasets
        All required datasets after transitive expansion.
    dataset_statuses
        Debug info for each dataset check.
    runs_considered
        Recent pipeline runs considered.
    data_satisfied
        Whether data-aware check passed.
    run_satisfied
        Whether run-based check passed.
    overall_satisfied
        Final determination of prerequisite satisfaction.
    """

    op_id: str
    repo: str
    commit: str
    required_datasets: tuple[str, ...]
    expanded_datasets: tuple[str, ...]
    dataset_statuses: tuple[DatasetDebugInfo, ...]
    runs_considered: tuple[RunDebugInfo, ...]
    data_satisfied: bool
    run_satisfied: bool
    overall_satisfied: bool


def build_prereq_debug_info(
    gateway: StorageGateway,
    op_id: str,
    *,
    repo: str,
    commit: str,
) -> PrereqDebugInfo:
    """Build complete debug information for prerequisite checking.

    This function collects all debugging data to understand why
    prerequisites are or are not satisfied for an operation.

    Parameters
    ----------
    gateway
        Storage gateway with connection and run tracking.
    op_id
        Operation identifier.
    repo
        Repository slug.
    commit
        Commit SHA.

    Returns
    -------
    PrereqDebugInfo
        Complete debug information.
    """
    op = get_operation(op_id)

    # Get required and expanded datasets
    required_datasets: tuple[str, ...] = ()
    if op is not None:
        required_datasets = op.required_datasets

    expanded_tables = get_required_table_keys_for_operation(op_id)

    # Check each dataset
    dataset_statuses: list[DatasetDebugInfo] = []
    for table_key in sorted(expanded_tables):
        contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
        if contract is None:
            dataset_statuses.append(
                DatasetDebugInfo(
                    table_key=table_key,
                    name="<unknown>",
                    has_rows=False,
                    checked=False,
                    error="Contract not found",
                )
            )
            continue

        try:
            has_rows = dataset_has_rows_for_snapshot(gateway, contract, repo=repo, commit=commit)
        except (RuntimeError, ValueError, OSError) as exc:
            dataset_statuses.append(
                DatasetDebugInfo(
                    table_key=table_key,
                    name=contract.name,
                    has_rows=False,
                    checked=False,
                    error=str(exc),
                )
            )
        else:
            dataset_statuses.append(
                DatasetDebugInfo(
                    table_key=table_key,
                    name=contract.name,
                    has_rows=has_rows,
                    checked=True,
                    error=None,
                )
            )

    # Get recent runs
    recent_runs = gateway.runs.fetch_recent_runs(limit=10)
    runs_considered = [
        RunDebugInfo(
            run_id=run.run_id,
            kind=run.kind,
            status=run.status,
            started_at=run.started_at,
            completed_at=run.completed_at,
        )
        for run in recent_runs
        if run.repo == repo and run.commit == commit
    ]

    # Compute satisfaction flags
    data_satisfied = has_required_data_for_operation(gateway, op_id, repo=repo, commit=commit)
    run_satisfied = has_successful_prereq_run(gateway.runs, repo=repo, commit=commit, op_id=op_id)
    overall_satisfied = operation_prereqs_satisfied(gateway, op_id, repo=repo, commit=commit)

    return PrereqDebugInfo(
        op_id=op_id,
        repo=repo,
        commit=commit,
        required_datasets=required_datasets,
        expanded_datasets=tuple(sorted(expanded_tables)),
        dataset_statuses=tuple(dataset_statuses),
        runs_considered=tuple(runs_considered),
        data_satisfied=data_satisfied,
        run_satisfied=run_satisfied,
        overall_satisfied=overall_satisfied,
    )


def should_run_auto_pipeline(
    config: ServingConfig,
    backend: QueryBackend,
) -> tuple[bool, StorageGateway | None, str]:
    """Check if auto-pipeline should run and return gateway if available.

    Consolidates gate checks for both HTTP and MCP auto-pipeline paths.

    Parameters
    ----------
    config
        Serving configuration.
    backend
        Query backend instance.

    Returns
    -------
    tuple[bool, StorageGateway | None, str]
        A tuple of (should_run, gateway, skip_reason).
        If should_run is False, skip_reason explains why.
    """
    # Only run for local_db mode
    if config.mode != "local_db":
        return False, None, f"mode={config.mode} is not local_db"

    # Check if auto-pipeline is enabled
    if not is_auto_pipeline_enabled():
        return False, None, "not enabled"

    # Get gateway from backend (DuckDBBackend has it)
    gateway: StorageGateway | None = getattr(backend, "gateway", None)
    if gateway is None:
        return False, None, "backend has no gateway"

    return True, gateway, ""


def run_operation_prereqs(
    *,
    op_id: str,
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    paths: BuildPaths,
    tools: ToolsConfig,
) -> BuildResult | None:
    """Execute prerequisites for an operation using the build system.

    Public API for CLI and other callers that already have the required
    configuration objects.

    Parameters
    ----------
    op_id
        Operation identifier.
    gateway
        Storage gateway with database connection.
    snapshot
        Repository snapshot reference.
    paths
        Build paths configuration.
    tools
        Tools configuration.

    Returns
    -------
    BuildResult | None
        The build result if executed, None if all targets are current.
    """
    # Get required targets for the operation
    op_targets = get_targets_for_operation(op_id)
    if not op_targets.required_targets:
        LOG.debug("run_operation_prereqs: op=%s has no required targets", op_id)
        return None

    # Convert to list for the build system
    goal_targets = list(op_targets.required_targets)

    # Get target graph and validate state
    graph = get_target_graph()
    validator = StateValidator(graph, gateway, snapshot)
    state = validator.validate()

    # Resolve minimal work needed
    resolver = BuildResolver(graph, state)
    resolution = resolver.resolve(goals=goal_targets)

    # If nothing to compute, return early
    if not resolution.to_compute:
        LOG.debug("run_operation_prereqs: all targets current for op=%s", op_id)
        return None

    # Generate build plan
    planner = PlanGenerator(graph)
    plan = planner.generate(resolution)

    LOG.info(
        "run_operation_prereqs executing op=%s targets=%s",
        op_id,
        resolution.to_compute,
    )

    # Execute via build system
    env = ExecutorEnv(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        tools=tools,
    )
    executor = BuildExecutor(graph=graph, env=env)
    return executor.execute(plan)


def _run_prereqs_build(
    *,
    op_id: str,
    config: ServingConfig,
    gateway: StorageGateway,
) -> BuildResult | None:
    """Execute prerequisites for an operation using the build system.

    Internal helper for serving layer that uses ServingConfig.

    Parameters
    ----------
    op_id
        Operation identifier.
    config
        Serving configuration.
    gateway
        Storage gateway with database connection.

    Returns
    -------
    BuildResult | None
        The build result if executed, None if all targets are current.
    """
    # Build snapshot and paths from config
    paths = build_paths_for_serving(config)
    snapshot = SnapshotRef(
        repo=config.repo,
        commit=config.commit,
        repo_root=config.repo_root or Path.cwd(),
    )
    tools = ToolsConfig.default()

    return run_operation_prereqs(
        op_id=op_id,
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        tools=tools,
    )


def ensure_prereqs_for_http(
    *,
    op_id: str,
    config: ServingConfig,
    backend: QueryBackend,
) -> BuildResult | None:
    """Ensure prerequisites are run for an HTTP operation if needed.

    This function is called before serving an HTTP request. If auto-pipeline
    is enabled and no previous successful run exists, it will execute the
    necessary pipeline stages.

    Parameters
    ----------
    op_id
        Operation identifier.
    config
        Serving configuration.
    backend
        Query backend (must be DuckDBBackend for local_db mode).

    Returns
    -------
    BuildResult | None
        The build result if a run was executed, None if skipped.
    """
    should_run, gateway, skip_reason = should_run_auto_pipeline(config, backend)
    if not should_run or gateway is None:
        LOG.debug("auto_pipeline skipped: %s", skip_reason)
        return None

    # Check if prereqs have already been satisfied
    if has_successful_prereq_run(gateway.runs, repo=config.repo, commit=config.commit, op_id=op_id):
        LOG.debug("auto_pipeline skipped: prereqs already satisfied for %s", op_id)
        return None

    return _run_prereqs_build(op_id=op_id, config=config, gateway=gateway)


def ensure_prereqs_for_mcp(
    *,
    op_id: str,
    config: ServingConfig,
    backend: QueryBackend,
) -> BuildResult | None:
    """Ensure prerequisites are run for an MCP tool invocation if needed.

    This function is called before executing an MCP tool. If auto-pipeline
    is enabled and no previous successful run exists, it will execute the
    necessary pipeline stages.

    Parameters
    ----------
    op_id
        Operation identifier.
    config
        Serving configuration.
    backend
        Query backend (must be DuckDBBackend for local_db mode).

    Returns
    -------
    BuildResult | None
        The build result if a run was executed, None if skipped.
    """
    should_run, gateway, skip_reason = should_run_auto_pipeline(config, backend)
    if not should_run or gateway is None:
        LOG.debug("auto_pipeline skipped: %s", skip_reason)
        return None

    # Check if prereqs have already been satisfied
    if has_successful_prereq_run(gateway.runs, repo=config.repo, commit=config.commit, op_id=op_id):
        LOG.debug("auto_pipeline skipped: prereqs already satisfied for %s", op_id)
        return None

    return _run_prereqs_build(op_id=op_id, config=config, gateway=gateway)


__all__ = [
    "AUTO_PIPELINE_ENV",
    "DatasetDebugInfo",
    "PrereqDebugInfo",
    "PrerequisiteError",
    "RunDebugInfo",
    "build_paths_for_serving",
    "build_prereq_debug_info",
    "dataset_has_rows_for_snapshot",
    "diagnose_prereq_failure",
    "ensure_prereqs_for_http",
    "ensure_prereqs_for_mcp",
    "get_required_table_keys_for_operation",
    "has_required_data_for_operation",
    "has_successful_prereq_run",
    "is_auto_pipeline_enabled",
    "operation_prereqs_satisfied",
    "run_operation_prereqs",
    "should_run_auto_pipeline",
]
