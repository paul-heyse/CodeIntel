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

from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY
from codeintel.config.models import CliPathsInput, ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.pipeline.op_planner import (
    OperationPrereqOptions,
    build_prereq_summary,
    ensure_prerequisites_for_operation,
)
from codeintel.runtime import TriggerKind
from codeintel.serving.operations.catalog import get_operation
from codeintel.storage.data_checks import table_has_rows_for_snapshot
from codeintel.storage.run_tracking import PipelineRunRecord

if TYPE_CHECKING:
    from codeintel.config.datasets import DatasetContract
    from codeintel.config.serving_models import ServingConfig
    from codeintel.serving.mcp.backend import QueryBackend
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.run_tracking import PipelineRunTracking

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

    # Use the op_planner's summary to get expanded tables
    # We need a dummy snapshot for the call
    dummy_snapshot = SnapshotRef(repo="", commit="", repo_root=Path.cwd())
    summary = build_prereq_summary(op_id, dummy_snapshot)
    return summary.expanded_tables


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
) -> bool:
    """Check if prerequisites are satisfied using data-aware or run-based logic.

    This function implements the combined check:
    1. If operation has required_datasets, use data-aware checking
    2. Otherwise, fall back to run-based checking

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
    bool
        True if prerequisites are satisfied, False otherwise.
    """
    op = get_operation(op_id)

    # If operation has declared datasets, use data-aware check
    if op is not None and op.required_datasets:
        return has_required_data_for_operation(
            gateway,
            op_id,
            repo=repo,
            commit=commit,
        )

    # Fall back to run-based check
    return has_successful_prereq_run(
        gateway.runs,
        repo=repo,
        commit=commit,
        op_id=op_id,
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


def _run_prereqs(
    *,
    op_id: str,
    config: ServingConfig,
    gateway: StorageGateway,
    trigger: TriggerKind,
) -> PipelineRunRecord | None:
    """Execute prerequisites for an operation.

    Internal helper used by both HTTP and MCP auto-pipeline functions.

    Parameters
    ----------
    op_id
        Operation identifier.
    config
        Serving configuration.
    gateway
        Storage gateway with database connection.
    trigger
        Trigger kind for the pipeline run.

    Returns
    -------
    PipelineRunRecord | None
        The pipeline run record if executed, None if skipped.
    """
    # Check for existing successful run
    if has_successful_prereq_run(
        gateway.runs,
        repo=config.repo,
        commit=config.commit,
        op_id=op_id,
    ):
        LOG.debug("auto_pipeline skipped: found existing successful run")
        return None

    # Build prerequisite options
    paths = build_paths_for_serving(config)
    snapshot = SnapshotRef(
        repo=config.repo,
        commit=config.commit,
        repo_root=config.repo_root or Path.cwd(),
    )
    tools = ToolsConfig.default()

    prereq_options = OperationPrereqOptions(
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=tools,
        include_analytics=True,
        trigger=trigger,
    )

    LOG.info("auto_pipeline executing op=%s trigger=%s", op_id, trigger)
    return ensure_prerequisites_for_operation(op_id=op_id, options=prereq_options)


def ensure_prereqs_for_http(
    *,
    op_id: str,
    config: ServingConfig,
    backend: QueryBackend,
) -> PipelineRunRecord | None:
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
    PipelineRunRecord | None
        The pipeline run record if a run was executed, None if skipped.
    """
    should_run, gateway, skip_reason = should_run_auto_pipeline(config, backend)
    if not should_run or gateway is None:
        LOG.debug("auto_pipeline skipped: %s", skip_reason)
        return None

    return _run_prereqs(op_id=op_id, config=config, gateway=gateway, trigger="http")


def ensure_prereqs_for_mcp(
    *,
    op_id: str,
    config: ServingConfig,
    backend: QueryBackend,
) -> PipelineRunRecord | None:
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
    PipelineRunRecord | None
        The pipeline run record if a run was executed, None if skipped.
    """
    should_run, gateway, skip_reason = should_run_auto_pipeline(config, backend)
    if not should_run or gateway is None:
        LOG.debug("auto_pipeline skipped: %s", skip_reason)
        return None

    return _run_prereqs(op_id=op_id, config=config, gateway=gateway, trigger="mcp")


__all__ = [
    "AUTO_PIPELINE_ENV",
    "DatasetDebugInfo",
    "PrereqDebugInfo",
    "RunDebugInfo",
    "build_paths_for_serving",
    "build_prereq_debug_info",
    "dataset_has_rows_for_snapshot",
    "ensure_prereqs_for_http",
    "ensure_prereqs_for_mcp",
    "get_required_table_keys_for_operation",
    "has_required_data_for_operation",
    "has_successful_prereq_run",
    "is_auto_pipeline_enabled",
    "operation_prereqs_satisfied",
    "should_run_auto_pipeline",
]
