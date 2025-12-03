"""Auto-pipeline integration for serving layer.

This module provides automatic prerequisite pipeline execution for serving
operations. When enabled, the serving layer will automatically run the
necessary pipeline stages before executing an operation if the required
data is not already available.

The auto-pipeline feature is opt-in and controlled via environment variable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.models import CliPathsInput, ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.pipeline.op_planner import (
    OperationPrereqOptions,
    ensure_prerequisites_for_operation,
)
from codeintel.runtime import TriggerKind
from codeintel.storage.run_tracking import PipelineRunRecord

if TYPE_CHECKING:
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
    "build_paths_for_serving",
    "ensure_prereqs_for_http",
    "ensure_prereqs_for_mcp",
    "has_successful_prereq_run",
    "is_auto_pipeline_enabled",
    "should_run_auto_pipeline",
]
