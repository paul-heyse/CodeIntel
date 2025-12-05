"""Pipeline runner utilities for history and export operations.

This module provides execution utilities for specific pipeline operations:

- History timeseries analysis across commits
- Export of Parquet/JSONL artifacts

For general pipeline execution, use :func:`codeintel.pipeline.run_pipeline`
with a :class:`PipelineSpec`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.config import ConfigBuilder, SnapshotRef
from codeintel.config.primitives import BuildPaths
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.pipeline.export.export_jsonl import ExportCallOptions
from codeintel.pipeline.export.runner import ExportOptions, ExportRunner, run_validated_exports
from codeintel.serving.backend.datasets import validate_dataset_registry
from codeintel.storage.gateway import (
    StorageConfig,
    StorageGateway,
    build_snapshot_gateway_resolver,
)
from codeintel.storage.gateway_cache import get_gateway
from codeintel.storage.views import create_all_views

log = logging.getLogger(__name__)

DEFAULT_BUILD_SUBDIR = Path("build")


# -----------------------------------------------------------------------------
# History Timeseries
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryTimeseriesParams:
    """Parameters for history timeseries execution.

    Attributes
    ----------
    repo_root
        Path to the repository root.
    repo
        Repository identifier.
    commits
        Tuple of commit SHAs to process.
    history_db_dir
        Directory containing per-commit DuckDB snapshots.
    db_path
        Path to output database.
    runner
        Optional tool runner for git operations.
    """

    repo_root: Path
    repo: str
    commits: tuple[str, ...]
    history_db_dir: Path
    db_path: Path
    runner: ToolRunner | None = None


def run_history_timeseries(params: HistoryTimeseriesParams) -> None:
    """Execute history timeseries analytics across provided commits.

    Parameters
    ----------
    params
        History timeseries execution parameters.
    """
    snapshot = SnapshotRef(repo_root=params.repo_root, repo=params.repo, commit=params.commits[0])
    paths = BuildPaths.from_layout(
        repo_root=params.repo_root,
        build_dir=params.repo_root / DEFAULT_BUILD_SUBDIR,
        db_path=params.db_path,
    )
    builder = ConfigBuilder.from_primitives(snapshot=snapshot, paths=paths)
    cfg = builder.history_timeseries(commits=params.commits)
    # Note: Don't use history_db_path here - the snapshot_resolver handles
    # loading individual snapshot DBs from the directory. attach_history is
    # for attaching a single history DB file, not a directory of snapshots.
    gateway = get_gateway(StorageConfig.for_ingest(params.db_path))
    snapshot_resolver = build_snapshot_gateway_resolver(
        db_dir=params.history_db_dir,
        repo=params.repo,
        primary_gateway=gateway,
    )
    compute_history_timeseries_gateways(
        gateway,
        cfg,
        snapshot_resolver,
        runner=params.runner,
    )


# -----------------------------------------------------------------------------
# Export Docs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportHooks:
    """Override hooks for export docs execution.

    Attributes
    ----------
    validator
        Function to validate dataset registry.
    export_runner
        Function to run exports.
    gateway_factory
        Function to create storage gateway.
    create_views
        Function to create database views.
    """

    validator: Callable[[StorageGateway], None] = validate_dataset_registry
    export_runner: ExportRunner = run_validated_exports
    gateway_factory: Callable[[Path], StorageGateway] = lambda db_path: get_gateway(
        StorageConfig.for_ingest(db_path)
    )
    create_views: Callable[[Any], None] = create_all_views


def run_export_docs(
    *,
    db_path: Path,
    document_output_dir: Path,
    options: ExportOptions | None = None,
    hooks: ExportHooks | None = None,
) -> None:
    """Create views and export Parquet/JSONL artifacts.

    Parameters
    ----------
    db_path
        Path to the DuckDB database.
    document_output_dir
        Directory for exported artifacts.
    options
        Export options configuration.
    hooks
        Override hooks for customizing export behavior.
    """
    resolved_hooks = hooks or ExportHooks()
    export_options = options or ExportOptions(
        export=ExportCallOptions(
            validate_exports=False,
            schemas=None,
            datasets=None,
            validation_profile=None,
            force_full_export=False,
        )
    )
    export_options = replace(export_options, validator=resolved_hooks.validator)
    gateway = resolved_hooks.gateway_factory(db_path)
    resolved_hooks.create_views(gateway.con)
    resolved_hooks.export_runner(
        gateway=gateway,
        output_dir=document_output_dir,
        options=export_options,
    )


__all__ = [
    "DEFAULT_BUILD_SUBDIR",
    "ExportHooks",
    "HistoryTimeseriesParams",
    "run_export_docs",
    "run_history_timeseries",
]
