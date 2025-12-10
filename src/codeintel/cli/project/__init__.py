"""Project detection and pipeline management.

This package provides:

- ``ProjectConfig``: Project configuration models
- ``find_project_root()``: Project root detection
- Pipeline and batch execution
- Dry-run planning
"""

from __future__ import annotations

# Project configuration and runtime
from codeintel.cli.project._project import (
    PROJECT_FILE,
    AnalyticsProjectConfig,
    GraphsProjectConfig,
    IngestProjectConfig,
    ProjectConfig,
    ProjectConfigError,
    ProjectNotFoundError,
    StorageProjectConfig,
    detect_commit,
    find_project_root,
    load_project_config,
)

# Dry-run planning
from codeintel.cli.project.dry_run import (
    plan_dry_run,
    render_dry_run,
    render_dry_run_to,
)

# Pipeline execution
from codeintel.cli.project.pipelines import (
    BatchItemResult,
    BatchOperation,
    BatchResult,
    PipelineConfig,
    execute_batch,
    load_batch,
    read_stdin_operations,
    stream_results,
)

__all__ = [
    # Project
    "PROJECT_FILE",
    "AnalyticsProjectConfig",
    # Pipelines
    "BatchItemResult",
    "BatchOperation",
    "BatchResult",
    "GraphsProjectConfig",
    "IngestProjectConfig",
    "PipelineConfig",
    "ProjectConfig",
    "ProjectConfigError",
    "ProjectNotFoundError",
    "StorageProjectConfig",
    "detect_commit",
    "execute_batch",
    "find_project_root",
    "load_batch",
    "load_project_config",
    # Dry-run
    "plan_dry_run",
    "read_stdin_operations",
    "render_dry_run",
    "render_dry_run_to",
    "stream_results",
]
