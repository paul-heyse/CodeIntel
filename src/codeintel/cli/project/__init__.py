"""Project detection and pipeline management.

This package provides:

- ``ProjectConfig``: Project configuration models
- ``find_project_root()``: Project root detection
- Pipeline and batch execution
"""

from __future__ import annotations

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
    "PROJECT_FILE",
    "AnalyticsProjectConfig",
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
    "read_stdin_operations",
    "stream_results",
]
