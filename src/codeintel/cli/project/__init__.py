"""Project detection and configuration.

This package provides:

- ``ProjectConfig``: Project configuration models
- ``find_project_root()``: Project root detection
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

__all__ = [
    "PROJECT_FILE",
    "AnalyticsProjectConfig",
    "GraphsProjectConfig",
    "IngestProjectConfig",
    "ProjectConfig",
    "ProjectConfigError",
    "ProjectNotFoundError",
    "StorageProjectConfig",
    "detect_commit",
    "find_project_root",
    "load_project_config",
]
