"""Compatibility shim for project module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.project`` package instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.project import ProjectConfig, find_project_root

    # New (preferred):
    from codeintel.cli.project import ProjectConfig, find_project_root
    # (import from the package, not this file)
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.project' (module) is deprecated. "
    "The project package now provides these exports directly. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.project._project import (
    PROJECT_FILE,
    AnalyticsProjectConfig,
    GraphsProjectConfig,
    IngestProjectConfig,
    ProjectConfig,
    ProjectConfigError,
    ProjectNotFoundError,
    ProjectRuntime,
    StorageProjectConfig,
    build_project_runtime,
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
    "ProjectRuntime",
    "StorageProjectConfig",
    "build_project_runtime",
    "detect_commit",
    "find_project_root",
    "load_project_config",
]
