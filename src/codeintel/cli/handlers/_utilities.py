"""Handler utilities - gateway management and runtime conversion.

This module provides shared utilities for CLI handlers:
- Gateway management for database access
- Runtime conversion helpers

Note: Logging setup is handled by `execution/bootstrap.py:bootstrap_cli()`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway

if TYPE_CHECKING:
    from codeintel.cli.project import ProjectRuntime
    from codeintel.cli.resolution.types import ResolvedRuntime

LOG = logging.getLogger(__name__)


def get_handler_logger(name: str) -> logging.Logger:
    """Get a logger for a handler.

    Parameters
    ----------
    name
        Logger name (typically operation_id or handler name).

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    return logging.getLogger(f"codeintel.cli.handlers.{name}")


def open_handler_gateway(
    db_path: Path,
    *,
    read_only: bool = True,
) -> StorageGateway:
    """Open a gateway for handler use.

    Parameters
    ----------
    db_path
        Path to the database file.
    read_only
        Whether to open in read-only mode.

    Returns
    -------
    StorageGateway
        Open gateway.
    """
    storage_config = StorageConfig(db_path=db_path, read_only=read_only)
    return open_gateway(storage_config)


def resolved_to_project_runtime(runtime: ResolvedRuntime) -> ProjectRuntime:
    """Convert ResolvedRuntime to ProjectRuntime.

    This conversion is needed when interfacing with code that expects
    the older ProjectRuntime type.

    Parameters
    ----------
    runtime
        ResolvedRuntime from handler context.

    Returns
    -------
    ProjectRuntime
        Compatible ProjectRuntime instance.
    """
    # Import here to avoid circular dependency at module load time
    from codeintel.cli.project import (  # noqa: PLC0415
        ProjectRuntime as ProjectRuntimeClass,
    )

    gateway = open_handler_gateway(runtime.paths.db_path, read_only=True)
    return ProjectRuntimeClass(
        root=runtime.root,
        project=runtime.project,
        cfg=runtime.config,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        gateway=gateway,
        tools=runtime.config.tools,
        serving=runtime.serving,
    )


__all__ = [
    "get_handler_logger",
    "open_handler_gateway",
    "resolved_to_project_runtime",
]
