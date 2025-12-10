"""Handler utilities - logging, gateway management.

This module provides shared utilities for CLI handlers:
- Logging setup and configuration
- Gateway management for database access
- Runtime conversion helpers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway

if TYPE_CHECKING:
    from codeintel.cli.config import CliConfig
    from codeintel.cli.project import ProjectRuntime
    from codeintel.cli.resolution.types import ResolvedRuntime

LOG = logging.getLogger(__name__)

# Track if logging has been configured to avoid duplicate setup
_LOGGING_CONFIGURED = False

# Verbosity thresholds
VERBOSITY_DEBUG = 2
VERBOSITY_INFO = 1


def setup_logging(
    verbosity: int = 0,
    *,
    config: CliConfig | None = None,
    force: bool = False,
) -> None:
    """Configure logging for CLI handlers.

    This function should be called once at the start of CLI command execution.
    It configures the root logger based on verbosity level.

    Parameters
    ----------
    verbosity
        Verbosity level:
        - 0 = use config default (or WARNING)
        - 1 = INFO level
        - 2+ = DEBUG level
    config
        Optional CliConfig for default log level when verbosity is 0.
    force
        If True, reconfigure logging even if already configured.

    Examples
    --------
    >>> setup_logging(0)  # Uses config default or WARNING
    >>> setup_logging(1)  # INFO level
    >>> setup_logging(2)  # DEBUG level
    """
    global _LOGGING_CONFIGURED  # noqa: PLW0603

    if _LOGGING_CONFIGURED and not force:
        return

    level = _determine_log_level(verbosity, config)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=force,
    )
    _LOGGING_CONFIGURED = True


def _determine_log_level(verbosity: int, config: CliConfig | None) -> int:
    """Determine log level from verbosity and config.

    Parameters
    ----------
    verbosity
        Verbosity level from CLI.
    config
        Optional configuration.

    Returns
    -------
    int
        Logging level constant.
    """
    if verbosity >= VERBOSITY_DEBUG:
        return logging.DEBUG
    if verbosity >= VERBOSITY_INFO:
        return logging.INFO
    if config is not None:
        return getattr(logging, config.log_level, logging.WARNING)
    return logging.WARNING


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
    "VERBOSITY_DEBUG",
    "VERBOSITY_INFO",
    "get_handler_logger",
    "open_handler_gateway",
    "resolved_to_project_runtime",
    "setup_logging",
]
