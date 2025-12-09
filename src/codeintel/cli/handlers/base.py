"""Base handler utilities - single implementation for all CLI handlers.

This module provides the unified implementation of:
1. Logging setup (one implementation, used by all handlers)
2. HandlerContext (unified context for all handlers)
3. Common gateway and runtime building utilities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from codeintel.cli.config import CliConfig, load_config
from codeintel.cli.execution.context import ExecutionContext
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway

LOG = logging.getLogger(__name__)

# Track if logging has been configured to avoid duplicate setup
_LOGGING_CONFIGURED = False

# Verbosity thresholds
VERBOSITY_DEBUG = 2
VERBOSITY_INFO = 1


# =============================================================================
# Logging Setup (Single Implementation)
# =============================================================================


def setup_logging(
    verbosity: int = 0,
    *,
    config: CliConfig | None = None,
    force: bool = False,
) -> None:
    """Configure logging - SINGLE IMPLEMENTATION for all handlers.

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


# =============================================================================
# Handler Context (Unified)
# =============================================================================


@dataclass(frozen=True)
class HandlerContext:
    """Unified context for all CLI handlers.

    This replaces scattered RuntimeCliOptions, BuildRunContext, etc.
    with a single context type that provides access to:
    - Configuration (CliConfig)
    - Execution context (ExecutionContext)
    - Project root
    - Verbosity level
    - Logger

    Parameters
    ----------
    config
        CLI configuration.
    execution
        Execution context with operation_id and params.
    project_root
        Optional project root directory.
    verbosity
        Verbosity level (0-2+).

    Examples
    --------
    >>> ctx = build_handler_context("build.run", {"target": "all"})
    >>> ctx.logger.info("Starting build")
    >>> ctx.output_format
    'text'
    """

    config: CliConfig
    execution: ExecutionContext
    project_root: Path | None = None
    verbosity: int = 0

    @property
    def operation_id(self) -> str:
        """Get the operation ID.

        Returns
        -------
        str
            Operation identifier.
        """
        return self.execution.operation_id

    @property
    def logger(self) -> logging.Logger:
        """Get a logger for this handler.

        Returns
        -------
        logging.Logger
            Configured logger.
        """
        return get_handler_logger(self.operation_id)

    @property
    def output_format(self) -> str:
        """Get the output format.

        Returns
        -------
        str
            Output format ('text' or 'json').
        """
        return self.config.output_format

    @property
    def color_enabled(self) -> bool:
        """Check if colored output is enabled.

        Returns
        -------
        bool
            True if color is enabled.
        """
        return self.config.color

    @property
    def progress_enabled(self) -> bool:
        """Check if progress display is enabled.

        Returns
        -------
        bool
            True if progress is enabled.
        """
        return self.config.progress.enabled


def build_handler_context(
    operation_id: str,
    params: dict[str, object],
    *,
    config: CliConfig | None = None,
    verbosity: int = 0,
    project_root: Path | None = None,
) -> HandlerContext:
    """Build unified handler context.

    Parameters
    ----------
    operation_id
        Operation identifier (e.g., "build.run").
    params
        Operation parameters.
    config
        Optional CliConfig (loaded from sources if not provided).
    verbosity
        Verbosity level.
    project_root
        Optional project root directory.

    Returns
    -------
    HandlerContext
        Unified handler context.
    """
    config = config or load_config(validate=False)
    setup_logging(verbosity, config=config)
    execution = ExecutionContext.for_sync(operation_id, params)
    return HandlerContext(
        config=config,
        execution=execution,
        verbosity=verbosity,
        project_root=project_root,
    )


# =============================================================================
# Gateway Management (Shared Utilities)
# =============================================================================


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


__all__ = [
    "HandlerContext",
    "build_handler_context",
    "get_handler_logger",
    "open_handler_gateway",
    "setup_logging",
]
