"""Handler utilities - gateway management.

This module provides shared utilities for CLI handlers:
- Gateway management for database access

Note: Logging setup is handled by `execution/bootstrap.py:bootstrap_cli()`.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

from codeintel.storage.gateway import StorageConfig, open_gateway

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.cli.resolution.types import ResolvedRuntime
    from codeintel.storage.gateway import StorageGateway

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

    .. note::

       For CLI handlers, prefer using ``CommandContext.gateway`` property
       or ``CommandContext.write_gateway()`` instead. Those methods provide
       automatic lifecycle management and consistent configuration.

       This function is retained for internal use by ``runtime_gateway()``
       and should not be called directly from handlers.

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


@contextmanager
def runtime_gateway(
    runtime: ResolvedRuntime,
    *,
    read_only: bool = True,
) -> Iterator[StorageGateway]:
    """Open a gateway for a runtime as a context manager.

    Use this to get a gateway that is automatically closed when the
    context exits.

    Parameters
    ----------
    runtime
        ResolvedRuntime with paths.db_path.
    read_only
        Whether to open in read-only mode.

    Yields
    ------
    StorageGateway
        Open gateway that will be closed on context exit.

    Examples
    --------
    >>> with runtime_gateway(ctx.runtime) as gateway:
    ...     gateway.execute("SELECT 1")
    """
    gateway = open_handler_gateway(runtime.paths.db_path, read_only=read_only)
    try:
        yield gateway
    finally:
        gateway.close()


__all__ = [
    "get_handler_logger",
    "open_handler_gateway",
    "runtime_gateway",
]
