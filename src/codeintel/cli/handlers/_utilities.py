"""Handler utilities - gateway management.

This module provides shared utilities for CLI handlers:
- Gateway management for database access

Note: Logging setup is handled by `execution/bootstrap.py:bootstrap_cli()`.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.cli.services.storage import default_validation_summary_path
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.validation import ContractValidationMode

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.cli.resolution.types import ResolvedRuntime
    from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HandlerGatewayOptions:
    """Options for constructing a handler gateway."""

    read_only: bool = True
    validation_mode: ContractValidationMode = ContractValidationMode.LENIENT
    dataset_root_dir: Path | None = None
    snapshot_id: str | None = None
    repo: str | None = None


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
    options: HandlerGatewayOptions | None = None,
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
    options
        Optional configuration overrides for gateway construction.

    Returns
    -------
    StorageGateway
        Open gateway.
    """
    resolved_options = options or HandlerGatewayOptions()
    validation_summary_path = (
        default_validation_summary_path(db_path)
        if resolved_options.validation_mode != ContractValidationMode.OFF
        else None
    )
    storage_config = StorageConfig(
        db_path=db_path,
        dataset_root_dir=resolved_options.dataset_root_dir,
        read_only=resolved_options.read_only,
        validate_schema=resolved_options.validation_mode != ContractValidationMode.OFF,
        validation_mode=resolved_options.validation_mode,
        validation_summary_path=validation_summary_path,
        repo=resolved_options.repo,
        commit=resolved_options.snapshot_id,
    )
    gateway = open_gateway(storage_config)
    if not resolved_options.read_only:
        gateway.policy.ensure_schemas_preserve()
    return gateway


@contextmanager
def runtime_gateway(
    runtime: ResolvedRuntime,
    *,
    read_only: bool = True,
    validation_mode: ContractValidationMode = ContractValidationMode.LENIENT,
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
    validation_mode
        Contract validation behavior when opening the gateway.

    Yields
    ------
    StorageGateway
        Open gateway that will be closed on context exit.

    Examples
    --------
    >>> with runtime_gateway(ctx.runtime) as gateway:
    ...     gateway.execute("SELECT 1")
    """
    options = HandlerGatewayOptions(
        read_only=read_only,
        validation_mode=validation_mode,
        dataset_root_dir=runtime.paths.dataset_root_dir,
        snapshot_id=runtime.commit,
        repo=runtime.repo,
    )
    gateway = open_handler_gateway(
        runtime.paths.db_path,
        options=options,
    )
    try:
        yield gateway
    finally:
        gateway.close()


__all__ = [
    "get_handler_logger",
    "open_handler_gateway",
    "runtime_gateway",
]
