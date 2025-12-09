"""Gateway lifecycle management for CLI operations.

This module provides centralized gateway management for CLI operations,
ensuring consistent opening, caching, and cleanup of storage gateways.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.cli.resolution.runtime import resolve_runtime
from codeintel.storage.gateway import StorageConfig, open_gateway

if TYPE_CHECKING:
    from codeintel.cli.execution.context import ExecutionContext
    from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)


class GatewayManager:
    """Manage gateway lifecycle for ExecutionContext.

    The manager opens gateways on demand using the resolved runtime's db_path.
    Gateways should be cached in the ExecutionContext and cleaned up when the
    context is closed.

    This class is stateless - all state should be stored in ExecutionContext.

    Examples
    --------
    >>> manager = GatewayManager()
    >>> gateway = GatewayManager.open_gateway(ctx, read_only=True)  # doctest: +SKIP
    >>> # ... use gateway ...
    >>> gateway.close()  # doctest: +SKIP
    """

    @staticmethod
    def open_gateway(
        ctx: ExecutionContext,
        *,
        read_only: bool = True,
    ) -> StorageGateway:
        """Open gateway for context.

        Use the resolved runtime to determine the database path.
        If the context doesn't have a resolved runtime yet, this will
        trigger resolution.

        Parameters
        ----------
        ctx
            Execution context. Must have resolvable runtime.
        read_only
            Whether to open in read-only mode. Defaults to True for safety.

        Returns
        -------
        StorageGateway
            Open gateway connected to the database.

        Notes
        -----
        May propagate ResolutionError from resolve_runtime if runtime cannot
        be resolved, or StorageConnectionError from open_gateway if the
        database cannot be opened.
        """
        # Resolve runtime using the resolver
        runtime = resolve_runtime(ctx)

        LOG.debug(
            "Opening gateway for %s: db_path=%s, read_only=%s",
            ctx.operation_id,
            runtime.db_path,
            read_only,
        )

        if read_only:
            storage_config = StorageConfig.for_readonly(db_path=runtime.db_path)
        else:
            storage_config = StorageConfig.for_ingest(db_path=runtime.db_path)

        return open_gateway(storage_config)


def open_gateway_for_context(
    ctx: ExecutionContext,
    *,
    read_only: bool = True,
) -> StorageGateway:
    """Open gateway for context (module-level convenience function).

    Parameters
    ----------
    ctx
        Execution context.
    read_only
        Whether to open in read-only mode.

    Returns
    -------
    StorageGateway
        Open gateway.
    """
    return GatewayManager.open_gateway(ctx, read_only=read_only)


__all__ = [
    "GatewayManager",
    "open_gateway_for_context",
]
