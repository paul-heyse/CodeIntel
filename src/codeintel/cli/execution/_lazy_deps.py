"""Lazy dependency loading for execution context.

This module provides lazy-loaded references to resolution functions
to avoid circular imports. Import this module at the point of use
rather than at module load time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.cli.resolution import open_gateway_for_context, resolve_runtime

if TYPE_CHECKING:
    from codeintel.cli.execution.context import ExecutionContext
    from codeintel.cli.resolution.types import ResolvedRuntime
    from codeintel.storage.gateway import StorageGateway


def lazy_resolve_runtime(ctx: ExecutionContext) -> ResolvedRuntime:
    """Resolve runtime for context.

    Parameters
    ----------
    ctx
        Execution context with params.

    Returns
    -------
    ResolvedRuntime
        Resolved runtime.
    """
    return resolve_runtime(ctx)


def lazy_open_gateway(
    ctx: ExecutionContext,
    *,
    read_only: bool = True,
) -> StorageGateway:
    """Open gateway for context.

    Parameters
    ----------
    ctx
        Execution context.
    read_only
        Whether to open read-only.

    Returns
    -------
    StorageGateway
        Open gateway.
    """
    return open_gateway_for_context(ctx, read_only=read_only)


__all__ = [
    "lazy_open_gateway",
    "lazy_resolve_runtime",
]
