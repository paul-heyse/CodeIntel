"""Shared runtime builders for serving transports."""

from __future__ import annotations

from codeintel.serving.context import ServingContext
from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig


def build_db_manager(settings: ServingSettings) -> ServingDBManager:
    """Build a configured DB manager for the serving snapshot pointer.

    Parameters
    ----------
    settings
        Serving settings.

    Returns
    -------
    ServingDBManager
        Configured DB manager.
    """
    return ServingDBManager(
        pointer_path=settings.serve_dir / "current.json",
        pool_cfg=PoolConfig(size=settings.pool_size),
        poll_interval_s=settings.poll_interval_s,
        hot_swap=settings.hot_swap,
    )


def build_kernel(db_manager: ServingDBManager, settings: ServingSettings) -> SemanticQueryKernel:
    """Build the semantic query kernel.

    Parameters
    ----------
    db_manager
        Serving DB manager.
    settings
        Serving settings.

    Returns
    -------
    SemanticQueryKernel
        Configured kernel.
    """
    return SemanticQueryKernel(db=db_manager, settings=settings)


def build_runtime(settings: ServingSettings) -> ServingContext:
    """Build the shared runtime for serving transports.

    Parameters
    ----------
    settings
        Serving settings.

    Returns
    -------
    ServingContext
        Constructed runtime dependencies.
    """
    db_manager = build_db_manager(settings)
    kernel = build_kernel(db_manager, settings)
    ops = ServingOperations(kernel=kernel, settings=settings)
    return ServingContext(settings=settings, db_manager=db_manager, kernel=kernel, ops=ops)


__all__ = ["build_db_manager", "build_kernel", "build_runtime"]
