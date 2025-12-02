"""Runtime resolution helper for analytics metric computations.

This module provides a unified pattern for resolving the analytics runtime
needed by secondary metric plugins, eliminating boilerplate duplication.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.analytics.graph_runtime import GraphRuntime
    from codeintel.graphs.core import GraphExecutionContext
    from codeintel.storage.gateway import StorageGateway


@dataclass
class ResolvedRuntime:
    """Resolved runtime context for analytics computations.

    Attributes
    ----------
    gateway
        Storage gateway for database access.
    runtime
        Graph runtime for graph operations.
    repo
        Repository identifier.
    commit
        Commit hash.
    """

    gateway: StorageGateway
    runtime: GraphRuntime
    repo: str
    commit: str


@contextmanager
def resolve_analytics_runtime(
    ctx: GraphExecutionContext,
) -> Iterator[ResolvedRuntime]:
    """Resolve runtime for analytics computations.

    Handles gateway access via resource injection with fallback,
    and creates a standard runtime with default backend configuration.

    Parameters
    ----------
    ctx
        Graph execution context.

    Yields
    ------
    ResolvedRuntime
        Resolved runtime context with gateway, runtime, repo, and commit.

    Examples
    --------
    >>> with resolve_analytics_runtime(ctx) as rt:
    ...     compute_metrics(rt.gateway, repo=rt.repo, commit=rt.commit, runtime=rt.runtime)
    """
    # Late imports to avoid circular dependencies - this module is specifically
    # designed to encapsulate these imports for cleaner plugin code
    from codeintel.analytics.graph_runtime import (  # noqa: PLC0415
        GraphRuntimeOptions,
        resolve_graph_runtime,
    )
    from codeintel.config.primitives import GraphBackendConfig  # noqa: PLC0415
    from codeintel.graphs.resources import StorageResource  # noqa: PLC0415

    # Get gateway via resource injection or fallback
    if ctx.resources is not None and ctx.has_resource(StorageResource.RESOURCE_NAME):
        storage = ctx.require(StorageResource)
        gateway = storage.gateway
    else:
        gateway = ctx.gateway

    # Resolve runtime with default backend configuration
    runtime = resolve_graph_runtime(
        gateway,
        ctx.snapshot,
        GraphRuntimeOptions(snapshot=ctx.snapshot, backend=GraphBackendConfig()),
    )

    yield ResolvedRuntime(
        gateway=gateway,
        runtime=runtime,
        repo=ctx.repo,
        commit=ctx.commit,
    )


__all__ = ["ResolvedRuntime", "resolve_analytics_runtime"]
