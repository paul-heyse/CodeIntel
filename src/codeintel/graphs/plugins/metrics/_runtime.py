"""Runtime resolution helper for analytics metric computations.

This module provides a unified pattern for resolving the analytics runtime
needed by secondary metric plugins, eliminating boilerplate duplication.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.analytics.graph_runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    resolve_graph_runtime,
)
from codeintel.config.primitives import GraphBackendConfig
from codeintel.graphs.resources import StorageResource

if TYPE_CHECKING:
    from codeintel.graphs.core import GraphPluginExecutionContext
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
    ctx: GraphPluginExecutionContext,
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
    # Get gateway via resource injection or fallback
    if ctx.has_graph_resource(StorageResource.RESOURCE_NAME):
        storage = ctx.graph_resources.require(StorageResource)
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
