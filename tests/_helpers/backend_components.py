"""Shared builder for backend query layer components used in tests."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.graphs.engine import GraphEngine
from codeintel.serving.backend import BackendContext, BackendLimits, DuckDBRepositories
from codeintel.serving.backend.core import GraphEngineProvider
from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class BackendComponents:
    """Aggregated backend context pieces for query layer tests."""

    context: BackendContext
    repositories: DuckDBRepositories
    provider: GraphEngineProvider


def build_backend_components(
    gateway: StorageGateway,
    *,
    limits: BackendLimits | None = None,
    graph_engine: GraphEngine | None = None,
) -> BackendComponents:
    """Construct BackendContext, repositories, and provider for tests."""
    repo = gateway.config.repo or "demo/repo"
    commit = gateway.config.commit or "deadbeef"
    context = BackendContext(
        gateway=gateway,
        repo=repo,
        commit=commit,
        limits=limits or BackendLimits(),
        graph_engine=graph_engine,
    )
    repositories = DuckDBRepositories(gateway, repo, commit)
    provider = GraphEngineProvider(context=context, graph_engine=graph_engine)
    return BackendComponents(context=context, repositories=repositories, provider=provider)


__all__ = ["BackendComponents", "build_backend_components"]
