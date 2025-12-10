"""Shared fixtures for MCP tool and backend tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from codeintel.serving.backend import BackendLimits, DuckDBQueryService
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.observability import ServiceObservability
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import BackendOptions, build_duckdb_backend, build_duckdb_query_service

if TYPE_CHECKING:
    from tests._helpers.context import TestContext

DEFAULT_LIMIT = 10
MAX_ROWS = 100


@dataclass(frozen=True)
class McpBackendComponents:
    """Aggregated gateway, query, service, and backend for MCP tests."""

    gateway: StorageGateway
    repo: str
    commit: str
    limits: BackendLimits
    query: DuckDBQueryService
    service: LocalQueryService
    observability: ServiceObservability | None
    backend: DuckDBBackend


def _build_components(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    limits: BackendLimits | None = None,
    observability: ServiceObservability | None = None,
) -> McpBackendComponents:
    """Construct query/service/backend trio for a gateway snapshot.

    Returns
    -------
    McpBackendComponents
        Aggregated gateway, query, service, and backend.
    """
    effective_limits = limits or BackendLimits(
        default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS
    )
    query = build_duckdb_query_service(
        gateway,
        repo=repo,
        commit=commit,
        limits=effective_limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(gateway.datasets.mapping),
        observability=observability,
    )
    backend = build_duckdb_backend(
        gateway,
        repo=repo,
        commit=commit,
        service=service,
        options=BackendOptions(limits=effective_limits),
    )
    return McpBackendComponents(
        gateway=gateway,
        repo=repo,
        commit=commit,
        limits=effective_limits,
        query=query,
        service=service,
        observability=observability,
        backend=backend,
    )


@pytest.fixture
def mcp_backend_factory() -> Callable[..., McpBackendComponents]:
    """Build MCP backend components for any gateway snapshot.

    Returns
    -------
    Callable[..., McpBackendComponents]
        Factory that produces backend components given gateway, repo, and commit.
    """

    def _build(
        *,
        gateway: StorageGateway,
        repo: str,
        commit: str,
        limits: BackendLimits | None = None,
        observability: ServiceObservability | None = None,
    ) -> McpBackendComponents:
        return _build_components(
            gateway,
            repo=repo,
            commit=commit,
            limits=limits,
            observability=observability,
        )

    return _build


@pytest.fixture
def mcp_backend_components(
    provisioned_repo: TestContext,
    mcp_backend_factory: Callable[..., McpBackendComponents],
) -> McpBackendComponents:
    """Provide provisioned gateway components reused across MCP tool tests.

    Returns
    -------
    McpBackendComponents
        Aggregated components built from the provisioned gateway snapshot.
    """
    return mcp_backend_factory(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )


@pytest.fixture
def mcp_backend(mcp_backend_components: McpBackendComponents) -> McpBackendComponents:
    """Return aggregated backend components for the provisioned gateway snapshot.

    Returns
    -------
    McpBackendComponents
        Aggregated components constructed from the provisioned gateway snapshot.
    """
    return mcp_backend_components


@pytest.fixture
def mcp_service(mcp_backend_components: McpBackendComponents) -> LocalQueryService:
    """Return LocalQueryService bound to the provisioned gateway snapshot.

    Returns
    -------
    LocalQueryService
        Service constructed from the provisioned gateway snapshot.
    """
    return mcp_backend_components.service


@pytest.fixture
def mcp_query_service(mcp_backend_components: McpBackendComponents) -> DuckDBQueryService:
    """Return DuckDBQueryService bound to the provisioned gateway snapshot.

    Returns
    -------
    DuckDBQueryService
        Query service constructed from the provisioned gateway snapshot.
    """
    return mcp_backend_components.query


__all__ = [
    "McpBackendComponents",
    "mcp_backend",
    "mcp_backend_components",
    "mcp_backend_factory",
    "mcp_query_service",
    "mcp_service",
]
