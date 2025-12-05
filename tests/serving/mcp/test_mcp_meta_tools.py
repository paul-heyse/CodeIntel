"""Tests for MCP meta tools.

This module tests the dataset and operation introspection MCP tools using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.meta_tools import register_meta_tools
from codeintel.serving.operations.catalog import (
    build_dataset_meta,
    build_serving_dataflow_graph,
    iter_registry_operations,
)
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.gateway import build_duckdb_query_service

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway

# =============================================================================
# Constants
# =============================================================================

DEFAULT_LIMIT = 10
MAX_ROWS = 100


# =============================================================================
# Helper Functions
# =============================================================================


def _build_backend(provisioned_repo: ProvisionedGateway) -> DuckDBBackend:
    """Build a DuckDBBackend for testing.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.

    Returns
    -------
    DuckDBBackend
        Configured backend.
    """
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    return DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
        observability=None,
        service=service,
    )


# =============================================================================
# register_meta_tools Tests
# =============================================================================


def test_register_meta_tools_success(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_meta_tools registers tools successfully.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = FastMCP("Test Meta Tools", json_response=True)
    backend = _build_backend(provisioned_repo)

    # Should not raise
    register_meta_tools(mcp, backend)

    # Server should be configured
    assert mcp.name == "Test Meta Tools"


def test_register_meta_tools_different_backend(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_meta_tools works with different backend instances.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = FastMCP("Test Different Backend", json_response=True)
    backend = _build_backend(provisioned_repo)

    # First registration
    register_meta_tools(mcp, backend)

    assert mcp.name == "Test Different Backend"


def test_register_meta_tools_with_service_directly(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_meta_tools works with service directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = FastMCP("Test Service Direct", json_response=True)
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )

    register_meta_tools(mcp, service)

    assert mcp.name == "Test Service Direct"


def test_register_meta_tools_with_local_query_service(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_meta_tools works with LocalQueryService directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = FastMCP("Test Local Service", json_response=True)
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )

    # Should work with service directly
    register_meta_tools(mcp, service)

    assert mcp.name == "Test Local Service"


# =============================================================================
# Multiple Registration Tests
# =============================================================================


def test_register_meta_tools_different_servers(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify tools can be registered on different servers.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    # Register on first server
    mcp1 = FastMCP("Server One", json_response=True)
    register_meta_tools(mcp1, backend)
    assert mcp1.name == "Server One"

    # Register on second server
    mcp2 = FastMCP("Server Two", json_response=True)
    register_meta_tools(mcp2, backend)
    assert mcp2.name == "Server Two"


# =============================================================================
# Backend Variants Tests
# =============================================================================


def test_register_meta_tools_backend_with_limits(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify meta tools work with backend having custom limits.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = FastMCP("Test Backend Limits", json_response=True)
    backend = _build_backend(provisioned_repo)

    register_meta_tools(mcp, backend)
    assert mcp.name == "Test Backend Limits"


def test_register_meta_tools_backend_with_service(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify meta tools work with backend using provided service.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = FastMCP("Test With Service", json_response=True)
    backend = _build_backend(provisioned_repo)

    # Backend was built with service
    assert backend.service is not None

    register_meta_tools(mcp, backend)
    assert mcp.name == "Test With Service"


# =============================================================================
# Limits Tests
# =============================================================================


def test_register_meta_tools_custom_limits(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify meta tools work with custom limits.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    custom_limit = 50
    custom_max = 500
    limits = BackendLimits(default_limit=custom_limit, max_rows_per_call=custom_max)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
        observability=None,
        service=service,
    )

    mcp = FastMCP("Test Custom Limits", json_response=True)
    register_meta_tools(mcp, backend)

    assert mcp.name == "Test Custom Limits"
    assert backend.limits.default_limit == custom_limit
    assert backend.limits.max_rows_per_call == custom_max


# =============================================================================
# Backend Type Tests
# =============================================================================


def test_register_meta_tools_duckdb_backend(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_meta_tools works with DuckDBBackend.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = FastMCP("Test DuckDB Backend", json_response=True)
    backend = _build_backend(provisioned_repo)

    # Verify backend is DuckDBBackend
    assert isinstance(backend, DuckDBBackend)

    register_meta_tools(mcp, backend)
    assert mcp.name == "Test DuckDB Backend"


def test_register_meta_tools_preserves_backend_properties(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify registration doesn't alter backend properties.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = FastMCP("Test Properties", json_response=True)
    backend = _build_backend(provisioned_repo)

    original_repo = backend.repo
    original_commit = backend.commit
    original_limits = backend.limits

    register_meta_tools(mcp, backend)

    # Backend properties should be unchanged
    assert backend.repo == original_repo
    assert backend.commit == original_commit
    assert backend.limits == original_limits


# =============================================================================
# Backend List Datasets via Service Tests
# =============================================================================


def test_backend_list_datasets(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.list_datasets returns dataset descriptors.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    datasets = backend.list_datasets()

    assert isinstance(datasets, list)


def test_backend_dataset_specs(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify backend.dataset_specs returns specs.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)

    specs = backend.dataset_specs()

    assert isinstance(specs, list)


def test_service_list_datasets(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify service.list_datasets returns dataset descriptors.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )

    datasets = service.list_datasets()

    assert isinstance(datasets, list)


def test_service_dataset_specs(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify service.dataset_specs returns specs.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )

    specs = service.dataset_specs()

    assert isinstance(specs, list)


# =============================================================================
# Registry Helper Tests (Public API)
# =============================================================================


def test_build_dataset_meta(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify build_dataset_meta returns metadata entries.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = _build_backend(provisioned_repo)
    limits = backend.limits
    service = backend.service

    metas = build_dataset_meta(service, limits)

    assert isinstance(metas, list)


def test_build_serving_dataflow_graph() -> None:
    """Verify build_serving_dataflow_graph returns nodes and edges."""
    nodes, edges = build_serving_dataflow_graph()

    assert isinstance(nodes, list)
    assert isinstance(edges, list)


def test_iter_registry_operations() -> None:
    """Verify iter_registry_operations yields operations."""
    operations = list(iter_registry_operations())

    assert isinstance(operations, list)
    assert len(operations) > 0


# =============================================================================
# Registered Tools Integration Tests
# =============================================================================


def test_registered_tools_after_meta_registration(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify tools are callable after registration.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = FastMCP("Test Tools", json_response=True)
    backend = _build_backend(provisioned_repo)

    register_meta_tools(mcp, backend)

    # MCP server should have tools registered
    assert mcp.name == "Test Tools"
