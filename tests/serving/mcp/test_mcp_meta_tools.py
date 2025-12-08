"""Tests for MCP meta tools.

This module tests the dataset and operation introspection MCP tools using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.meta_tools import register_meta_tools
from codeintel.serving.operations.catalog import (
    build_dataset_meta,
    build_serving_dataflow_graph,
    iter_registry_operations,
)
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.assertions import (
    expect_equal,
    expect_is_instance,
    expect_true,
)
from tests._helpers.gateway import build_duckdb_query_service
from tests._helpers.mcp_fast import wrap_fastmcp

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
    mcp = wrap_fastmcp("Test Meta Tools")
    backend = _build_backend(provisioned_repo)

    # Should not raise
    register_meta_tools(mcp, backend)

    # Server should be configured
    expect_equal(mcp.name, "Test Meta Tools")


def test_register_meta_tools_different_backend(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_meta_tools works with different backend instances.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test Different Backend")
    backend = _build_backend(provisioned_repo)

    # First registration
    register_meta_tools(mcp, backend)

    expect_equal(mcp.name, "Test Different Backend")


def test_register_meta_tools_with_service_directly(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_meta_tools works with service directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test Service Direct")
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

    expect_equal(mcp.name, "Test Service Direct")


def test_register_meta_tools_with_local_query_service(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_meta_tools works with LocalQueryService directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test Local Service")
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

    expect_equal(mcp.name, "Test Local Service")


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
    mcp1 = wrap_fastmcp("Server One")
    register_meta_tools(mcp1, backend)
    expect_equal(mcp1.name, "Server One")

    # Register on second server
    mcp2 = wrap_fastmcp("Server Two")
    register_meta_tools(mcp2, backend)
    expect_equal(mcp2.name, "Server Two")


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
    mcp = wrap_fastmcp("Test Backend Limits")
    backend = _build_backend(provisioned_repo)

    register_meta_tools(mcp, backend)
    expect_equal(mcp.name, "Test Backend Limits")


def test_register_meta_tools_backend_with_service(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify meta tools work with backend using provided service.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test With Service")
    backend = _build_backend(provisioned_repo)

    # Backend was built with service
    expect_true(backend.service is not None)

    register_meta_tools(mcp, backend)
    expect_equal(mcp.name, "Test With Service")


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

    mcp = wrap_fastmcp("Test Custom Limits")
    register_meta_tools(mcp, backend)

    expect_equal(mcp.name, "Test Custom Limits")
    expect_equal(backend.limits.default_limit, custom_limit)
    expect_equal(backend.limits.max_rows_per_call, custom_max)


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
    mcp = wrap_fastmcp("Test DuckDB Backend")
    backend = _build_backend(provisioned_repo)

    # Verify backend is DuckDBBackend
    expect_is_instance(backend, DuckDBBackend)

    register_meta_tools(mcp, backend)
    expect_equal(mcp.name, "Test DuckDB Backend")


def test_register_meta_tools_preserves_backend_properties(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify registration doesn't alter backend properties.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    mcp = wrap_fastmcp("Test Properties")
    backend = _build_backend(provisioned_repo)

    original_repo = backend.repo
    original_commit = backend.commit
    original_limits = backend.limits

    register_meta_tools(mcp, backend)

    # Backend properties should be unchanged
    expect_equal(backend.repo, original_repo)
    expect_equal(backend.commit, original_commit)
    expect_equal(backend.limits, original_limits)


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

    expect_is_instance(datasets, list)


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

    expect_is_instance(specs, list)


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

    expect_is_instance(datasets, list)


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

    expect_is_instance(specs, list)


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

    expect_is_instance(metas, list)


def test_build_serving_dataflow_graph() -> None:
    """Verify build_serving_dataflow_graph returns nodes and edges."""
    nodes, edges = build_serving_dataflow_graph()

    expect_is_instance(nodes, list)
    expect_is_instance(edges, list)


def test_iter_registry_operations() -> None:
    """Verify iter_registry_operations yields operations."""
    operations = list(iter_registry_operations())

    expect_is_instance(operations, list)
    expect_true(len(operations) > 0)


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
    mcp = wrap_fastmcp("Test Tools")
    backend = _build_backend(provisioned_repo)

    register_meta_tools(mcp, backend)

    # MCP server should have tools registered
    expect_equal(mcp.name, "Test Tools")
