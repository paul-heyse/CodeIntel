"""Comprehensive tests for MCP backend implementations.

This module tests the DuckDBBackend using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import build_duckdb_query_service

if TYPE_CHECKING:
    from tests._helpers.fixtures import ProvisionedGateway

# Test constants
CUSTOM_DEFAULT_LIMIT = 25
CUSTOM_MAX_ROWS = 250


# =============================================================================
# DuckDBBackend Construction Tests
# =============================================================================


def test_duckdb_backend_creation(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify DuckDBBackend can be constructed with provisioned gateway.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    assert backend.gateway is provisioned_repo.gateway
    assert backend.repo == provisioned_repo.repo
    assert backend.commit == provisioned_repo.commit


def test_duckdb_backend_with_custom_limits(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify DuckDBBackend respects custom limits.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    custom_limits = BackendLimits(
        default_limit=CUSTOM_DEFAULT_LIMIT, max_rows_per_call=CUSTOM_MAX_ROWS
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=custom_limits,
    )

    assert backend.limits.default_limit == CUSTOM_DEFAULT_LIMIT
    assert backend.limits.max_rows_per_call == CUSTOM_MAX_ROWS


def test_duckdb_backend_with_service_override(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify DuckDBBackend accepts service_override.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )

    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    assert backend.service is service


# =============================================================================
# Dataset Operations Tests
# =============================================================================


def test_duckdb_backend_list_datasets(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_datasets returns dataset descriptors.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    datasets = backend.list_datasets()

    assert isinstance(datasets, list)


def test_duckdb_backend_dataset_specs(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset_specs returns spec descriptors.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    specs = backend.dataset_specs()

    assert isinstance(specs, list)


# =============================================================================
# Function Operations Tests
# =============================================================================


def test_duckdb_backend_list_high_risk_functions(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_high_risk_functions works with real gateway.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    response = backend.list_high_risk_functions(min_risk=0.5, limit=10)

    assert hasattr(response, "functions")


def test_duckdb_backend_list_high_risk_functions_with_tested_only(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_high_risk_functions accepts tested_only filter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    response = backend.list_high_risk_functions(
        min_risk=0.5, limit=10, tested_only=True
    )

    assert hasattr(response, "functions")


# =============================================================================
# Subsystem Operations Tests
# =============================================================================


def test_duckdb_backend_list_subsystems(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystems works with architecture gateway.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        service_override=service,
    )

    response = backend.list_subsystems(limit=10)

    assert hasattr(response, "subsystems")


def test_duckdb_backend_list_subsystems_with_role_filter(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystems accepts role filter.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        service_override=service,
    )

    response = backend.list_subsystems(limit=10, role="test_role")

    assert hasattr(response, "subsystems")


def test_duckdb_backend_list_subsystems_with_query_filter(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify list_subsystems accepts query filter.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        service_override=service,
    )

    response = backend.list_subsystems(limit=10, q="test")

    assert hasattr(response, "subsystems")


def test_duckdb_backend_search_subsystems(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify search_subsystems works with architecture gateway.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        service_override=service,
    )

    response = backend.search_subsystems(limit=10)

    assert hasattr(response, "results") or hasattr(response, "subsystems")


# =============================================================================
# Service Access Tests
# =============================================================================


def test_duckdb_backend_service_attribute(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify DuckDBBackend exposes service attribute.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        service_override=service,
    )

    assert backend.service is not None


def test_duckdb_backend_limits_attribute(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify DuckDBBackend exposes limits attribute.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    assert backend.limits is not None
    assert hasattr(backend.limits, "default_limit")
    assert hasattr(backend.limits, "max_rows_per_call")
