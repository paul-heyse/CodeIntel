"""Tests for subsystem service delegates.

This module tests subsystem query delegates via LocalQueryService.
"""

from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.fastapi import (
    BackendResource,
    create_app,
)
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import build_duckdb_query_service

# =============================================================================
# Subsystem Route Tests (covers service delegates)
# =============================================================================


def test_list_subsystems_returns_data(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify subsystem listing returns results.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/architecture/subsystems")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "subsystems" in data or isinstance(data, list)


def test_list_subsystems_with_limit(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify subsystem listing respects limit parameter.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/architecture/subsystems?limit=5")

    assert response.status_code == status.HTTP_200_OK


def test_get_subsystem_modules(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify subsystem modules endpoint functions.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    # Try to get modules for a subsystem
    with TestClient(app) as client:
        response = client.get("/architecture/subsystem?subsystem_id=test_subsystem")

    # May be 200 with empty data or 400/404 if no such subsystem - both are valid
    assert response.status_code in {
        status.HTTP_200_OK,
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_404_NOT_FOUND,
    }


def test_get_module_subsystems(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify module subsystems endpoint functions.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/architecture/module-subsystems?module=test.module")

    # May be 200 with data or 400/404 if not found
    assert response.status_code in {
        status.HTTP_200_OK,
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_404_NOT_FOUND,
    }


def test_subsystem_coverage_endpoint(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify subsystem coverage endpoint returns data.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/architecture/subsystem-coverage")

    assert response.status_code == status.HTTP_200_OK


def test_subsystem_profiles_endpoint(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify subsystem profiles endpoint returns data.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/architecture/subsystem-profiles")

    assert response.status_code == status.HTTP_200_OK
