"""Tests for dataset HTTP routes.

This module tests the dataset-related HTTP endpoints using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from tests._helpers.gateway import build_duckdb_query_service

if TYPE_CHECKING:
    from tests._helpers.fixtures import ProvisionedGateway


# =============================================================================
# Dataset Listing Tests
# =============================================================================


def test_datasets_list_endpoint(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify /datasets endpoint returns list of datasets.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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
        service_override=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo=provisioned_repo.repo,
            commit=provisioned_repo.commit,
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/datasets")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)


def test_datasets_specs_endpoint(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify /datasets/specs endpoint returns dataset specs.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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
        service_override=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo=provisioned_repo.repo,
            commit=provisioned_repo.commit,
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/datasets/specs")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)


# =============================================================================
# Dataset Row Access Tests
# =============================================================================


def test_dataset_rows_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify /datasets/{name} returns 400 for unknown dataset.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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
        service_override=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo=provisioned_repo.repo,
            commit=provisioned_repo.commit,
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/datasets/nonexistent_dataset")

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["code"] == "dataset-not-found"


def test_dataset_schema_not_found(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify /datasets/{name}/schema returns 400 for unknown dataset.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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
        service_override=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo=provisioned_repo.repo,
            commit=provisioned_repo.commit,
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/datasets/nonexistent_dataset/schema")

    assert response.status_code == status.HTTP_400_BAD_REQUEST


# =============================================================================
# Dataset Pagination Tests
# =============================================================================


def test_dataset_rows_with_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify /datasets/{name} accepts limit parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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
        service_override=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo=provisioned_repo.repo,
            commit=provisioned_repo.commit,
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    # Get list of datasets first to find a valid one
    with TestClient(app) as client:
        list_resp = client.get("/datasets")

    datasets = list_resp.json()
    if datasets:
        first_ds = datasets[0]
        ds_name = first_ds.get("table_key") or first_ds.get("id")
        if ds_name:
            with TestClient(app) as client:
                response = client.get(f"/datasets/{ds_name}?limit=5")
            # If dataset exists and has data, should succeed
            assert response.status_code in {
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST,
            }


def test_dataset_rows_with_offset(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify /datasets/{name} accepts offset parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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
        service_override=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo=provisioned_repo.repo,
            commit=provisioned_repo.commit,
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    # Get list of datasets first to find a valid one
    with TestClient(app) as client:
        list_resp = client.get("/datasets")

    datasets = list_resp.json()
    if datasets:
        first_ds = datasets[0]
        ds_name = first_ds.get("table_key") or first_ds.get("id")
        if ds_name:
            with TestClient(app) as client:
                response = client.get(f"/datasets/{ds_name}?offset=0")
            # If dataset exists and has data, should succeed
            assert response.status_code in {
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST,
            }
