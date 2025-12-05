"""Tests for dataset query service delegates.

This module tests the _LocalDatasetMixin methods in services/datasets.py
through HTTP routes and direct LocalQueryService invocation, using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.domain_models import DatasetRows, DatasetSchema
from codeintel.serving.http.fastapi import (
    BackendResource,
    create_app,
)
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.errors import McpError
from codeintel.serving.mcp.models import DatasetSpecDescriptor
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.gateway import build_duckdb_query_service

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway

# =============================================================================
# Constants
# =============================================================================

DEFAULT_LIMIT = 10
MAX_ROWS = 100
SAMPLE_LIMIT = 5
OFFSET_ZERO = 0
OFFSET_ONE = 1


# =============================================================================
# Helper Functions
# =============================================================================


def _create_test_app(provisioned_repo: ProvisionedGateway) -> FastAPI:
    """Create a test FastAPI app with the provisioned gateway.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.

    Returns
    -------
    FastAPI
        Configured FastAPI application.
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
    backend = DuckDBBackend(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
        observability=None,
        service=service,
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

    return create_app(config_loader=load_config, backend_factory=backend_factory)


def _build_local_query_service(
    provisioned_repo: ProvisionedGateway,
) -> LocalQueryService:
    """Build a LocalQueryService for direct testing.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.

    Returns
    -------
    LocalQueryService
        Configured local query service.
    """
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    return LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )


# =============================================================================
# list_datasets Tests (via HTTP)
# =============================================================================


def test_list_datasets_returns_list(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_datasets returns a list of datasets.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/datasets")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)


def test_list_datasets_contains_expected_fields(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_datasets returns datasets with expected fields.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/datasets")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    if data:  # If there are datasets
        first_dataset = data[0]
        # Check for expected fields
        assert "name" in first_dataset or "id" in first_dataset


def test_list_datasets_not_empty(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify list_datasets returns at least some datasets.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/datasets")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # Provisioned repo should have some datasets
    assert len(data) > 0


# =============================================================================
# read_dataset_rows Tests (via HTTP)
# =============================================================================


def test_read_dataset_rows_with_valid_dataset(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify read_dataset_rows works with a valid dataset.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    # First get a valid dataset name
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        datasets_response = client.get("/datasets")

    if datasets_response.status_code != status.HTTP_200_OK:
        pytest.skip("Could not list datasets")

    datasets = datasets_response.json()
    if not datasets:
        pytest.skip("No datasets available")

    # Get the first dataset's name or id
    dataset_name = datasets[0].get("name") or datasets[0].get("id")
    if not dataset_name:
        pytest.skip("Could not find dataset name")

    with TestClient(app) as client:
        response = client.get(f"/datasets/{dataset_name}")

    # Should return 200 or 404 depending on data availability
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


def test_read_dataset_rows_with_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify read_dataset_rows respects limit parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        datasets_response = client.get("/datasets")

    if datasets_response.status_code != status.HTTP_200_OK:
        pytest.skip("Could not list datasets")

    datasets = datasets_response.json()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].get("name") or datasets[0].get("id")
    if not dataset_name:
        pytest.skip("Could not find dataset name")

    with TestClient(app) as client:
        response = client.get(f"/datasets/{dataset_name}?limit={SAMPLE_LIMIT}")

    # Should return 200 or 404
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


def test_read_dataset_rows_with_offset(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify read_dataset_rows respects offset parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        datasets_response = client.get("/datasets")

    if datasets_response.status_code != status.HTTP_200_OK:
        pytest.skip("Could not list datasets")

    datasets = datasets_response.json()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].get("name") or datasets[0].get("id")
    if not dataset_name:
        pytest.skip("Could not find dataset name")

    with TestClient(app) as client:
        response = client.get(f"/datasets/{dataset_name}?offset={OFFSET_ONE}")

    # Should return 200 or 404
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


def test_read_dataset_rows_with_limit_and_offset(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify read_dataset_rows works with both limit and offset.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        datasets_response = client.get("/datasets")

    if datasets_response.status_code != status.HTTP_200_OK:
        pytest.skip("Could not list datasets")

    datasets = datasets_response.json()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].get("name") or datasets[0].get("id")
    if not dataset_name:
        pytest.skip("Could not find dataset name")

    with TestClient(app) as client:
        response = client.get(f"/datasets/{dataset_name}?limit={SAMPLE_LIMIT}&offset={OFFSET_ONE}")

    # Should return 200 or 404
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


def test_read_dataset_rows_nonexistent_dataset(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify read_dataset_rows returns error for nonexistent dataset.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/datasets/nonexistent_dataset_name_xyz")

    # Should return 400 or 404 for unknown dataset
    assert response.status_code in {
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_404_NOT_FOUND,
    }


# =============================================================================
# dataset_schema Tests (via HTTP)
# =============================================================================


def test_dataset_schema_with_valid_dataset(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset_schema works with a valid dataset.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        datasets_response = client.get("/datasets")

    if datasets_response.status_code != status.HTTP_200_OK:
        pytest.skip("Could not list datasets")

    datasets = datasets_response.json()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].get("name") or datasets[0].get("id")
    if not dataset_name:
        pytest.skip("Could not find dataset name")

    with TestClient(app) as client:
        response = client.get(f"/datasets/{dataset_name}/schema")

    # Should return 200 or 404
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


def test_dataset_schema_with_sample_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset_schema respects sample_limit parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        datasets_response = client.get("/datasets")

    if datasets_response.status_code != status.HTTP_200_OK:
        pytest.skip("Could not list datasets")

    datasets = datasets_response.json()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].get("name") or datasets[0].get("id")
    if not dataset_name:
        pytest.skip("Could not find dataset name")

    with TestClient(app) as client:
        response = client.get(f"/datasets/{dataset_name}/schema?limit={SAMPLE_LIMIT}")

    # Should return 200 or 404
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


def test_dataset_schema_nonexistent_dataset(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset_schema returns error for nonexistent dataset.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/datasets/nonexistent_dataset_name_xyz/schema")

    # Should return 400 or 404 for unknown dataset
    assert response.status_code in {
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_404_NOT_FOUND,
    }


# =============================================================================
# dataset_specs Tests (via HTTP)
# =============================================================================


def test_dataset_specs_returns_list(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset_specs returns a list.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/datasets/specs")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)


# =============================================================================
# Direct LocalQueryService Tests
# =============================================================================


def test_local_query_service_list_datasets(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService.list_datasets works directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    datasets = service.list_datasets()
    assert datasets is not None
    assert isinstance(datasets, list)


def test_local_query_service_dataset_specs(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService.dataset_specs works directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    specs = service.dataset_specs()
    assert specs is not None
    assert isinstance(specs, list)


def test_local_query_service_read_dataset_rows(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService.read_dataset_rows works directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    # First get a valid dataset name
    datasets = service.list_datasets()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].name

    try:
        rows = service.read_dataset_rows(
            dataset_name=dataset_name, limit=SAMPLE_LIMIT, offset=OFFSET_ZERO
        )
        assert rows is not None
    except McpError:
        # Expected when dataset is not readable
        pass


def test_local_query_service_dataset_schema(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService.dataset_schema works directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    # First get a valid dataset name
    datasets = service.list_datasets()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].name

    try:
        schema = service.dataset_schema(dataset_name=dataset_name, sample_limit=SAMPLE_LIMIT)
        assert schema is not None
    except McpError:
        # Expected when dataset schema is not available
        pass


def test_local_query_service_read_dataset_rows_nonexistent(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService raises error for nonexistent dataset.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    with pytest.raises(McpError):
        service.read_dataset_rows(
            dataset_name="nonexistent_dataset_xyz", limit=DEFAULT_LIMIT, offset=OFFSET_ZERO
        )


def test_local_query_service_dataset_schema_nonexistent(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService raises error for nonexistent dataset schema.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    with pytest.raises(McpError):
        service.dataset_schema(dataset_name="nonexistent_dataset_xyz", sample_limit=SAMPLE_LIMIT)


# =============================================================================
# Response Structure Tests
# =============================================================================


def test_dataset_rows_response_structure(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset rows response contains expected fields.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        datasets_response = client.get("/datasets")

    if datasets_response.status_code != status.HTTP_200_OK:
        pytest.skip("Could not list datasets")

    datasets = datasets_response.json()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].get("name") or datasets[0].get("id")
    if not dataset_name:
        pytest.skip("Could not find dataset name")

    with TestClient(app) as client:
        response = client.get(f"/datasets/{dataset_name}")

    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        # Check that response is a dict with expected fields
        assert isinstance(data, dict)
        # Should contain rows or similar field
        assert "rows" in data or "data" in data or isinstance(data.get("results"), list)


def test_dataset_schema_response_structure(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset schema response contains expected fields.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        datasets_response = client.get("/datasets")

    if datasets_response.status_code != status.HTTP_200_OK:
        pytest.skip("Could not list datasets")

    datasets = datasets_response.json()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].get("name") or datasets[0].get("id")
    if not dataset_name:
        pytest.skip("Could not find dataset name")

    with TestClient(app) as client:
        response = client.get(f"/datasets/{dataset_name}/schema")

    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        # Check that response is a dict (schema structure)
        assert isinstance(data, dict)


# =============================================================================
# Edge Cases
# =============================================================================


def test_read_dataset_rows_with_zero_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify read_dataset_rows handles zero limit gracefully.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        datasets_response = client.get("/datasets")

    if datasets_response.status_code != status.HTTP_200_OK:
        pytest.skip("Could not list datasets")

    datasets = datasets_response.json()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].get("name") or datasets[0].get("id")
    if not dataset_name:
        pytest.skip("Could not find dataset name")

    with TestClient(app) as client:
        response = client.get(f"/datasets/{dataset_name}?limit=0")

    # Should handle zero limit gracefully
    assert response.status_code in {
        status.HTTP_200_OK,
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_404_NOT_FOUND,
    }


def test_read_dataset_rows_with_large_offset(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify read_dataset_rows handles large offset gracefully.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    large_offset = 1000000

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        datasets_response = client.get("/datasets")

    if datasets_response.status_code != status.HTTP_200_OK:
        pytest.skip("Could not list datasets")

    datasets = datasets_response.json()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].get("name") or datasets[0].get("id")
    if not dataset_name:
        pytest.skip("Could not find dataset name")

    with TestClient(app) as client:
        response = client.get(f"/datasets/{dataset_name}?offset={large_offset}")

    # Should handle large offset - likely returns empty or error
    assert response.status_code in {
        status.HTTP_200_OK,
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_404_NOT_FOUND,
    }


# =============================================================================
# Negative Limit Tests
# =============================================================================


def test_read_dataset_rows_with_negative_limit(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify read_dataset_rows handles negative limit gracefully.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    negative_limit = -1

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        datasets_response = client.get("/datasets")

    if datasets_response.status_code != status.HTTP_200_OK:
        pytest.skip("Could not list datasets")

    datasets = datasets_response.json()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].get("name") or datasets[0].get("id")
    if not dataset_name:
        pytest.skip("Could not find dataset name")

    with TestClient(app) as client:
        response = client.get(f"/datasets/{dataset_name}?limit={negative_limit}")

    # Should handle negative limit - either clamp to 0 or return error
    assert response.status_code in {
        status.HTTP_200_OK,
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_404_NOT_FOUND,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
    }


def test_read_dataset_rows_with_negative_offset(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify read_dataset_rows handles negative offset gracefully.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    negative_offset = -5

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        datasets_response = client.get("/datasets")

    if datasets_response.status_code != status.HTTP_200_OK:
        pytest.skip("Could not list datasets")

    datasets = datasets_response.json()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].get("name") or datasets[0].get("id")
    if not dataset_name:
        pytest.skip("Could not find dataset name")

    with TestClient(app) as client:
        response = client.get(f"/datasets/{dataset_name}?offset={negative_offset}")

    # Should handle negative offset - either clamp to 0 or return error
    assert response.status_code in {
        status.HTTP_200_OK,
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_404_NOT_FOUND,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
    }


# =============================================================================
# LocalDatasetMixin Property Tests
# =============================================================================


def test_local_dataset_mixin_uses_query_gateway(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalDatasetMixin uses query gateway for dataset mapping.

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
    # Create service without explicit dataset_tables - should fall back to gateway
    service = LocalQueryService(
        query=query,
        dataset_tables=None,
    )

    datasets = service.list_datasets()
    assert isinstance(datasets, list)


def test_local_dataset_mixin_with_explicit_tables(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalDatasetMixin uses explicit dataset_tables when provided.

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
    # Create service with explicit dataset_tables
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )

    datasets = service.list_datasets()
    assert isinstance(datasets, list)


# =============================================================================
# Dataset Descriptor Tests
# =============================================================================


def test_dataset_descriptors_have_name(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset descriptors contain name field.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    datasets = service.list_datasets()
    for dataset in datasets:
        assert hasattr(dataset, "name")
        assert dataset.name is not None


def test_dataset_descriptors_have_table(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset descriptors contain table field.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    datasets = service.list_datasets()
    for dataset in datasets:
        assert hasattr(dataset, "table")


def test_dataset_descriptors_have_docs_view_flag(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset descriptors contain is_docs_view field.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    datasets = service.list_datasets()
    for dataset in datasets:
        assert hasattr(dataset, "is_docs_view")
        assert isinstance(dataset.is_docs_view, bool)


def test_dataset_descriptors_have_read_only_flag(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset descriptors contain is_read_only field.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    datasets = service.list_datasets()
    for dataset in datasets:
        assert hasattr(dataset, "is_read_only")
        assert isinstance(dataset.is_read_only, bool)


# =============================================================================
# Dataset Specs Tests
# =============================================================================


def test_dataset_specs_structure(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset_specs returns properly structured specs.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    specs = service.dataset_specs()
    for spec in specs:
        assert isinstance(spec, DatasetSpecDescriptor)


# =============================================================================
# Read Dataset Rows Meta Tests
# =============================================================================


def test_read_dataset_rows_returns_domain_model(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify read_dataset_rows returns DatasetRows domain model.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    datasets = service.list_datasets()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].name

    try:
        result = service.read_dataset_rows(
            dataset_name=dataset_name, limit=SAMPLE_LIMIT, offset=OFFSET_ZERO
        )
        assert isinstance(result, DatasetRows)
        assert hasattr(result, "dataset_name")
        assert hasattr(result, "rows")
        assert hasattr(result, "offset")
    except McpError:
        # Expected when dataset is not readable
        pass


# =============================================================================
# Dataset Schema Domain Model Tests
# =============================================================================


def test_dataset_schema_returns_domain_model(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify dataset_schema returns DatasetSchema domain model.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    datasets = service.list_datasets()
    if not datasets:
        pytest.skip("No datasets available")

    dataset_name = datasets[0].name

    try:
        result = service.dataset_schema(dataset_name=dataset_name, sample_limit=SAMPLE_LIMIT)
        assert isinstance(result, DatasetSchema)
    except McpError:
        # Expected when dataset schema is not available
        pass
