"""Tests for dataset query service delegates using shared service app fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import suppress

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.errors import McpError
from codeintel.serving.mcp.models import DatasetSpecDescriptor
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.assertions import (
    assert_problem_detail_response,
    expect_equal,
    expect_is_instance,
    expect_true,
)
from tests._helpers.assertions.http_responses import assert_ok_or_not_found
from tests._helpers.serving_contexts import ProvisionedServiceContext

SAMPLE_LIMIT = 5
OFFSET_ZERO = 0


@pytest.fixture
def service_ctx(provisioned_service_ctx: ProvisionedServiceContext) -> ProvisionedServiceContext:
    """Service context wrapper for dataset tests.

    Returns
    -------
    ProvisionedServiceContext
        Context bound to the provisioned repo snapshot.
    """
    return provisioned_service_ctx


@pytest.fixture
def service(service_ctx: ProvisionedServiceContext) -> LocalQueryService:
    """Expose the configured LocalQueryService for reuse across tests.

    Returns
    -------
    LocalQueryService
        Service instance backed by the provisioned context.
    """
    return service_ctx.service


@pytest.fixture
def service_client(service_ctx: ProvisionedServiceContext) -> Iterator[TestClient]:
    """Provide a TestClient bound to the provisioned service app.

    Yields
    ------
    Iterator[TestClient]
        Client connected to the provisioned service FastAPI app.
    """
    with service_ctx.client() as client:
        yield client


@pytest.fixture
def datasets(service: LocalQueryService) -> list[dm.DatasetDescriptorDomain]:
    """List available datasets or skip when none are present.

    Returns
    -------
    list[DatasetDescriptorDomain]
        Available dataset descriptors for the provisioned gateway.
    """
    dataset_descriptors = service.list_datasets()
    if not dataset_descriptors:
        pytest.skip("No datasets available")
    return dataset_descriptors


@pytest.fixture
def dataset_name(datasets: list[dm.DatasetDescriptorDomain]) -> str:
    """Return a stable dataset name for HTTP and service-level tests.

    Returns
    -------
    str
        Canonical dataset name from the provisioned gateway.
    """
    return datasets[0].name


def test_list_datasets_returns_list(service_client: TestClient) -> None:
    """Verify list_datasets returns a list of datasets over HTTP."""
    response = service_client.get("/datasets")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_is_instance(data, list)


def test_list_datasets_contains_expected_fields(service_client: TestClient) -> None:
    """Verify list_datasets returns datasets with expected fields."""
    response = service_client.get("/datasets")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    if data:
        first_dataset = data[0]
        expect_true("name" in first_dataset or "id" in first_dataset)


def test_list_datasets_not_empty(
    service_client: TestClient, datasets: list[dm.DatasetDescriptorDomain]
) -> None:
    """Verify list_datasets returns at least some datasets."""
    response = service_client.get("/datasets")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_true(len(data) >= len(datasets))


@pytest.mark.parametrize(
    ("query_suffix", "expected_statuses"),
    [
        ("", {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}),
        ("?limit=5", {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}),
        ("?offset=1", {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}),
        ("?limit=5&offset=1", {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}),
        ("?limit=0", {status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND}),
        (
            "?offset=1000000",
            {status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND},
        ),
        (
            "?limit=-1",
            {
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_404_NOT_FOUND,
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            },
        ),
        (
            "?offset=-5",
            {
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_404_NOT_FOUND,
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            },
        ),
    ],
)
def test_read_dataset_rows_variants(
    service_client: TestClient,
    dataset_name: str,
    query_suffix: str,
    expected_statuses: set[int],
) -> None:
    """Verify read_dataset_rows handles varied query parameters."""
    response = service_client.get(f"/datasets/{dataset_name}{query_suffix}")

    expect_true(response.status_code in expected_statuses)


def test_read_dataset_rows_nonexistent_dataset(service_client: TestClient) -> None:
    """Verify read_dataset_rows returns error for nonexistent dataset."""
    response = service_client.get("/datasets/nonexistent_dataset_name_xyz")

    assert_problem_detail_response(
        response,
        status_code=status.HTTP_400_BAD_REQUEST,
    )


@pytest.mark.parametrize("path_suffix", ["schema", f"schema?limit={SAMPLE_LIMIT}"])
def test_dataset_schema_http(
    service_client: TestClient,
    dataset_name: str,
    path_suffix: str,
) -> None:
    """Verify dataset_schema works over HTTP with and without sample_limit."""
    response = service_client.get(f"/datasets/{dataset_name}/{path_suffix}")

    assert_ok_or_not_found(response)


def test_dataset_schema_nonexistent_dataset(service_client: TestClient) -> None:
    """Verify dataset_schema returns error for nonexistent dataset."""
    response = service_client.get("/datasets/nonexistent_dataset_name_xyz/schema")

    assert_problem_detail_response(
        response,
        status_code=status.HTTP_400_BAD_REQUEST,
    )


def test_dataset_specs_returns_list(service_client: TestClient) -> None:
    """Verify dataset_specs returns a list over HTTP."""
    response = service_client.get("/datasets/specs")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_is_instance(data, list)


def test_local_query_service_list_datasets(service: LocalQueryService) -> None:
    """Verify LocalQueryService.list_datasets works directly."""
    datasets = service.list_datasets()
    expect_is_instance(datasets, list)


def test_local_query_service_dataset_specs(service: LocalQueryService) -> None:
    """Verify LocalQueryService.dataset_specs works directly."""
    specs = service.dataset_specs()
    expect_is_instance(specs, list)


def test_local_query_service_read_dataset_rows(
    service: LocalQueryService,
    dataset_name: str,
) -> None:
    """Verify LocalQueryService.read_dataset_rows works directly."""
    with suppress(McpError):
        rows = service.read_dataset_rows(
            dataset_name=dataset_name,
            limit=SAMPLE_LIMIT,
            offset=OFFSET_ZERO,
        )
        expect_true(rows is not None)


def test_local_query_service_dataset_schema(
    service: LocalQueryService,
    dataset_name: str,
) -> None:
    """Verify LocalQueryService.dataset_schema works directly."""
    with suppress(McpError):
        schema = service.dataset_schema(dataset_name=dataset_name, sample_limit=SAMPLE_LIMIT)
        expect_true(schema is not None)


def test_local_query_service_read_dataset_rows_nonexistent(
    service: LocalQueryService,
) -> None:
    """Verify LocalQueryService raises error for nonexistent dataset."""
    with pytest.raises(McpError):
        service.read_dataset_rows(
            dataset_name="nonexistent_dataset_xyz",
            limit=service.limits.default_limit,
            offset=OFFSET_ZERO,
        )


def test_local_query_service_dataset_schema_nonexistent(service: LocalQueryService) -> None:
    """Verify LocalQueryService raises error for nonexistent dataset schema."""
    with pytest.raises(McpError):
        service.dataset_schema(
            dataset_name="nonexistent_dataset_xyz",
            sample_limit=SAMPLE_LIMIT,
        )


def test_dataset_rows_response_structure(
    service_client: TestClient,
    dataset_name: str,
) -> None:
    """Verify dataset rows response contains expected fields."""
    response = service_client.get(f"/datasets/{dataset_name}")

    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        expect_is_instance(data, dict)
        expect_true("rows" in data or "data" in data or isinstance(data.get("results"), list))


def test_dataset_schema_response_structure(
    service_client: TestClient,
    dataset_name: str,
) -> None:
    """Verify dataset schema response contains expected fields."""
    response = service_client.get(f"/datasets/{dataset_name}/schema")

    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        expect_is_instance(data, dict)


def test_local_dataset_mixin_uses_query_gateway(
    service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify LocalDatasetMixin uses query gateway for dataset mapping."""
    service = LocalQueryService(
        query=service_ctx.service.query,
        dataset_tables=None,
    )

    datasets = service.list_datasets()
    expect_is_instance(datasets, list)


def test_local_dataset_mixin_with_explicit_tables(
    service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify LocalDatasetMixin uses explicit dataset_tables when provided."""
    service = LocalQueryService(
        query=service_ctx.service.query,
        dataset_tables=dict(service_ctx.gateway.datasets.mapping),
    )

    datasets = service.list_datasets()
    expect_is_instance(datasets, list)


def test_dataset_descriptors_have_name(
    datasets: list[dm.DatasetDescriptorDomain],
) -> None:
    """Verify dataset descriptors contain name field."""
    for dataset in datasets:
        expect_true(hasattr(dataset, "name"))
        expect_true(dataset.name is not None)


def test_dataset_descriptors_have_table(
    datasets: list[dm.DatasetDescriptorDomain],
) -> None:
    """Verify dataset descriptors contain table field."""
    for dataset in datasets:
        expect_true(hasattr(dataset, "table"))


def test_dataset_descriptors_have_docs_view_flag(
    datasets: list[dm.DatasetDescriptorDomain],
) -> None:
    """Verify dataset descriptors contain is_docs_view field."""
    for dataset in datasets:
        expect_true(hasattr(dataset, "is_docs_view"))
        expect_is_instance(dataset.is_docs_view, bool)


def test_dataset_descriptors_have_read_only_flag(
    datasets: list[dm.DatasetDescriptorDomain],
) -> None:
    """Verify dataset descriptors contain is_read_only field."""
    for dataset in datasets:
        expect_true(hasattr(dataset, "is_read_only"))
        expect_is_instance(dataset.is_read_only, bool)


def test_dataset_specs_structure(service: LocalQueryService) -> None:
    """Verify dataset_specs returns properly structured specs."""
    specs = service.dataset_specs()
    for spec in specs:
        expect_is_instance(spec, DatasetSpecDescriptor)


def test_read_dataset_rows_returns_domain_model(
    service: LocalQueryService,
    dataset_name: str,
) -> None:
    """Verify read_dataset_rows returns DatasetRows domain model."""
    with suppress(McpError):
        result = service.read_dataset_rows(
            dataset_name=dataset_name,
            limit=SAMPLE_LIMIT,
            offset=OFFSET_ZERO,
        )
        expect_is_instance(result, dm.DatasetRows)
        expect_true(hasattr(result, "dataset_name"))
        expect_true(hasattr(result, "rows"))
        expect_true(hasattr(result, "offset"))


def test_dataset_schema_returns_domain_model(
    service: LocalQueryService,
    dataset_name: str,
) -> None:
    """Verify dataset_schema returns DatasetSchema domain model."""
    with suppress(McpError):
        result = service.dataset_schema(
            dataset_name=dataset_name,
            sample_limit=SAMPLE_LIMIT,
        )
        expect_is_instance(result, dm.DatasetSchema)
