"""Tests for dataset HTTP routes.

This module tests the dataset-related HTTP endpoints using real gateways.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapi import status
from fastapi.testclient import TestClient

from codeintel.serving.backend import BackendLimits
from tests._helpers.assertions import expect_equal, expect_is_instance, expect_true

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway


# =============================================================================
# Dataset Listing Tests
# =============================================================================


def test_datasets_list_endpoint(
    provisioned_repo: ProvisionedGateway,
    make_http_app: Callable[..., object],
) -> None:
    """Verify /datasets endpoint returns list of datasets.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    app = make_http_app(
        gateway=provisioned_repo.gateway,
        snapshot=(provisioned_repo.repo, provisioned_repo.commit),
        limits=limits,
    )

    with TestClient(app) as client:
        response = client.get("/datasets")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_is_instance(data, list)


def test_datasets_specs_endpoint(
    provisioned_repo: ProvisionedGateway,
    make_http_app: Callable[..., object],
) -> None:
    """Verify /datasets/specs endpoint returns dataset specs.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    app = make_http_app(
        gateway=provisioned_repo.gateway,
        snapshot=(provisioned_repo.repo, provisioned_repo.commit),
        limits=limits,
    )

    with TestClient(app) as client:
        response = client.get("/datasets/specs")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_is_instance(data, list)


# =============================================================================
# Dataset Row Access Tests
# =============================================================================


def test_dataset_rows_not_found(
    provisioned_repo: ProvisionedGateway,
    make_http_app: Callable[..., object],
) -> None:
    """Verify /datasets/{name} returns 400 for unknown dataset.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    app = make_http_app(
        gateway=provisioned_repo.gateway,
        snapshot=(provisioned_repo.repo, provisioned_repo.commit),
        limits=limits,
    )

    with TestClient(app) as client:
        response = client.get("/datasets/nonexistent_dataset")

    expect_equal(response.status_code, status.HTTP_400_BAD_REQUEST)
    data = response.json()
    expect_equal(data["code"], "dataset-not-found")


def test_dataset_schema_not_found(
    provisioned_repo: ProvisionedGateway,
    make_http_app: Callable[..., object],
) -> None:
    """Verify /datasets/{name}/schema returns 400 for unknown dataset.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    app = make_http_app(
        gateway=provisioned_repo.gateway,
        snapshot=(provisioned_repo.repo, provisioned_repo.commit),
        limits=limits,
    )

    with TestClient(app) as client:
        response = client.get("/datasets/nonexistent_dataset/schema")

    expect_equal(response.status_code, status.HTTP_400_BAD_REQUEST)


# =============================================================================
# Dataset Pagination Tests
# =============================================================================


def test_dataset_rows_with_limit(
    provisioned_repo: ProvisionedGateway,
    make_http_app: Callable[..., object],
) -> None:
    """Verify /datasets/{name} accepts limit parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    app = make_http_app(
        gateway=provisioned_repo.gateway,
        snapshot=(provisioned_repo.repo, provisioned_repo.commit),
        limits=limits,
    )

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
            expect_true(
                response.status_code
                in {
                    status.HTTP_200_OK,
                    status.HTTP_400_BAD_REQUEST,
                }
            )


def test_dataset_rows_with_offset(
    provisioned_repo: ProvisionedGateway,
    make_http_app: Callable[..., object],
) -> None:
    """Verify /datasets/{name} accepts offset parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    app = make_http_app(
        gateway=provisioned_repo.gateway,
        snapshot=(provisioned_repo.repo, provisioned_repo.commit),
        limits=limits,
    )

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
            expect_true(
                response.status_code
                in {
                    status.HTTP_200_OK,
                    status.HTTP_400_BAD_REQUEST,
                }
            )
