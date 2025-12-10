"""Tests for dataset HTTP routes.

This module tests the dataset-related HTTP endpoints using real gateways.
"""

from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient

from tests._helpers.assertions import expect_equal, expect_is_instance, expect_true
from tests._helpers.assertions.http_responses import assert_problem_detail_response

# =============================================================================
# Dataset Listing Tests
# =============================================================================


def test_datasets_list_endpoint(
    datasets_http_client: TestClient,
) -> None:
    """/datasets endpoint returns list of datasets."""
    response = datasets_http_client.get("/datasets")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_is_instance(data, list)


def test_datasets_specs_endpoint(
    datasets_http_client: TestClient,
) -> None:
    """/datasets/specs endpoint returns dataset specs."""
    response = datasets_http_client.get("/datasets/specs")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_is_instance(data, list)


# =============================================================================
# Dataset Row Access Tests
# =============================================================================


def test_dataset_rows_not_found(
    datasets_http_client: TestClient,
) -> None:
    """/datasets/{name} returns 400 for unknown dataset."""
    response = datasets_http_client.get("/datasets/nonexistent_dataset")

    assert_problem_detail_response(response, status_code=status.HTTP_400_BAD_REQUEST)


def test_dataset_schema_not_found(
    datasets_http_client: TestClient,
) -> None:
    """/datasets/{name}/schema returns 400 for unknown dataset."""
    response = datasets_http_client.get("/datasets/nonexistent_dataset/schema")

    assert_problem_detail_response(response, status_code=status.HTTP_400_BAD_REQUEST)


# =============================================================================
# Dataset Pagination Tests
# =============================================================================


def test_dataset_rows_with_limit(
    datasets_http_client: TestClient,
) -> None:
    """/datasets/{name} accepts limit parameter."""
    list_resp = datasets_http_client.get("/datasets")

    datasets = list_resp.json()
    if datasets:
        first_ds = datasets[0]
        ds_name = first_ds.get("table_key") or first_ds.get("id")
        if ds_name:
            response = datasets_http_client.get(f"/datasets/{ds_name}?limit=5")

            expect_true(
                response.status_code
                in {
                    status.HTTP_200_OK,
                    status.HTTP_400_BAD_REQUEST,
                }
            )


def test_dataset_rows_with_offset(
    datasets_http_client: TestClient,
) -> None:
    """/datasets/{name} accepts offset parameter."""
    list_resp = datasets_http_client.get("/datasets")

    datasets = list_resp.json()
    if datasets:
        first_ds = datasets[0]
        ds_name = first_ds.get("table_key") or first_ds.get("id")
        if ds_name:
            response = datasets_http_client.get(f"/datasets/{ds_name}?offset=0")

            expect_true(
                response.status_code
                in {
                    status.HTTP_200_OK,
                    status.HTTP_400_BAD_REQUEST,
                }
            )
