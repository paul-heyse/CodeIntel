"""Tests for function HTTP routes.

This module tests the function-related HTTP endpoints using real gateways.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.routes.functions import RouterOptions
from tests._helpers.assertions import expect_equal, expect_false, expect_in, expect_true

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway


# =============================================================================
# High Risk Functions Tests
# =============================================================================


def test_high_risk_functions_endpoint(
    functions_http_client: TestClient,
) -> None:
    """/functions/high-risk endpoint returns results."""
    response = functions_http_client.get("/functions/high-risk")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("functions", data)


def test_high_risk_functions_with_min_risk(
    functions_http_client: TestClient,
) -> None:
    """/functions/high-risk accepts min_risk parameter."""
    response = functions_http_client.get("/functions/high-risk?min_risk=0.5")

    expect_equal(response.status_code, status.HTTP_200_OK)


def test_high_risk_functions_with_limit(
    functions_http_client: TestClient,
) -> None:
    """/functions/high-risk accepts limit parameter."""
    response = functions_http_client.get("/functions/high-risk?limit=5")

    expect_equal(response.status_code, status.HTTP_200_OK)


def test_high_risk_functions_with_tested_only(
    functions_http_client: TestClient,
) -> None:
    """/functions/high-risk accepts tested_only parameter."""
    response = functions_http_client.get("/functions/high-risk?tested_only=true")

    expect_equal(response.status_code, status.HTTP_200_OK)


# =============================================================================
# Function Summary Tests
# =============================================================================


def test_function_summary_missing_params(
    functions_http_client: TestClient,
) -> None:
    """/function/summary returns error when no identifier provided."""
    response = functions_http_client.get("/function/summary")

    # Should return 400 because no identifier was provided
    expect_equal(response.status_code, status.HTTP_400_BAD_REQUEST)


# =============================================================================
# Router Options Tests
# =============================================================================


def test_router_options_default() -> None:
    """Verify RouterOptions defaults to auto_pipeline=False."""
    options = RouterOptions()
    expect_false(options.auto_pipeline)


def test_router_options_with_auto_pipeline() -> None:
    """Verify RouterOptions accepts auto_pipeline=True."""
    options = RouterOptions(auto_pipeline=True)
    expect_true(options.auto_pipeline)


def test_app_with_auto_pipeline_options(
    provisioned_repo: ProvisionedGateway,
    make_http_app: Callable[..., FastAPI],
) -> None:
    """Verify create_app works with auto_pipeline option."""
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    app = make_http_app(
        gateway=provisioned_repo.gateway,
        snapshot=(provisioned_repo.repo, provisioned_repo.commit),
        limits=limits,
        auto_pipeline=True,
    )

    with TestClient(app) as client:
        response = client.get("/functions/high-risk")

    expect_equal(response.status_code, status.HTTP_200_OK)
