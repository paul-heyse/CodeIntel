"""Tests for function HTTP routes.

This module tests the function-related HTTP endpoints using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import status

from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.routes.functions import RouterOptions, build_functions_router
from tests._helpers.assertions import expect_equal, expect_false, expect_in, expect_true
from tests._helpers.assertions.http_responses import assert_problem_detail_response
from tests._helpers.serving_routes import RouteAppOptions, service_app_factory_with_routes
from tests.serving.http.client_harness import adapt_route

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tests._helpers.context import TestContext

# =============================================================================
# High Risk Functions Tests
# =============================================================================


@pytest.mark.parametrize(
    ("query", "expect_payload_key"),
    [
        ("", "functions"),
        ("min_risk=0.5", "functions"),
        ("limit=5", "functions"),
        ("tested_only=true", "functions"),
    ],
)
def test_high_risk_functions_endpoint(
    functions_http_client: TestClient,
    query: str,
    expect_payload_key: str,
) -> None:
    """/functions/high-risk endpoint returns results with optional filters."""
    path = "/functions/high-risk"
    if query:
        path = f"{path}?{query}"

    response = functions_http_client.get(path)

    expect_equal(response.status_code, status.HTTP_200_OK)
    payload = response.json()
    expect_in(expect_payload_key, payload)


# =============================================================================
# Function Summary Tests
# =============================================================================


def test_function_summary_missing_params(
    functions_http_client: TestClient,
) -> None:
    """/function/summary returns error when no identifier provided."""
    response = functions_http_client.get("/function/summary")

    assert_problem_detail_response(response, status_code=status.HTTP_400_BAD_REQUEST)


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
    provisioned_repo: TestContext,
) -> None:
    """Verify create_app works with auto_pipeline option."""
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    route_app = service_app_factory_with_routes(
        route_builders=[adapt_route(build_functions_router)],
        backend_source=(provisioned_repo.gateway, (provisioned_repo.repo, provisioned_repo.commit)),
        options=RouteAppOptions(limits=limits, auto_pipeline=True),
    )

    with route_app.client() as client:
        response = client.get("/functions/high-risk")

    expect_equal(response.status_code, status.HTTP_200_OK)
