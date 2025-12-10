"""Tests for subsystem HTTP routes.

This module tests the subsystem-related HTTP endpoints using real gateways.
"""

from __future__ import annotations

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from codeintel.serving.http.routes.functions import RouterOptions
from tests._helpers.assertions.http_responses import (
    assert_ok_or_not_found,
    assert_problem_detail_response,
)
from tests._helpers.analytics_samples import AnalyticsSamples
from tests._helpers.assertions import expect_equal, expect_false, expect_in, expect_true

@pytest.mark.parametrize(
    "query",
    [
        "",
        "limit=5",
        "role=test_role",
        "q=test",
    ],
)
def test_list_subsystems_endpoint(
    architecture_route_client: TestClient,
    query: str,
) -> None:
    """/architecture/subsystems returns results with optional filters."""
    path = "/architecture/subsystems"
    if query:
        path = f"{path}?{query}"

    response = architecture_route_client.get(path)

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("subsystems", data)


# =============================================================================
# module_subsystems Tests
# =============================================================================


def test_module_subsystems_endpoint(
    architecture_route_client: TestClient,
    architecture_samples: AnalyticsSamples,
) -> None:
    """/architecture/module-subsystems returns results."""
    response = architecture_route_client.get(
        f"/architecture/module-subsystems?module={architecture_samples.module}"
    )

    assert_ok_or_not_found(response)


def test_module_subsystems_missing_module(
    architecture_route_client: TestClient,
) -> None:
    """/architecture/module-subsystems returns error when module missing."""
    response = architecture_route_client.get("/architecture/module-subsystems")

    assert_problem_detail_response(
        response,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )


# =============================================================================
# subsystem_detail Tests
# =============================================================================


def test_subsystem_detail_endpoint(
    architecture_route_client: TestClient,
    architecture_samples: AnalyticsSamples,
) -> None:
    """/architecture/subsystem endpoint returns results."""
    response = architecture_route_client.get(
        f"/architecture/subsystem?subsystem_id={architecture_samples.subsystem_id}"
    )

    assert_ok_or_not_found(response)


def test_subsystem_detail_with_module_limit(
    architecture_route_client: TestClient,
    architecture_samples: AnalyticsSamples,
) -> None:
    """/architecture/subsystem accepts module_limit."""
    response = architecture_route_client.get(
        f"/architecture/subsystem?subsystem_id={architecture_samples.subsystem_id}&module_limit=5"
    )

    assert_ok_or_not_found(response)


def test_subsystem_detail_nonexistent(
    architecture_route_client: TestClient,
) -> None:
    """/architecture/subsystem handles nonexistent subsystem."""
    response = architecture_route_client.get(
        "/architecture/subsystem?subsystem_id=nonexistent_subsystem_xyz"
    )

    assert_ok_or_not_found(response)


# =============================================================================
# subsystem_profiles Tests
# =============================================================================


def test_subsystem_profiles_endpoint(
    architecture_route_client: TestClient,
    architecture_samples: AnalyticsSamples,
) -> None:
    """/subsystems/{subsystem_id}/profiles returns results."""
    response = architecture_route_client.get(
        f"/subsystems/{architecture_samples.subsystem_id}/profiles"
    )

    assert_ok_or_not_found(response)


@pytest.mark.parametrize("query", ["", "limit=5"])
def test_subsystem_profiles_with_options(
    architecture_route_client: TestClient,
    architecture_samples: AnalyticsSamples,
    query: str,
) -> None:
    """/subsystems/{subsystem_id}/profiles supports optional limit."""
    path = f"/subsystems/{architecture_samples.subsystem_id}/profiles"
    if query:
        path = f"{path}?{query}"

    response = architecture_route_client.get(path)

    assert_ok_or_not_found(response)


# =============================================================================
# subsystem_coverage Tests
# =============================================================================


def test_subsystem_coverage_endpoint(
    architecture_route_client: TestClient,
    architecture_samples: AnalyticsSamples,
) -> None:
    """/subsystems/{subsystem_id}/coverage returns results."""
    response = architecture_route_client.get(
        f"/subsystems/{architecture_samples.subsystem_id}/coverage"
    )

    assert_ok_or_not_found(response)


@pytest.mark.parametrize("query", ["", "limit=5"])
def test_subsystem_coverage_with_options(
    architecture_route_client: TestClient,
    architecture_samples: AnalyticsSamples,
    query: str,
) -> None:
    """/subsystems/{subsystem_id}/coverage supports optional limit."""
    path = f"/subsystems/{architecture_samples.subsystem_id}/coverage"
    if query:
        path = f"{path}?{query}"

    response = architecture_route_client.get(path)

    assert_ok_or_not_found(response)


# =============================================================================
# Router Options Tests
# =============================================================================


@pytest.mark.skip(reason="auto_pipeline mode not fully configured for subsystem routes")
def test_router_with_auto_pipeline(
    architecture_route_client: TestClient,
) -> None:
    """Subsystem routes work with auto_pipeline enabled."""
    response = architecture_route_client.get("/subsystems")

    expect_equal(response.status_code, status.HTTP_200_OK)


def test_router_options_default() -> None:
    """RouterOptions defaults to auto_pipeline=False."""
    options = RouterOptions()
    expect_false(options.auto_pipeline)


def test_router_options_with_auto_pipeline() -> None:
    """RouterOptions accepts auto_pipeline=True."""
    options = RouterOptions(auto_pipeline=True)
    expect_true(options.auto_pipeline)
