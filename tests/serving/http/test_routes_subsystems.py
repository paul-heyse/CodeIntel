"""Tests for subsystem HTTP routes.

This module tests the subsystem-related HTTP endpoints using real gateways.
"""

from __future__ import annotations

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from codeintel.serving.http.routes.functions import RouterOptions
from tests._helpers.analytics_samples import AnalyticsSamples
from tests._helpers.assertions import expect_equal, expect_false, expect_in, expect_true

# =============================================================================
# list_subsystems Tests
# =============================================================================


def test_list_subsystems_endpoint(
    architecture_route_client: TestClient,
) -> None:
    """/architecture/subsystems returns results."""
    response = architecture_route_client.get("/architecture/subsystems")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("subsystems", data)


def test_list_subsystems_with_limit(
    architecture_route_client: TestClient,
) -> None:
    """/architecture/subsystems accepts limit parameter."""
    response = architecture_route_client.get("/architecture/subsystems?limit=5")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("subsystems", data)


def test_list_subsystems_with_role_filter(
    architecture_route_client: TestClient,
) -> None:
    """/architecture/subsystems accepts role filter."""
    response = architecture_route_client.get("/architecture/subsystems?role=test_role")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("subsystems", data)


def test_list_subsystems_with_query_filter(
    architecture_route_client: TestClient,
) -> None:
    """/architecture/subsystems accepts query filter."""
    response = architecture_route_client.get("/architecture/subsystems?q=test")

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

    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_module_subsystems_missing_module(
    architecture_route_client: TestClient,
) -> None:
    """/architecture/module-subsystems returns error when module missing."""
    response = architecture_route_client.get("/architecture/module-subsystems")

    expect_equal(response.status_code, status.HTTP_422_UNPROCESSABLE_CONTENT)


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

    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_subsystem_detail_with_module_limit(
    architecture_route_client: TestClient,
    architecture_samples: AnalyticsSamples,
) -> None:
    """/architecture/subsystem accepts module_limit."""
    response = architecture_route_client.get(
        f"/architecture/subsystem?subsystem_id={architecture_samples.subsystem_id}&module_limit=5"
    )

    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_subsystem_detail_nonexistent(
    architecture_route_client: TestClient,
) -> None:
    """/architecture/subsystem handles nonexistent subsystem."""
    response = architecture_route_client.get(
        "/architecture/subsystem?subsystem_id=nonexistent_subsystem_xyz"
    )

    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


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

    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_subsystem_profiles_with_limit(
    architecture_route_client: TestClient,
    architecture_samples: AnalyticsSamples,
) -> None:
    """/subsystems/{subsystem_id}/profiles accepts limit."""
    response = architecture_route_client.get(
        f"/subsystems/{architecture_samples.subsystem_id}/profiles?limit=5"
    )

    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


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

    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_subsystem_coverage_with_limit(
    architecture_route_client: TestClient,
    architecture_samples: AnalyticsSamples,
) -> None:
    """/subsystems/{subsystem_id}/coverage accepts limit."""
    response = architecture_route_client.get(
        f"/subsystems/{architecture_samples.subsystem_id}/coverage?limit=5"
    )

    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


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
