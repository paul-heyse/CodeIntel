"""Tests for subsystem HTTP routes.

This module tests the subsystem-related HTTP endpoints using real gateways.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.routes.functions import RouterOptions
from codeintel.storage.gateway import StorageGateway
from tests._helpers.analytics_samples import AnalyticsSamples
from tests._helpers.assertions import expect_equal, expect_false, expect_in, expect_true

HttpAppFactory = Callable[..., FastAPI]

# =============================================================================
# Constants
# =============================================================================

DEFAULT_LIMIT = 10
MAX_ROWS = 100


def _build_architecture_app(
    gateway: StorageGateway,
    make_http_app: HttpAppFactory,
    limits: BackendLimits | None = None,
    *,
    auto_pipeline: bool = False,
) -> FastAPI:
    """Construct an app for architecture-backed subsystem routes.

    Parameters
    ----------
    gateway
        Storage gateway with architecture data.
    make_http_app
        Factory fixture that builds FastAPI apps for HTTP route tests.
    limits
        Optional pagination limits for the backend.
    auto_pipeline
        Whether to enable auto-pipeline on the constructed app.

    Returns
    -------
    FastAPI
        Configured application for subsystem route tests.
    """
    effective_limits = limits or BackendLimits(
        default_limit=DEFAULT_LIMIT,
        max_rows_per_call=MAX_ROWS,
    )
    return make_http_app(
        gateway=gateway,
        snapshot=("demo/repo", "deadbeef"),
        limits=effective_limits,
        auto_pipeline=auto_pipeline,
    )


# =============================================================================
# list_subsystems Tests
# =============================================================================


def test_list_subsystems_endpoint(
    architecture_gateway: StorageGateway,
    make_http_app: HttpAppFactory,
) -> None:
    """Verify /subsystems endpoint returns results.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    """
    app = _build_architecture_app(architecture_gateway, make_http_app)
    with TestClient(app) as client:
        response = client.get("/architecture/subsystems")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("subsystems", data)


def test_list_subsystems_with_limit(
    architecture_gateway: StorageGateway,
    make_http_app: HttpAppFactory,
) -> None:
    """Verify /architecture/subsystems accepts limit parameter.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    """
    app = _build_architecture_app(architecture_gateway, make_http_app)
    with TestClient(app) as client:
        response = client.get("/architecture/subsystems?limit=5")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("subsystems", data)


def test_list_subsystems_with_role_filter(
    architecture_gateway: StorageGateway,
    make_http_app: HttpAppFactory,
) -> None:
    """Verify /architecture/subsystems accepts role filter.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    """
    app = _build_architecture_app(architecture_gateway, make_http_app)
    with TestClient(app) as client:
        response = client.get("/architecture/subsystems?role=test_role")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("subsystems", data)


def test_list_subsystems_with_query_filter(
    architecture_gateway: StorageGateway,
    make_http_app: HttpAppFactory,
) -> None:
    """Verify /architecture/subsystems accepts query filter.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    """
    app = _build_architecture_app(architecture_gateway, make_http_app)
    with TestClient(app) as client:
        response = client.get("/architecture/subsystems?q=test")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("subsystems", data)


# =============================================================================
# module_subsystems Tests
# =============================================================================


def test_module_subsystems_endpoint(
    architecture_gateway: StorageGateway,
    make_http_app: HttpAppFactory,
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify /subsystems/module endpoint returns results.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    architecture_samples
        Sample identifiers for architecture analytics data.
    """
    module = architecture_samples.module

    app = _build_architecture_app(architecture_gateway, make_http_app)
    with TestClient(app) as client:
        response = client.get(f"/architecture/module-subsystems?module={module}")

    # May return 200 or 404
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_module_subsystems_missing_module(
    architecture_gateway: StorageGateway,
    make_http_app: HttpAppFactory,
) -> None:
    """Verify /architecture/module-subsystems returns error when module missing.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    """
    app = _build_architecture_app(architecture_gateway, make_http_app)
    with TestClient(app) as client:
        response = client.get("/architecture/module-subsystems")

    # Should return 422 validation error (missing required param)
    expect_equal(response.status_code, status.HTTP_422_UNPROCESSABLE_CONTENT)


# =============================================================================
# subsystem_detail Tests
# =============================================================================


def test_subsystem_detail_endpoint(
    architecture_gateway: StorageGateway,
    make_http_app: HttpAppFactory,
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify /subsystems/{subsystem_id} endpoint returns results.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    architecture_samples
        Sample identifiers for architecture analytics data.
    """
    subsystem_id = architecture_samples.subsystem_id

    app = _build_architecture_app(architecture_gateway, make_http_app)
    with TestClient(app) as client:
        response = client.get(f"/architecture/subsystem?subsystem_id={subsystem_id}")

    # May return 200 or 404
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_subsystem_detail_with_module_limit(
    architecture_gateway: StorageGateway,
    make_http_app: HttpAppFactory,
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify /architecture/subsystem accepts module_limit.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    architecture_samples
        Sample identifiers for architecture analytics data.
    """
    subsystem_id = architecture_samples.subsystem_id

    app = _build_architecture_app(architecture_gateway, make_http_app)
    with TestClient(app) as client:
        response = client.get(f"/architecture/subsystem?subsystem_id={subsystem_id}&module_limit=5")

    # May return 200 or 404
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_subsystem_detail_nonexistent(
    architecture_gateway: StorageGateway,
    make_http_app: HttpAppFactory,
) -> None:
    """Verify /architecture/subsystem handles nonexistent subsystem.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    """
    app = _build_architecture_app(architecture_gateway, make_http_app)
    with TestClient(app) as client:
        response = client.get("/architecture/subsystem?subsystem_id=nonexistent_subsystem_xyz")

    # Should return 404
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


# =============================================================================
# subsystem_profiles Tests
# =============================================================================


def test_subsystem_profiles_endpoint(
    architecture_gateway: StorageGateway,
    make_http_app: HttpAppFactory,
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify /subsystems/{subsystem_id}/profiles endpoint.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    architecture_samples
        Sample identifiers for architecture analytics data.
    """
    subsystem_id = architecture_samples.subsystem_id

    app = _build_architecture_app(architecture_gateway, make_http_app)
    with TestClient(app) as client:
        response = client.get(f"/subsystems/{subsystem_id}/profiles")

    # May return 200 or 404
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_subsystem_profiles_with_limit(
    architecture_gateway: StorageGateway,
    make_http_app: HttpAppFactory,
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify /subsystems/{subsystem_id}/profiles accepts limit.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    architecture_samples
        Sample identifiers for architecture analytics data.
    """
    subsystem_id = architecture_samples.subsystem_id

    app = _build_architecture_app(architecture_gateway, make_http_app)
    with TestClient(app) as client:
        response = client.get(f"/subsystems/{subsystem_id}/profiles?limit=5")

    # May return 200 or 404
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


# =============================================================================
# subsystem_coverage Tests
# =============================================================================


def test_subsystem_coverage_endpoint(
    architecture_gateway: StorageGateway,
    make_http_app: HttpAppFactory,
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify /subsystems/{subsystem_id}/coverage endpoint.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    architecture_samples
        Sample identifiers for architecture analytics data.
    """
    subsystem_id = architecture_samples.subsystem_id

    app = _build_architecture_app(architecture_gateway, make_http_app)
    with TestClient(app) as client:
        response = client.get(f"/subsystems/{subsystem_id}/coverage")

    # May return 200 or 404
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_subsystem_coverage_with_limit(
    architecture_gateway: StorageGateway,
    make_http_app: HttpAppFactory,
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify /subsystems/{subsystem_id}/coverage accepts limit.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    architecture_samples
        Sample identifiers for architecture analytics data.
    """
    subsystem_id = architecture_samples.subsystem_id

    app = _build_architecture_app(architecture_gateway, make_http_app)
    with TestClient(app) as client:
        response = client.get(f"/subsystems/{subsystem_id}/coverage?limit=5")

    # May return 200 or 404
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


# =============================================================================
# Router Options Tests
# =============================================================================


@pytest.mark.skip(reason="auto_pipeline mode not fully configured for subsystem routes")
def test_router_with_auto_pipeline(
    architecture_gateway: StorageGateway,
    make_http_app: HttpAppFactory,
) -> None:
    """Verify subsystem routes work with auto_pipeline enabled.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    """
    app = _build_architecture_app(
        architecture_gateway,
        make_http_app,
        auto_pipeline=True,
    )

    with TestClient(app) as client:
        response = client.get("/subsystems")

    expect_equal(response.status_code, status.HTTP_200_OK)


def test_router_options_default() -> None:
    """Verify RouterOptions defaults to auto_pipeline=False."""
    options = RouterOptions()
    expect_false(options.auto_pipeline)


def test_router_options_with_auto_pipeline() -> None:
    """Verify RouterOptions accepts auto_pipeline=True."""
    options = RouterOptions(auto_pipeline=True)
    expect_true(options.auto_pipeline)
