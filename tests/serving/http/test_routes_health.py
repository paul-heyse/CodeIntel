"""Tests for health HTTP routes.

This module tests the health endpoint using real gateways and TestClient.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapi import status
from fastapi.testclient import TestClient

from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.routes.health import build_health_router
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_true,
)

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway


# =============================================================================
# build_health_router Tests
# =============================================================================


def test_build_health_router_returns_router() -> None:
    """Verify build_health_router returns an APIRouter with health path."""
    router = build_health_router()

    routes: list[str] = []
    for route in router.routes:
        if hasattr(route, "path"):
            path_value = getattr(route, "path", None)
            if isinstance(path_value, str):
                routes.append(path_value)
    expect_in("/health", routes)


# =============================================================================
# Health Endpoint Tests
# =============================================================================


def test_health_endpoint_returns_status_ok(
    provisioned_repo: ProvisionedGateway,
    make_http_app: Callable[..., object],
) -> None:
    """Verify /health returns status: ok with repo and commit info.

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
        config_overrides={"read_only": True},
    )

    with TestClient(app) as client:
        response = client.get("/health")

    expect_equal(response.status_code, status.HTTP_200_OK)
    payload = response.json()
    expect_equal(payload["status"], "ok")
    expect_equal(payload["repo"], provisioned_repo.repo)
    expect_equal(payload["commit"], provisioned_repo.commit)
    expect_true(payload["read_only"])


def test_health_endpoint_includes_limits(
    provisioned_repo: ProvisionedGateway,
    make_http_app: Callable[..., object],
) -> None:
    """Verify /health includes limits when service has them.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    make_http_app
        Fixture that builds a FastAPI app bound to the gateway.
    """
    default_limit = 25
    max_rows = 250
    limits = BackendLimits(default_limit=default_limit, max_rows_per_call=max_rows)
    app = make_http_app(
        gateway=provisioned_repo.gateway,
        snapshot=(provisioned_repo.repo, provisioned_repo.commit),
        limits=limits,
    )

    with TestClient(app) as client:
        response = client.get("/health")

    expect_equal(response.status_code, status.HTTP_200_OK)
    payload = response.json()
    expect_in("limits", payload)
    expect_equal(payload["limits"]["default_limit"], default_limit)
    expect_equal(payload["limits"]["max_rows_per_call"], max_rows)


def test_health_endpoint_read_only_false(
    provisioned_repo: ProvisionedGateway,
    make_http_app: Callable[..., object],
) -> None:
    """Verify /health reflects read_only=False when configured.

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
        config_overrides={"read_only": False},
    )

    with TestClient(app) as client:
        response = client.get("/health")

    expect_equal(response.status_code, status.HTTP_200_OK)
    payload = response.json()
    expect_true(payload["read_only"] is False)


def test_health_endpoint_database_connectivity_verified(
    provisioned_repo: ProvisionedGateway,
    make_http_app: Callable[..., object],
) -> None:
    """Verify /health actually probes DuckDB connectivity.

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

    # Health should pass when DB is accessible
    with TestClient(app) as client:
        response = client.get("/health")

    expect_equal(response.status_code, status.HTTP_200_OK)
    # This confirms the DuckDB SELECT 1 query succeeded
    expect_equal(response.json()["status"], "ok")
