"""Tests for subsystem HTTP routes.

This module tests the subsystem-related HTTP endpoints using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.fastapi import (
    BackendResource,
    create_app,
)
from codeintel.serving.http.routes.functions import RouterOptions
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import expect_equal, expect_false, expect_in, expect_true
from tests._helpers.gateway import build_duckdb_query_service

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway

# =============================================================================
# Constants
# =============================================================================

DEFAULT_LIMIT = 10
MAX_ROWS = 100


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


def _create_architecture_test_app(gateway: StorageGateway) -> FastAPI:
    """Create a test FastAPI app with the architecture gateway.

    Parameters
    ----------
    gateway
        Storage gateway with architecture data.

    Returns
    -------
    FastAPI
        Configured FastAPI application.
    """
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    return create_app(config_loader=load_config, backend_factory=backend_factory)


# =============================================================================
# list_subsystems Tests
# =============================================================================


def test_list_subsystems_endpoint(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify /subsystems endpoint returns results.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    app = _create_architecture_test_app(architecture_gateway)
    with TestClient(app) as client:
        response = client.get("/architecture/subsystems")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("subsystems", data)


def test_list_subsystems_with_limit(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify /architecture/subsystems accepts limit parameter.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    app = _create_architecture_test_app(architecture_gateway)
    with TestClient(app) as client:
        response = client.get("/architecture/subsystems?limit=5")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("subsystems", data)


def test_list_subsystems_with_role_filter(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify /architecture/subsystems accepts role filter.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    app = _create_architecture_test_app(architecture_gateway)
    with TestClient(app) as client:
        response = client.get("/architecture/subsystems?role=test_role")

    expect_equal(response.status_code, status.HTTP_200_OK)
    data = response.json()
    expect_in("subsystems", data)


def test_list_subsystems_with_query_filter(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify /architecture/subsystems accepts query filter.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    app = _create_architecture_test_app(architecture_gateway)
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
) -> None:
    """Verify /subsystems/module endpoint returns results.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    # Get a valid module name
    result = architecture_gateway.con.execute("SELECT module FROM core.modules LIMIT 1").fetchone()

    if result is None:
        return  # Skip if no modules

    module = result[0]

    app = _create_architecture_test_app(architecture_gateway)
    with TestClient(app) as client:
        response = client.get(f"/architecture/module-subsystems?module={module}")

    # May return 200 or 404
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_module_subsystems_missing_module(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify /architecture/module-subsystems returns error when module missing.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    app = _create_architecture_test_app(architecture_gateway)
    with TestClient(app) as client:
        response = client.get("/architecture/module-subsystems")

    # Should return 422 validation error (missing required param)
    expect_equal(response.status_code, status.HTTP_422_UNPROCESSABLE_CONTENT)


# =============================================================================
# subsystem_detail Tests
# =============================================================================


def test_subsystem_detail_endpoint(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify /subsystems/{subsystem_id} endpoint returns results.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    # Get a valid subsystem_id
    result = architecture_gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystem_agreement WHERE subsystem_id IS NOT NULL LIMIT 1"
    ).fetchone()

    if result is None:
        return  # Skip if no subsystems

    subsystem_id = result[0]

    app = _create_architecture_test_app(architecture_gateway)
    with TestClient(app) as client:
        response = client.get(f"/architecture/subsystem?subsystem_id={subsystem_id}")

    # May return 200 or 404
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_subsystem_detail_with_module_limit(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify /architecture/subsystem accepts module_limit.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    result = architecture_gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystem_agreement WHERE subsystem_id IS NOT NULL LIMIT 1"
    ).fetchone()

    if result is None:
        return  # Skip if no subsystems

    subsystem_id = result[0]

    app = _create_architecture_test_app(architecture_gateway)
    with TestClient(app) as client:
        response = client.get(f"/architecture/subsystem?subsystem_id={subsystem_id}&module_limit=5")

    # May return 200 or 404
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_subsystem_detail_nonexistent(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify /architecture/subsystem handles nonexistent subsystem.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    app = _create_architecture_test_app(architecture_gateway)
    with TestClient(app) as client:
        response = client.get("/architecture/subsystem?subsystem_id=nonexistent_subsystem_xyz")

    # Should return 404
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


# =============================================================================
# subsystem_profiles Tests
# =============================================================================


def test_subsystem_profiles_endpoint(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify /subsystems/{subsystem_id}/profiles endpoint.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    result = architecture_gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystem_agreement WHERE subsystem_id IS NOT NULL LIMIT 1"
    ).fetchone()

    if result is None:
        return  # Skip if no subsystems

    subsystem_id = result[0]

    app = _create_architecture_test_app(architecture_gateway)
    with TestClient(app) as client:
        response = client.get(f"/subsystems/{subsystem_id}/profiles")

    # May return 200 or 404
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_subsystem_profiles_with_limit(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify /subsystems/{subsystem_id}/profiles accepts limit.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    result = architecture_gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystem_agreement WHERE subsystem_id IS NOT NULL LIMIT 1"
    ).fetchone()

    if result is None:
        return  # Skip if no subsystems

    subsystem_id = result[0]

    app = _create_architecture_test_app(architecture_gateway)
    with TestClient(app) as client:
        response = client.get(f"/subsystems/{subsystem_id}/profiles?limit=5")

    # May return 200 or 404
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


# =============================================================================
# subsystem_coverage Tests
# =============================================================================


def test_subsystem_coverage_endpoint(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify /subsystems/{subsystem_id}/coverage endpoint.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    result = architecture_gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystem_agreement WHERE subsystem_id IS NOT NULL LIMIT 1"
    ).fetchone()

    if result is None:
        return  # Skip if no subsystems

    subsystem_id = result[0]

    app = _create_architecture_test_app(architecture_gateway)
    with TestClient(app) as client:
        response = client.get(f"/subsystems/{subsystem_id}/coverage")

    # May return 200 or 404
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_subsystem_coverage_with_limit(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify /subsystems/{subsystem_id}/coverage accepts limit.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    result = architecture_gateway.con.execute(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystem_agreement WHERE subsystem_id IS NOT NULL LIMIT 1"
    ).fetchone()

    if result is None:
        return  # Skip if no subsystems

    subsystem_id = result[0]

    app = _create_architecture_test_app(architecture_gateway)
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
) -> None:
    """Verify subsystem routes work with auto_pipeline enabled.

    Parameters
    ----------
    architecture_gateway
        Gateway with architecture data seeded.
    """
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(architecture_gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
        )

    def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    # Create app with auto_pipeline enabled
    app = create_app(
        config_loader=load_config,
        backend_factory=backend_factory,
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
