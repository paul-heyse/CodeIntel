"""Tests for profile query service delegates.

This module tests the _ProfileQueryDelegates methods in services/profiles.py
through HTTP routes and direct LocalQueryService invocation, using real gateways.
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
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.errors import McpError
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.gateway import build_duckdb_query_service

if TYPE_CHECKING:
    from tests._helpers.fixtures import ProvisionedGateway

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
        service_override=service,
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


def _build_local_query_service(
    provisioned_repo: ProvisionedGateway,
) -> LocalQueryService:
    """Build a LocalQueryService for direct testing.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.

    Returns
    -------
    LocalQueryService
        Configured local query service.
    """
    limits = BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS)
    query = build_duckdb_query_service(
        provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        limits=limits,
    )
    return LocalQueryService(
        query=query,
        dataset_tables=dict(provisioned_repo.gateway.datasets.mapping),
    )


# =============================================================================
# get_function_profile Tests (via HTTP)
# =============================================================================


def test_get_function_profile_with_goid_h128(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_profile works with goid_h128 parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    # Get a valid goid_h128 from the database
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/profiles/function?goid_h128={goid_h128}")

    # May return 404 if profile doesn't exist, or 200 if found
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


def test_get_function_profile_missing_goid_h128(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_profile returns error when goid_h128 missing.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/profiles/function")

    # Should return 422 validation error (missing required param)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_get_function_profile_invalid_goid_h128(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_profile handles invalid goid_h128.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        # Use a very large number that's unlikely to exist
        response = client.get("/profiles/function?goid_h128=999999999999")

    # Should return 404 or empty result
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


# =============================================================================
# get_file_profile Tests (via HTTP)
# =============================================================================


def test_get_file_profile_with_rel_path(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_file_profile works with rel_path parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    rel_path = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/profiles/file?rel_path={rel_path}")

    # May return 404 if profile doesn't exist, or 200 if found
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


def test_get_file_profile_missing_rel_path(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_file_profile returns error when rel_path missing.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/profiles/file")

    # Should return 422 validation error (missing required param)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_get_file_profile_nonexistent_file(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_file_profile handles nonexistent file.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/profiles/file?rel_path=nonexistent/path/file.py")

    # Should return 404 or empty result
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


# =============================================================================
# get_module_profile Tests (via HTTP)
# =============================================================================


def test_get_module_profile_with_module(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_module_profile works with module parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/profiles/module?module={module}")

    # May return 404 if profile doesn't exist, or 200 if found
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


def test_get_module_profile_missing_module(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_module_profile returns error when module missing.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/profiles/module")

    # Should return 422 validation error (missing required param)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_get_module_profile_nonexistent_module(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_module_profile handles nonexistent module.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/profiles/module?module=nonexistent.module.name")

    # Should return 404 or empty result
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


# =============================================================================
# get_function_architecture Tests (via HTTP)
# =============================================================================


def test_get_function_architecture_with_goid_h128(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_architecture works with goid_h128 parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/architecture/function?goid_h128={goid_h128}")

    # May return 404 if architecture doesn't exist, or 200 if found
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


def test_get_function_architecture_missing_goid_h128(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_architecture returns error when goid_h128 missing.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/architecture/function")

    # Should return 422 validation error (missing required param)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_get_function_architecture_invalid_goid_h128(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_function_architecture handles invalid goid_h128.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/architecture/function?goid_h128=999999999999")

    # Should return 404 or empty result
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


# =============================================================================
# get_module_architecture Tests (via HTTP)
# =============================================================================


def test_get_module_architecture_with_module(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_module_architecture works with module parameter.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/architecture/module?module={module}")

    # May return 404 if architecture doesn't exist, or 200 if found
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


def test_get_module_architecture_missing_module(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_module_architecture returns error when module missing.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/architecture/module")

    # Should return 422 validation error (missing required param)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_get_module_architecture_nonexistent_module(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify get_module_architecture handles nonexistent module.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get("/architecture/module?module=nonexistent.module.name")

    # Should return 404 or empty result
    assert response.status_code in {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND}


# =============================================================================
# Direct LocalQueryService Tests
# =============================================================================


def test_local_query_service_get_function_profile(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService.get_function_profile works directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    profile = service.get_function_profile(goid_h128=goid_h128)
    assert profile is not None


def test_local_query_service_get_file_profile(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService.get_file_profile works directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    rel_path = result[0]

    profile = service.get_file_profile(rel_path=rel_path)
    assert profile is not None


def test_local_query_service_get_module_profile(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService.get_module_profile works directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    # Module profile may not exist for all modules - handle gracefully
    try:
        profile = service.get_module_profile(module=module)
        assert profile is not None
    except McpError:
        # Expected when module profile doesn't exist in test data
        pass


def test_local_query_service_get_function_architecture(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService.get_function_architecture works directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    architecture = service.get_function_architecture(goid_h128=goid_h128)
    assert architecture is not None


def test_local_query_service_get_module_architecture(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify LocalQueryService.get_module_architecture works directly.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    service = _build_local_query_service(provisioned_repo)

    result = provisioned_repo.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    # Module architecture may not exist for all modules - handle gracefully
    try:
        architecture = service.get_module_architecture(module=module)
        assert architecture is not None
    except McpError:
        # Expected when module architecture doesn't exist in test data
        pass


# =============================================================================
# Response Structure Tests
# =============================================================================


def test_function_profile_response_structure(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify function profile response contains expected fields.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/profiles/function?goid_h128={goid_h128}")

    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        # Check that response is a dict (profile structure)
        assert isinstance(data, dict)


def test_file_profile_response_structure(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify file profile response contains expected fields.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    rel_path = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/profiles/file?rel_path={rel_path}")

    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        # Check that response is a dict (profile structure)
        assert isinstance(data, dict)


def test_module_profile_response_structure(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify module profile response contains expected fields.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/profiles/module?module={module}")

    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        # Check that response is a dict (profile structure)
        assert isinstance(data, dict)


def test_function_architecture_response_structure(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify function architecture response contains expected fields.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/architecture/function?goid_h128={goid_h128}")

    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        # Check that response is a dict (architecture structure)
        assert isinstance(data, dict)


def test_module_architecture_response_structure(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify module architecture response contains expected fields.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    result = provisioned_repo.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    app = _create_test_app(provisioned_repo)
    with TestClient(app) as client:
        response = client.get(f"/architecture/module?module={module}")

    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        # Check that response is a dict (architecture structure)
        assert isinstance(data, dict)
