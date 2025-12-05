"""Tests for FastAPI app factory and middleware.

This module tests the FastAPI application factory, exception handlers,
and middleware using real gateways and TestClient - no mocking.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.bootstrap import BackendResource, build_backend_resource
from codeintel.serving.http.fastapi import (
    create_app,
    load_api_config,
    problem_response,
)
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.errors import ProblemDetail
from codeintel.serving.services.query_service import LocalQueryService
from tests._helpers.gateway import build_duckdb_query_service

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway


# =============================================================================
# load_api_config Tests (covers _ensure_readable_db via public interface)
# =============================================================================


def test_load_api_config_local_db_path_not_found(tmp_path: Path, codeintel_env: None) -> None:
    """Verify FileNotFoundError when local_db path does not exist.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    codeintel_env
        Fixture that saves/restores CODEINTEL_* env vars.
    """
    _ = codeintel_env  # Used for cleanup side effect
    nonexistent = tmp_path / "missing.duckdb"
    os.environ["CODEINTEL_REPO"] = "test/repo"
    os.environ["CODEINTEL_COMMIT"] = "abc123"
    os.environ["CODEINTEL_MCP_MODE"] = "local_db"
    os.environ["CODEINTEL_DB_PATH"] = str(nonexistent)

    with pytest.raises(FileNotFoundError, match="DuckDB database not found"):
        load_api_config()


def test_load_api_config_local_db_path_is_directory(tmp_path: Path, codeintel_env: None) -> None:
    """Verify ValueError when local_db path is a directory, not a file.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    codeintel_env
        Fixture that saves/restores CODEINTEL_* env vars.
    """
    _ = codeintel_env  # Used for cleanup side effect
    directory = tmp_path / "some_dir"
    directory.mkdir()
    os.environ["CODEINTEL_REPO"] = "test/repo"
    os.environ["CODEINTEL_COMMIT"] = "abc123"
    os.environ["CODEINTEL_MCP_MODE"] = "local_db"
    os.environ["CODEINTEL_DB_PATH"] = str(directory)

    with pytest.raises(ValueError, match="is not a file"):
        load_api_config()


def test_load_api_config_remote_api_success(codeintel_env: None) -> None:
    """Verify config loads successfully for remote_api mode.

    Parameters
    ----------
    codeintel_env
        Fixture that saves/restores CODEINTEL_* env vars.
    """
    _ = codeintel_env  # Used for cleanup side effect
    os.environ["CODEINTEL_REPO"] = "test/repo"
    os.environ["CODEINTEL_COMMIT"] = "abc123"
    os.environ["CODEINTEL_MCP_MODE"] = "remote_api"
    os.environ["CODEINTEL_API_BASE_URL"] = "http://example.com"

    config = load_api_config()

    assert config.repo == "test/repo"
    assert config.commit == "abc123"
    assert config.mode == "remote_api"


def test_load_api_config_local_db_success(tmp_path: Path, codeintel_env: None) -> None:
    """Verify config loads successfully for local_db mode with valid file.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    codeintel_env
        Fixture that saves/restores CODEINTEL_* env vars.
    """
    _ = codeintel_env  # Used for cleanup side effect
    db_file = tmp_path / "valid.duckdb"
    db_file.write_bytes(b"valid content")
    os.environ["CODEINTEL_REPO"] = "test/repo"
    os.environ["CODEINTEL_COMMIT"] = "abc123"
    os.environ["CODEINTEL_MCP_MODE"] = "local_db"
    os.environ["CODEINTEL_DB_PATH"] = str(db_file)

    config = load_api_config()

    assert config.repo == "test/repo"
    assert config.mode == "local_db"
    assert config.db_path == db_file


# =============================================================================
# problem_response Tests
# =============================================================================


def test_problem_response_returns_json_response() -> None:
    """Verify problem_response returns JSONResponse with correct payload."""
    detail = ProblemDetail(
        type="https://codeintel/problems/test",
        title="Test Error",
        detail="Something went wrong",
        status=status.HTTP_400_BAD_REQUEST,
        code="test-error",
    )

    response = problem_response(detail)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    body_bytes = response.body
    body = body_bytes.decode("utf-8") if isinstance(body_bytes, bytes) else str(body_bytes)
    assert "test-error" in body
    assert "Test Error" in body


def test_problem_response_defaults_status_500() -> None:
    """Verify problem_response defaults to 500 when status is None."""
    detail = ProblemDetail(
        type="https://codeintel/problems/test",
        title="Server Error",
        detail="Unexpected",
        status=None,
        code="server-error",
    )

    response = problem_response(detail)

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# =============================================================================
# build_backend_resource Tests
# =============================================================================


def test_build_backend_resource_raises_error_on_failure(
    tmp_path: Path,
) -> None:
    """Verify build_backend_resource raises ValueError when gateway is missing.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    """
    cfg = ServingConfig(
        repo="test/repo",
        commit="abc123",
        mode="local_db",
        db_path=tmp_path / "nonexistent.duckdb",
        repo_root=tmp_path,
    )

    with pytest.raises(ValueError, match="StorageGateway is required"):
        build_backend_resource(cfg)


def test_build_backend_resource_with_gateway(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify build_backend_resource returns BackendResource with gateway.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    cfg = ServingConfig(
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
        mode="local_db",
        db_path=provisioned_repo.db_path,
        repo_root=provisioned_repo.repo_root,
    )

    resource = build_backend_resource(cfg, gateway=provisioned_repo.gateway)

    assert resource.backend is not None
    assert callable(resource.close)


# =============================================================================
# create_app Integration Tests
# =============================================================================


def test_create_app_with_provisioned_gateway(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify create_app creates functional FastAPI app with real gateway.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["repo"] == provisioned_repo.repo


def test_create_app_correlation_id_from_header(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify correlation ID from X-Request-ID header is propagated.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/health", headers={"X-Request-ID": "test-correlation-id"})

    assert response.headers.get("X-Request-ID") == "test-correlation-id"


def test_create_app_correlation_id_generated_when_missing(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify correlation ID is generated when not provided in headers.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/health")

    correlation_id = response.headers.get("X-Request-ID")
    assert correlation_id is not None
    assert len(correlation_id) > 0


def test_create_app_x_correlation_id_header(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify X-Correlation-ID header is also accepted.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    with TestClient(app) as client:
        response = client.get("/health", headers={"X-Correlation-ID": "alt-correlation-id"})

    # Should use X-Correlation-ID when X-Request-ID is not present
    assert response.headers.get("X-Request-ID") == "alt-correlation-id"


# =============================================================================
# Exception Handler Tests
# =============================================================================


def test_exception_handler_problem_error(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify ProblemError is converted to proper JSON response.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    # Request a non-existent dataset to trigger ProblemError
    with TestClient(app) as client:
        response = client.get("/datasets/nonexistent_dataset")

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    payload = response.json()
    assert "code" in payload
    assert payload["code"] == "dataset-not-found"


def test_exception_handler_validation_error(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify validation errors return proper Problem Details.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    # Request with invalid limit value to trigger validation error
    with TestClient(app) as client:
        response = client.get("/functions/high-risk?limit=invalid")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    payload = response.json()
    assert payload["code"] == "invalid-request"


# =============================================================================
# Route Registration Tests
# =============================================================================


def test_register_routes_includes_all_routers(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify register_routes adds all expected route groups.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    routes: list[str] = []
    for route in app.routes:
        if hasattr(route, "path"):
            path_value = getattr(route, "path", None)
            if isinstance(path_value, str):
                routes.append(path_value)

    # Verify key route groups are registered
    assert "/health" in routes
    assert any("/function" in r for r in routes)
    assert any("/architecture" in r for r in routes)
    assert any("/datasets" in r for r in routes)


def test_create_app_with_auto_pipeline_option(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Verify create_app accepts auto_pipeline option.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
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

    # Should not raise
    app = create_app(
        config_loader=load_config,
        backend_factory=backend_factory,
        auto_pipeline=True,
    )

    assert app is not None
