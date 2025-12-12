"""Tests for FastAPI app factory and middleware.

This module tests the FastAPI application factory, exception handlers,
and middleware using real gateways and TestClient - no mocking.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from fastapi import status

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.bootstrap import build_backend_resource
from codeintel.serving.http.fastapi import load_api_config, problem_response
from codeintel.serving.services.errors import ProblemDetail
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.assertions.http_responses import assert_problem_detail_response

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tests._helpers.context import TestContext


def test_load_api_config_local_db_path_not_found(tmp_path: Path, codeintel_env: None) -> None:
    """Verify FileNotFoundError when local_db path does not exist.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    codeintel_env
        Fixture that saves/restores CODEINTEL_* env vars.
    """
    _ = codeintel_env
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
    _ = codeintel_env
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
    _ = codeintel_env
    os.environ["CODEINTEL_REPO"] = "test/repo"
    os.environ["CODEINTEL_COMMIT"] = "abc123"
    os.environ["CODEINTEL_MCP_MODE"] = "remote_api"
    os.environ["CODEINTEL_API_BASE_URL"] = "http://example.com"

    config = load_api_config()

    expect_equal(config.repo, "test/repo")
    expect_equal(config.commit, "abc123")
    expect_equal(config.mode, "remote_api")


def test_load_api_config_local_db_success(tmp_path: Path, codeintel_env: None) -> None:
    """Verify config loads successfully for local_db mode with valid file.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    codeintel_env
        Fixture that saves/restores CODEINTEL_* env vars.
    """
    _ = codeintel_env
    db_file = tmp_path / "valid.duckdb"
    db_file.write_bytes(b"valid content")
    os.environ["CODEINTEL_REPO"] = "test/repo"
    os.environ["CODEINTEL_COMMIT"] = "abc123"
    os.environ["CODEINTEL_MCP_MODE"] = "local_db"
    os.environ["CODEINTEL_DB_PATH"] = str(db_file)

    config = load_api_config()

    expect_equal(config.repo, "test/repo")
    expect_equal(config.mode, "local_db")
    expect_equal(config.db_path, db_file)


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

    expect_equal(response.status_code, status.HTTP_400_BAD_REQUEST)
    body_bytes = response.body
    body = body_bytes.decode("utf-8") if isinstance(body_bytes, bytes) else str(body_bytes)
    expect_in("test-error", body)
    expect_in("Test Error", body)


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

    expect_equal(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)


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
    provisioned_repo: TestContext,
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

    expect_is_not_none(resource.backend)
    expect_true(callable(resource.close))


def test_create_app_with_provisioned_gateway(
    provisioned_http_client: TestClient,
    provisioned_repo: TestContext,
) -> None:
    """Verify create_app creates functional FastAPI app with real gateway.

    Parameters
    ----------
    provisioned_http_client
        Test client bound to the provisioned app.
    provisioned_repo
        Provisioned gateway snapshot metadata.
    """
    response = provisioned_http_client.get("/health")

    expect_equal(response.status_code, status.HTTP_200_OK)
    payload = response.json()
    expect_equal(payload["status"], "ok")
    expect_equal(payload["repo"], provisioned_repo.repo)


def test_create_app_correlation_id_from_header(
    provisioned_http_client: TestClient,
) -> None:
    """Verify correlation ID from X-Request-ID header is propagated.

    Parameters
    ----------
    provisioned_http_client
        Test client bound to the provisioned app.
    """
    response = provisioned_http_client.get(
        "/health",
        headers={"X-Request-ID": "test-correlation-id"},
    )

    expect_equal(response.headers.get("X-Request-ID"), "test-correlation-id")


def test_create_app_correlation_id_generated_when_missing(
    provisioned_http_client: TestClient,
) -> None:
    """Verify correlation ID is generated when not provided in headers.

    Parameters
    ----------
    provisioned_http_client
        Test client bound to the provisioned app.
    """
    response = provisioned_http_client.get("/health")

    correlation_id = response.headers.get("X-Request-ID")
    correlation_id = expect_is_not_none(correlation_id)
    expect_true(len(correlation_id) > 0)


def test_create_app_x_correlation_id_header(
    provisioned_http_client: TestClient,
) -> None:
    """Verify X-Correlation-ID header is also accepted.

    Parameters
    ----------
    provisioned_http_client
        Test client bound to the provisioned app.
    """
    response = provisioned_http_client.get(
        "/health",
        headers={"X-Correlation-ID": "alt-correlation-id"},
    )

    expect_equal(response.headers.get("X-Request-ID"), "alt-correlation-id")


def test_exception_handler_problem_error(
    provisioned_http_client: TestClient,
) -> None:
    """Verify ProblemError is converted to proper JSON response.

    Parameters
    ----------
    provisioned_http_client
        Test client bound to the provisioned app.
    """
    response = provisioned_http_client.get("/datasets/nonexistent_dataset")

    assert_problem_detail_response(
        response,
        status_code=status.HTTP_400_BAD_REQUEST,
    )
    payload = response.json()
    expect_equal(payload.get("code"), "dataset-not-found")


def test_exception_handler_validation_error(
    provisioned_http_client: TestClient,
) -> None:
    """Verify validation errors return proper Problem Details.

    Parameters
    ----------
    provisioned_http_client
        Test client bound to the provisioned app.
    """
    response = provisioned_http_client.get("/functions/high-risk?limit=invalid")

    assert_problem_detail_response(
        response,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )
    payload = response.json()
    expect_equal(payload.get("code"), "invalid-request")


def test_register_routes_includes_all_routers(
    provisioned_http_app: FastAPI,
) -> None:
    """Verify register_routes adds all expected route groups.

    Parameters
    ----------
    provisioned_http_app
        FastAPI app bound to the provisioned gateway.
    """
    routes: list[str] = []
    for route in provisioned_http_app.routes:
        if hasattr(route, "path"):
            path_value = getattr(route, "path", None)
            if isinstance(path_value, str):
                routes.append(path_value)

    expect_in("/health", routes)
    expect_true(any("/function" in r for r in routes))
    expect_true(any("/architecture" in r for r in routes))
    expect_true(any("/datasets" in r for r in routes))


def test_create_app_with_auto_pipeline_option(
    provisioned_repo: TestContext,
    make_http_app: Callable[..., FastAPI],
) -> None:
    """Verify create_app accepts auto_pipeline option.

    Parameters
    ----------
    provisioned_repo
        Provisioned gateway fixture.
    make_http_app
        Factory fixture that constructs FastAPI applications for serving tests.
    """
    app = make_http_app(
        gateway=provisioned_repo.gateway,
        snapshot=(provisioned_repo.repo, provisioned_repo.commit),
        auto_pipeline=True,
    )

    expect_is_not_none(app)
