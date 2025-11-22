"""FastAPI route coverage for CodeIntel API surface."""

from __future__ import annotations

from collections.abc import Iterator
from http import HTTPStatus
from pathlib import Path

import duckdb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from codeintel.mcp.backend import DuckDBBackend
from codeintel.mcp.config import McpServerConfig
from codeintel.server.fastapi import ApiAppConfig, BackendResource, create_app


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """
    Provide a temporary DuckDB file path.

    Returns
    -------
    Path
        Filesystem path for the temporary database.
    """
    return tmp_path / "codeintel.duckdb"


@pytest.fixture
def con(db_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Create an in-memory DuckDB with minimal tables for API tests.

    Yields
    ------
    duckdb.DuckDBPyConnection
        Seeded connection that is cleaned up after each test.
    """
    connection = duckdb.connect(str(db_path))
    connection.execute("CREATE SCHEMA IF NOT EXISTS docs;")
    connection.execute("CREATE SCHEMA IF NOT EXISTS analytics;")
    connection.execute(
        """
        CREATE TABLE docs.v_function_summary (
            repo TEXT,
            commit TEXT,
            rel_path TEXT,
            qualname TEXT,
            urn TEXT,
            function_goid_h128 DECIMAL(38,0)
        );
        """
    )
    connection.execute(
        """
        INSERT INTO docs.v_function_summary VALUES
        ('r', 'c', 'foo.py', 'foo', 'urn:foo', 1);
        """
    )
    connection.execute(
        """
        CREATE TABLE docs.v_test_to_function (
            repo TEXT,
            commit TEXT,
            test_id TEXT,
            function_goid_h128 DECIMAL(38,0)
        );
        """
    )
    connection.execute(
        """
        INSERT INTO docs.v_test_to_function VALUES
        ('r', 'c', 't1', 1);
        """
    )

    yield connection
    connection.close()


@pytest.fixture
def api_config(db_path: Path) -> ApiAppConfig:
    """
    Build an ApiAppConfig pinned to the temporary DuckDB.

    Returns
    -------
    ApiAppConfig
        Application configuration for the test app.
    """
    server_cfg = McpServerConfig(
        mode="local_db",
        repo_root=db_path.parent,
        repo="r",
        commit="c",
        db_path=db_path,
    )
    return ApiAppConfig(server=server_cfg, read_only=True)


@pytest.fixture
def backend(con: duckdb.DuckDBPyConnection) -> DuckDBBackend:
    """
    Provide a DuckDBBackend bound to the seeded connection.

    Returns
    -------
    DuckDBBackend
        Backend backed by the temporary DuckDB.
    """
    return DuckDBBackend(
        con=con,
        repo="r",
        commit="c",
        dataset_tables={"v_function_summary": "docs.v_function_summary"},
    )


@pytest.fixture
def app(api_config: ApiAppConfig, backend: DuckDBBackend) -> FastAPI:
    """
    Construct a FastAPI app using injected configuration and backend.

    Returns
    -------
    FastAPI
        Application instance under test.
    """

    def _config_loader() -> ApiAppConfig:
        return api_config

    def _backend_factory(_: ApiAppConfig) -> BackendResource:
        return BackendResource(backend=backend, close=lambda: None)

    return create_app(config_loader=_config_loader, backend_factory=_backend_factory)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """
    Yield a TestClient that runs startup/shutdown events.

    Yields
    ------
    TestClient
        HTTP client for exercising the API.
    """
    with TestClient(app) as client:
        yield client


def test_function_summary_success(client: TestClient) -> None:
    """Return function summary when URN is present."""
    response = client.get("/function/summary", params={"urn": "urn:foo"})
    if response.status_code != HTTPStatus.OK:
        pytest.fail("Expected a successful function summary response")
    payload = response.json()
    if not payload.get("found"):
        pytest.fail("Expected function summary to be marked as found")
    summary = payload.get("summary") or {}
    if summary.get("qualname") != "foo":
        pytest.fail("Unexpected qualname in function summary payload")


def test_function_summary_not_found(client: TestClient) -> None:
    """Return Problem Details when the function does not exist."""
    response = client.get("/function/summary", params={"urn": "urn:missing"})
    if response.status_code != HTTPStatus.NOT_FOUND:
        pytest.fail("Expected 404 for missing function summary")
    problem = response.json()
    if problem.get("title") != "Not found":
        pytest.fail("Problem detail title mismatch for not-found summary")


def test_tests_for_function_validation(client: TestClient) -> None:
    """Reject missing identifiers when requesting tests for a function."""
    response = client.get("/function/tests")
    if response.status_code != HTTPStatus.BAD_REQUEST:
        pytest.fail("Expected validation error when identifiers are absent")
    problem = response.json()
    if problem.get("title") != "Invalid argument":
        pytest.fail("Problem detail title mismatch for invalid argument")


def test_tests_for_function_success(client: TestClient) -> None:
    """Return test rows for a valid function."""
    response = client.get("/function/tests", params={"goid_h128": 1, "limit": 1})
    if response.status_code != HTTPStatus.OK:
        pytest.fail("Expected success when fetching tests for function")
    payload = response.json()
    tests = payload.get("tests") or []
    if len(tests) != 1:
        pytest.fail("Expected exactly one test row in response")
    if tests[0].get("test_id") != "t1":
        pytest.fail("Unexpected test_id returned from API")


def test_dataset_rows_unknown_dataset(client: TestClient) -> None:
    """Return a Problem Detail for unknown dataset names."""
    response = client.get("/datasets/unknown")
    if response.status_code != HTTPStatus.BAD_REQUEST:
        pytest.fail("Expected 400 for unknown dataset name")
    problem = response.json()
    if problem.get("title") != "Invalid argument":
        pytest.fail("Problem detail title mismatch for unknown dataset")


def test_dataset_rows_limit_clamped(client: TestClient) -> None:
    """Reject dataset reads that exceed configured limits."""
    response = client.get("/datasets/v_function_summary", params={"limit": 500})
    if response.status_code != HTTPStatus.BAD_REQUEST:
        pytest.fail("Expected limit validation for oversized dataset request")


def test_dataset_rows_success(client: TestClient) -> None:
    """Return dataset rows for configured datasets."""
    response = client.get("/datasets/v_function_summary", params={"limit": 1})
    if response.status_code != HTTPStatus.OK:
        pytest.fail("Expected dataset rows for configured dataset")
    payload = response.json()
    rows = payload.get("rows") or []
    if len(rows) != 1:
        pytest.fail("Expected a single dataset row in response")
    if rows[0].get("urn") != "urn:foo":
        pytest.fail("Unexpected dataset content returned")
