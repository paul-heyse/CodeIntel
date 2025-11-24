"""FastAPI route coverage for CodeIntel API surface."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from codeintel.mcp.backend import MAX_ROWS_LIMIT, DuckDBBackend
from codeintel.mcp.config import McpServerConfig
from codeintel.server.fastapi import ApiAppConfig, BackendResource, create_app
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway
from codeintel.storage.views import create_all_views


def _seed_api_data(gateway: StorageGateway) -> None:
    """
    Populate minimal rows required for API tests.

    Parameters
    ----------
    gateway
        Storage gateway providing the DuckDB connection.
    """
    now = datetime.now(tz=UTC)
    con = gateway.con
    con.execute(
        """
        INSERT INTO analytics.goid_risk_factors (
            function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            coverage_ratio, risk_score, risk_level, tags, owners, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1.0, 0.1, 'low', '[]', '[]', ?)
        """,
        [1, "urn:foo", "r", "c", "foo.py", "python", "function", "foo", now],
    )
    con.execute(
        """
        INSERT INTO analytics.function_metrics (
            function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [1, "urn:foo", "r", "c", "foo.py", "python", "function", "foo", now],
    )
    con.execute(
        """
        INSERT INTO analytics.test_catalog (
            test_id, test_goid_h128, urn, repo, commit, rel_path, qualname, kind, status,
            duration_ms, markers, parametrized, flaky, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'passed', 10, '[]', FALSE, FALSE, ?)
        """,
        [
            "t1",
            2,
            "urn:test",
            "r",
            "c",
            "tests/test_sample.py",
            "tests.test_sample.test_case",
            "function",
            now,
        ],
    )
    con.execute(
        """
        INSERT INTO analytics.test_coverage_edges (
            test_id, test_goid_h128, function_goid_h128, urn, repo, commit, rel_path,
            qualname, covered_lines, executable_lines, coverage_ratio, last_status,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 1.0, 'passed', ?)
        """,
        ["t1", 2, 1, "urn:foo", "r", "c", "foo.py", "foo", now],
    )
    create_all_views(con)


@pytest.fixture
def gateway(tmp_path: Path) -> Iterator[StorageGateway]:
    """
    Create a temporary DuckDB gateway with schemas and views applied.

    Yields
    ------
    StorageGateway
        Gateway bound to a temporary DuckDB database.
    """
    db_path = tmp_path / "codeintel.duckdb"
    config = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=True,
        ensure_views=True,
        validate_schema=True,
    )
    gw = open_gateway(config)
    _seed_api_data(gw)
    yield gw
    gw.close()


@pytest.fixture
def api_config(gateway: StorageGateway) -> ApiAppConfig:
    """
    Build an ApiAppConfig pinned to the gateway database.

    Returns
    -------
    ApiAppConfig
        Application configuration for the test app.
    """
    db_path = gateway.config.db_path
    server_cfg = McpServerConfig(
        mode="local_db",
        repo_root=db_path.parent,
        repo="r",
        commit="c",
        db_path=db_path,
    )
    return ApiAppConfig(server=server_cfg, read_only=True)


@pytest.fixture
def backend(gateway: StorageGateway) -> DuckDBBackend:
    """
    Provide a DuckDBBackend bound to the seeded gateway.

    Returns
    -------
    DuckDBBackend
        Backend backed by the temporary DuckDB.
    """
    return DuckDBBackend(gateway=gateway, repo="r", commit="c")


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
        return BackendResource(backend=backend, service=backend.service, close=lambda: None)

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


def test_backend_registry_matches_gateway(gateway: StorageGateway, backend: DuckDBBackend) -> None:
    """Backend should inherit dataset registry from the gateway."""
    if backend.dataset_tables != dict(gateway.datasets.mapping):
        pytest.fail("Backend dataset registry should match gateway mapping")


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
    """Clamp dataset reads that exceed configured limits with messaging."""
    response = client.get("/datasets/v_function_summary", params={"limit": 2000})
    if response.status_code != HTTPStatus.OK:
        pytest.fail("Expected clamped success for oversized dataset request")
    payload = response.json()
    meta = payload.get("meta") or {}
    messages = {m.get("code") for m in meta.get("messages", [])}
    if "limit_clamped" not in messages:
        pytest.fail(f"Expected limit_clamped message; got {messages}")
    if payload.get("limit") != MAX_ROWS_LIMIT:
        pytest.fail("Expected applied limit to match server maximum")


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
