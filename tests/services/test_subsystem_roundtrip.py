"""Integration tests for subsystem profile/coverage round trips."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.http.fastapi import create_app
from codeintel.serving.mcp.query_service import BackendLimits, DuckDBQueryService
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageGateway, open_memory_gateway

REPO = "demo/repo"
COMMIT = "deadbeef"
DEFAULT_LIMIT = 50
MAX_ROWS_PER_CALL = 500
PROFILE_LIMIT = 10
COVERAGE_LIMIT = 5
EXPECTED_TEST_COUNT = 2
ENTRYPOINTS = [{"path": "/alpha"}, {"path": "/beta"}]
HTTP_OK = 200


def _seed_subsystem_fixture() -> tuple[StorageGateway, LocalQueryService]:
    gateway = open_memory_gateway(ensure_views=True, validate_schema=False)
    con = gateway.con
    now = datetime.now(UTC)
    gateway.core.insert_repo_map([(REPO, COMMIT, "[]", "[]", now.isoformat())])

    con.execute(
        """
        INSERT INTO analytics.subsystems (
            repo, commit, subsystem_id, name, description, module_count, modules_json,
            entrypoints_json, internal_edge_count, external_edge_count, fan_in, fan_out,
            function_count, avg_risk_score, max_risk_score, high_risk_function_count,
            risk_level, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            REPO,
            COMMIT,
            "subsysdemo",
            "Subsystem Demo",
            "Demo subsystem",
            2,
            ["pkg.alpha", "pkg.beta"],
            ENTRYPOINTS,
            1,
            2,
            3,
            4,
            5,
            0.4,
            0.7,
            1,
            "medium",
            now,
        ),
    )
    con.execute(
        """
        INSERT INTO analytics.subsystem_graph_metrics (
            repo, commit, subsystem_id, import_in_degree, import_out_degree,
            import_pagerank, import_betweenness, import_closeness, import_layer, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (REPO, COMMIT, "subsysdemo", 1.0, 2.0, 0.5, 0.1, 0.2, 1, now),
    )
    con.execute(
        """
        INSERT INTO analytics.test_profile (
            repo, commit, test_id, rel_path, status, functions_covered_count,
            subsystems_covered, subsystems_covered_count, primary_subsystem_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            REPO,
            COMMIT,
            "test::one",
            "tests/test_mod.py",
            "passed",
            3,
            ["subsysdemo"],
            1,
            "subsysdemo",
            now,
        ),
    )
    con.execute(
        """
        INSERT INTO analytics.test_profile (
            repo, commit, test_id, rel_path, status, functions_covered_count,
            subsystems_covered, subsystems_covered_count, primary_subsystem_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            REPO,
            COMMIT,
            "test::two",
            "tests/test_mod.py",
            "failed",
            1,
            ["subsysdemo"],
            1,
            "subsysdemo",
            now,
        ),
    )

    query = DuckDBQueryService(
        gateway=gateway,
        repo=REPO,
        commit=COMMIT,
        limits=BackendLimits(default_limit=DEFAULT_LIMIT, max_rows_per_call=MAX_ROWS_PER_CALL),
    )
    return gateway, LocalQueryService(query=query)


def _http_client_for(gateway: StorageGateway) -> TestClient:
    def _config_loader() -> ServingConfig:
        return ServingConfig(
            mode="local_db",
            repo=REPO,
            commit=COMMIT,
            db_path=Path(":memory:"),
            default_limit=DEFAULT_LIMIT,
            max_rows_per_call=MAX_ROWS_PER_CALL,
            read_only=True,
        )

    app = create_app(config_loader=_config_loader, gateway=gateway)
    return TestClient(app)


def test_subsystem_profile_roundtrip() -> None:
    """Profiles should return consistent payloads locally and via HTTP."""
    gateway, local = _seed_subsystem_fixture()

    local_profiles = local.list_subsystem_profiles(limit=PROFILE_LIMIT)
    if not local_profiles.profiles:
        pytest.fail("Expected subsystem profile rows")
    profile = local_profiles.profiles[0]
    if profile.entrypoints_json != ENTRYPOINTS:
        pytest.fail("Entrypoints were not normalized to list-of-dicts")
    if profile.risk_level is None or profile.risk_level.value != "medium":
        pytest.fail("Risk level did not round-trip as enum")

    with _http_client_for(gateway) as http:
        resp = http.get("/architecture/subsystem-profiles", params={"limit": PROFILE_LIMIT})
        if resp.status_code != HTTP_OK:
            pytest.fail(f"HTTP profile request failed: {resp.status_code}")
        payload = resp.json()
        rows = payload.get("profiles", [])
        if not rows:
            pytest.fail("Expected HTTP subsystem profile rows")
        if rows[0]["subsystem_id"] != profile.subsystem_id:
            pytest.fail("HTTP profile payload did not match local result")


def test_subsystem_coverage_roundtrip() -> None:
    """Coverage should round-trip locally and via HTTP with computed aggregates."""
    gateway, local = _seed_subsystem_fixture()
    local_cov = local.list_subsystem_coverage(limit=COVERAGE_LIMIT)
    if not local_cov.coverage:
        pytest.fail("Expected subsystem coverage rows")
    cov = local_cov.coverage[0]
    if cov.test_count != EXPECTED_TEST_COUNT:
        pytest.fail("Coverage aggregates did not compute expected test_count")

    with _http_client_for(gateway) as http:
        http_cov = http.get("/architecture/subsystem-coverage", params={"limit": COVERAGE_LIMIT})
        if http_cov.status_code != HTTP_OK:
            pytest.fail(f"HTTP coverage request failed: {http_cov.status_code}")
        cov_rows = http_cov.json().get("coverage", [])
        if not cov_rows:
            pytest.fail("Expected HTTP coverage rows")
        if cov_rows[0]["function_coverage_ratio"] is None:
            pytest.fail("Coverage ratio should be computed when counts exist")
        if cov_rows[0]["test_count"] != EXPECTED_TEST_COUNT:
            pytest.fail("HTTP coverage rows did not match expected test count")


def test_subsystem_coverage_not_found_meta() -> None:
    """When no coverage rows exist, meta should surface a not_found message."""
    gateway = open_memory_gateway(ensure_views=True, validate_schema=False)
    query = DuckDBQueryService(
        gateway=gateway,
        repo="demo/repo",
        commit="deadbeef",
        limits=BackendLimits(default_limit=10, max_rows_per_call=100),
    )
    local = LocalQueryService(query=query)
    resp = local.list_subsystem_coverage(limit=1)
    if resp.coverage:
        pytest.fail("Expected no coverage rows in empty database")
    if not resp.meta.messages or resp.meta.messages[0].code != "not_found":
        pytest.fail("Expected not_found message in meta for empty coverage")
