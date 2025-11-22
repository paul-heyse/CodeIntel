"""Integration smoke tests for FastAPI endpoints against a seeded DuckDB."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import httpx
import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from codeintel.mcp.backend import DuckDBBackend
from codeintel.mcp.config import McpServerConfig
from codeintel.server.fastapi import ApiAppConfig, BackendResource, create_app
from codeintel.storage.views import create_all_views


def _seed_db(db_path: Path, *, repo: str, commit: str) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS core;")
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")
    con.execute("CREATE SCHEMA IF NOT EXISTS graph;")
    con.execute("CREATE SCHEMA IF NOT EXISTS docs;")

    con.execute(
        """
        CREATE TABLE core.goids(
            goid_h128 DECIMAL(38,0),
            urn TEXT,
            repo TEXT,
            commit TEXT,
            rel_path TEXT,
            language TEXT,
            kind TEXT,
            qualname TEXT,
            start_line INTEGER,
            end_line INTEGER
        );
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.function_metrics(
            function_goid_h128 DECIMAL(38,0),
            urn TEXT,
            repo TEXT,
            commit TEXT,
            rel_path TEXT,
            language TEXT,
            kind TEXT,
            qualname TEXT,
            start_line INTEGER,
            end_line INTEGER,
            loc INTEGER,
            logical_loc INTEGER,
            param_count INTEGER,
            positional_params INTEGER,
            keyword_only_params INTEGER,
            has_varargs BOOLEAN,
            has_varkw BOOLEAN,
            is_async BOOLEAN,
            is_generator BOOLEAN,
            return_count INTEGER,
            yield_count INTEGER,
            raise_count INTEGER,
            cyclomatic_complexity INTEGER,
            max_nesting_depth INTEGER,
            stmt_count INTEGER,
            decorator_count INTEGER,
            has_docstring BOOLEAN,
            complexity_bucket TEXT,
            created_at TIMESTAMP
        );
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.goid_risk_factors(
            function_goid_h128 DECIMAL(38,0),
            urn TEXT,
            repo TEXT,
            commit TEXT,
            rel_path TEXT,
            language TEXT,
            kind TEXT,
            qualname TEXT,
            loc INTEGER,
            logical_loc INTEGER,
            cyclomatic_complexity INTEGER,
            complexity_bucket TEXT,
            typedness_bucket TEXT,
            typedness_source TEXT,
            hotspot_score DOUBLE,
            file_typed_ratio DOUBLE,
            static_error_count INTEGER,
            has_static_errors BOOLEAN,
            executable_lines INTEGER,
            covered_lines INTEGER,
            coverage_ratio DOUBLE,
            tested BOOLEAN,
            test_count INTEGER,
            failing_test_count INTEGER,
            last_test_status TEXT,
            risk_score DOUBLE,
            risk_level TEXT,
            tags TEXT,
            owners TEXT,
            created_at TIMESTAMP
        );
        """
    )
    con.execute(
        """
        CREATE TABLE graph.call_graph_edges(
            caller_goid_h128 DECIMAL(38,0),
            callee_goid_h128 DECIMAL(38,0),
            callsite_path TEXT,
            callsite_line INTEGER,
            callsite_col INTEGER,
            language TEXT,
            kind TEXT,
            resolved_via TEXT,
            confidence DOUBLE,
            evidence_json TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.test_catalog(
            test_id TEXT,
            test_goid_h128 DECIMAL(38,0),
            urn TEXT,
            repo TEXT,
            commit TEXT,
            rel_path TEXT,
            qualname TEXT,
            kind TEXT,
            status TEXT,
            duration_ms DOUBLE,
            markers TEXT,
            parametrized BOOLEAN,
            flaky BOOLEAN
        );
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.test_coverage_edges(
            test_id TEXT,
            test_goid_h128 DECIMAL(38,0),
            function_goid_h128 DECIMAL(38,0),
            urn TEXT,
            repo TEXT,
            commit TEXT,
            rel_path TEXT,
            qualname TEXT,
            covered_lines INTEGER,
            executable_lines INTEGER,
            coverage_ratio DOUBLE,
            last_status TEXT,
            created_at TIMESTAMP
        );
        """
    )
    con.execute(
        """
        CREATE TABLE core.modules(
            repo TEXT,
            commit TEXT,
            path TEXT,
            module TEXT,
            language TEXT,
            tags TEXT,
            owners TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.hotspots(
            rel_path TEXT,
            commit_count INTEGER,
            author_count INTEGER,
            lines_added INTEGER,
            lines_deleted INTEGER,
            complexity DOUBLE,
            score DOUBLE
        );
        """
    )
    con.execute(
        """
        CREATE TABLE core.ast_metrics(
            rel_path TEXT,
            node_count INTEGER,
            function_count INTEGER,
            class_count INTEGER,
            avg_depth DOUBLE,
            max_depth DOUBLE,
            complexity DOUBLE
        );
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.typedness(
            path TEXT,
            type_error_count INTEGER,
            annotation_ratio JSON,
            untyped_defs INTEGER,
            overlay_needed BOOLEAN
        );
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.static_diagnostics(
            rel_path TEXT,
            total_errors INTEGER,
            has_errors BOOLEAN
        );
        """
    )

    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO core.goids VALUES (?, ?, ?, ?, ?, 'python', 'function', 'pkg.mod.func', 1, 2)
        """,
        [1, "goid:demo/repo#python:function:pkg.mod.func", repo, commit, "pkg/mod.py"],
    )
    con.execute(
        """
        INSERT INTO analytics.function_metrics VALUES
        (?, ?, ?, ?, ?, 'python', 'function', ?, 1, 2, 2, 2, 0, 0, 0, FALSE, FALSE, FALSE, FALSE, 0, 0, 0, 1, 1, 0, 0, FALSE, 'low', ?)
        """,
        [1, "goid:demo/repo#python:function:pkg.mod.func", repo, commit, "pkg/mod.py", "pkg.mod.func", now],
    )
    con.execute(
        """
        INSERT INTO analytics.goid_risk_factors VALUES
        (?, ?, ?, ?, ?, 'python', 'function', ?, 10, 10, 1, 'low', 'full', 'annotations', 0.1, 1.0, 0, FALSE, 2, 2, 1.0, TRUE, 1, 0, 'passed', 0.2, 'low', '[]', '[]', ?)
        """,
        [1, "goid:demo/repo#python:function:pkg.mod.func", repo, commit, "pkg/mod.py", "pkg.mod.func", now],
    )
    con.execute(
        """
        INSERT INTO graph.call_graph_edges VALUES
        (?, ?, 'pkg/mod.py', 1, 1, 'python', 'direct', 'local', 0.9, '{}')
        """,
        [1, 1],
    )
    con.execute(
        """
        INSERT INTO analytics.test_catalog VALUES
        ('pkg/mod.py::test_func', 10, 'goid:demo/repo#python:function:pkg.mod.test_func', ?, ?, 'pkg/mod.py', 'pkg.mod.test_func', 'test', 'passed', 1.0, '[]', FALSE, FALSE)
        """,
        [repo, commit],
    )
    con.execute(
        """
        INSERT INTO analytics.test_coverage_edges VALUES
        ('pkg/mod.py::test_func', 10, 1, 'goid:demo/repo#python:function:pkg.mod.func', ?, ?, 'pkg/mod.py', 'pkg.mod.func', 2, 2, 1.0, 'passed', ?)
        """,
        [repo, commit, now],
    )
    con.execute(
        """
        INSERT INTO core.modules VALUES (?, ?, 'pkg/mod.py', 'pkg.mod', 'python', '[]', '[]')
        """,
        [repo, commit],
    )
    con.execute(
        """
        INSERT INTO analytics.hotspots VALUES ('pkg/mod.py', 1, 1, 1, 1, 1.0, 0.1)
        """
    )
    con.execute(
        """
        INSERT INTO core.ast_metrics VALUES ('pkg/mod.py', 1, 1, 0, 1.0, 1.0, 0.1)
        """
    )
    con.execute(
        """
        INSERT INTO analytics.typedness VALUES ('pkg/mod.py', 0, '{"params":1.0}', 0, FALSE)
        """
    )
    con.execute(
        """
        INSERT INTO analytics.static_diagnostics VALUES ('pkg/mod.py', 0, FALSE)
        """
    )

    create_all_views(con)
    return con


def _build_app(con: duckdb.DuckDBPyConnection, db_path: Path, *, repo: str, commit: str) -> FastAPI:
    """
    Construct a FastAPI app bound to the provided DuckDB connection.

    Returns
    -------
    FastAPI
        Configured application using the supplied backend.
    """
    cfg = ApiAppConfig(server=McpServerConfig(repo_root=db_path.parent, repo=repo, commit=commit), read_only=True)

    def _loader() -> ApiAppConfig:
        return cfg

    def _backend_factory(_: ApiAppConfig) -> BackendResource:
        backend = DuckDBBackend(con=con, repo=repo, commit=commit, dataset_tables={})
        return BackendResource(backend=backend, close=lambda: None)

    return create_app(config_loader=_loader, backend_factory=_backend_factory)


def test_fastapi_endpoints_smoke(tmp_path: Path) -> None:
    """Seed a DuckDB and verify key FastAPI endpoints return data."""
    repo = "demo/repo"
    commit = "deadbeef"
    db_path = tmp_path / "codeintel.duckdb"
    con = _seed_db(db_path, repo=repo, commit=commit)
    app = _build_app(con, db_path, repo=repo, commit=commit)
    with TestClient(app) as client:
        _assert_function_summary(client)
        _assert_callgraph(client)
        _assert_tests_for_function(client)
        _assert_file_summary(client)
        _assert_dataset_endpoints(client)


def _assert_function_summary(client: TestClient) -> None:
    resp = client.get(
        "/function/summary",
        params={"urn": "goid:demo/repo#python:function:pkg.mod.func"},
    )
    payload = _expect_dict(_json_or_fail(resp))
    summary = cast("dict[str, object]", payload["summary"])
    if not payload["found"]:
        pytest.fail("Function should be found")
    if summary.get("qualname") != "pkg.mod.func":
        pytest.fail("Unexpected qualname in function summary")


def _assert_callgraph(client: TestClient) -> None:
    resp = client.get("/function/callgraph", params={"goid_h128": 1, "direction": "both", "limit": 10})
    cg = _expect_dict(_json_or_fail(resp))
    if not cg["outgoing"]:
        pytest.fail("Expected outgoing callgraph edges")


def _assert_tests_for_function(client: TestClient) -> None:
    resp = client.get("/function/tests", params={"goid_h128": 1})
    tests_payload = _expect_dict(_json_or_fail(resp))
    tests = cast("list[dict[str, Any]]", tests_payload["tests"])
    if not tests:
        pytest.fail("Expected tests for function")
    if tests[0]["test_id"] != "pkg/mod.py::test_func":
        pytest.fail("Unexpected test_id in tests-for-function response")


def _assert_file_summary(client: TestClient) -> None:
    resp = client.get("/file/summary", params={"rel_path": "pkg/mod.py"})
    fs_payload = _expect_dict(_json_or_fail(resp))
    if not fs_payload["found"]:
        pytest.fail("File summary should be found")


def _assert_dataset_endpoints(client: TestClient) -> None:
    resp = client.get("/datasets")
    datasets = _json_or_fail(resp)
    if not isinstance(datasets, list):
        pytest.fail("Datasets payload was not a list")
    dataset_rows = [cast("dict[str, object]", ds) for ds in datasets if isinstance(ds, dict)]
    if not dataset_rows:
        pytest.fail("Dataset rows were not objects")
    if not any(ds.get("name") == "function_metrics" for ds in dataset_rows):
        pytest.fail("function_metrics dataset missing from list")

    resp = client.get("/datasets/function_metrics", params={"limit": 5})
    rows_payload = _expect_dict(_json_or_fail(resp))
    rows = cast("list[dict[str, Any]]", rows_payload["rows"])
    if not rows:
        pytest.fail("Expected dataset rows for function_metrics")
    if rows[0]["qualname"] != "pkg.mod.func":
        pytest.fail("Unexpected qualname in dataset rows")


def _json_or_fail(resp: httpx.Response) -> dict[str, object] | list[object]:
    """
    Parse JSON payload or fail with a helpful message.

    Returns
    -------
    dict[str, object] | list[object]
        Parsed JSON payload when status is 200.
    """
    if resp.status_code != status.HTTP_200_OK:
        pytest.fail(f"Unexpected status {resp.status_code}: {resp.text}")
    data = resp.json()
    if not isinstance(data, (dict, list)):
        pytest.fail("Response payload is not JSON object/array")
    return data


def _expect_dict(payload: dict[str, object] | list[object]) -> dict[str, object]:
    """
    Ensure payload is a dict or fail.

    Returns
    -------
    dict[str, object]
        Validated JSON object payload.
    """
    if not isinstance(payload, dict):
        pytest.fail("Expected JSON object payload")
    return payload
