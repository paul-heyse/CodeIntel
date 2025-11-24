"""Integration smoke tests for FastAPI endpoints against a seeded DuckDB."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import anyio
import httpx
import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from codeintel.mcp.backend import BackendLimits, DuckDBBackend, HttpBackend
from codeintel.mcp.config import McpServerConfig
from codeintel.server.fastapi import ApiAppConfig, BackendResource, create_app
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway


def _seed_db(db_path: Path, *, repo: str, commit: str) -> StorageGateway:
    config = StorageConfig(
        db_path=db_path,
        apply_schema=True,
        ensure_views=True,
        validate_schema=True,
    )
    gateway = open_gateway(config)
    now = datetime.now(UTC)
    now_iso = now.isoformat()

    gateway.core.insert_modules([("pkg.mod", "pkg/mod.py", repo, commit)])
    gateway.core.insert_goids(
        [
            (
                1,
                "goid:demo/repo#python:function:pkg.mod.func",
                repo,
                commit,
                "pkg/mod.py",
                "python",
                "function",
                "pkg.mod.func",
                1,
                2,
                now_iso,
            )
        ]
    )
    gateway.analytics.insert_function_metrics(
        [
            (
                1,
                "goid:demo/repo#python:function:pkg.mod.func",
                repo,
                commit,
                "pkg/mod.py",
                "python",
                "function",
                "pkg.mod.func",
                1,
                2,
                2,
                2,
                0,
                0,
                0,
                False,
                False,
                False,
                False,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                False,
                "low",
                now_iso,
            )
        ]
    )
    gateway.analytics.insert_goid_risk_factors(
        [
            (
                1,
                "goid:demo/repo#python:function:pkg.mod.func",
                repo,
                commit,
                "pkg/mod.py",
                "python",
                "function",
                "pkg.mod.func",
                10,
                10,
                1,
                "low",
                "full",
                "annotations",
                0.1,
                1.0,
                0,
                False,
                2,
                2,
                1.0,
                True,
                1,
                0,
                "passed",
                0.2,
                "low",
                "[]",
                "[]",
                now_iso,
            )
        ]
    )
    gateway.graph.insert_call_graph_edges(
        [
            (
                repo,
                commit,
                1,
                1,
                "pkg/mod.py",
                1,
                1,
                "python",
                "direct",
                "local",
                0.9,
                "{}",
            )
        ]
    )
    gateway.analytics.insert_graph_metrics_functions(
        [(repo, commit, 1, 2, 3, 2, 3, 0.5, 0.1, 0.2, False, 0, 1, now_iso)]
    )
    gateway.analytics.insert_graph_metrics_modules(
        [
            (
                repo,
                commit,
                "pkg.mod",
                3,
                2,
                3,
                2,
                0.4,
                0.2,
                0.3,
                False,
                0,
                1,
                5,
                4,
                now_iso,
            )
        ]
    )
    gateway.analytics.insert_subsystems(
        [
            (
                repo,
                commit,
                "subsysdemo",
                "api_pkg",
                "Subsystem api_pkg covering 1 modules",
                1,
                '["pkg.mod"]',
                "[]",
                1,
                0,
                0,
                0,
                1,
                0.1,
                0.1,
                0,
                "low",
                now_iso,
            )
        ]
    )
    gateway.analytics.insert_subsystem_modules([(repo, commit, "subsysdemo", "pkg.mod", "api")])
    gateway.analytics.insert_test_catalog(
        [
            (
                "pkg/mod.py::test_func",
                10,
                "goid:demo/repo#python:function:pkg.mod.test_func",
                repo,
                commit,
                "pkg/mod.py",
                "pkg.mod.test_func",
                "test",
                "passed",
                1,
                "[]",
                False,
                False,
                now_iso,
            )
        ]
    )
    gateway.analytics.insert_test_coverage_edges(
        [
            (
                "pkg/mod.py::test_func",
                10,
                1,
                "goid:demo/repo#python:function:pkg.mod.func",
                repo,
                commit,
                "pkg/mod.py",
                "pkg.mod.func",
                2,
                2,
                1.0,
                "passed",
                now_iso,
            )
        ]
    )
    gateway.analytics.insert_typedness(
        [(repo, commit, "pkg/mod.py", 0, '{"params":1.0}', 0, False)]
    )
    gateway.analytics.insert_static_diagnostics([(repo, commit, "pkg/mod.py", 0, 0, 0, 0, False)])
    gateway.con.execute(
        """
        INSERT INTO analytics.hotspots VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("pkg/mod.py", 1, 1, 1, 1, 1.0, 0.1),
    )
    gateway.con.execute(
        """
        INSERT INTO core.ast_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("pkg/mod.py", 1, 1, 0, 1.0, 1, 0.1, now),
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.function_validation VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (repo, commit, "pkg/mod.py", "pkg.mod.func", "span_not_found", "Span 1-2", now),
    )
    return gateway


def _build_app(gateway: StorageGateway, db_path: Path, *, repo: str, commit: str) -> FastAPI:
    """
    Construct a FastAPI app bound to the provided gateway connection.

    Returns
    -------
    FastAPI
        Configured application using the supplied backend.
    """
    cfg = ApiAppConfig(
        server=McpServerConfig(repo_root=db_path.parent, repo=repo, commit=commit), read_only=True
    )

    def _loader() -> ApiAppConfig:
        return cfg

    def _backend_factory(_: ApiAppConfig) -> BackendResource:
        backend = DuckDBBackend(con=gateway.con, repo=repo, commit=commit)
        return BackendResource(backend=backend, service=backend.service, close=lambda: None)

    return create_app(config_loader=_loader, backend_factory=_backend_factory)


def test_fastapi_endpoints_smoke(tmp_path: Path) -> None:
    """Seed a DuckDB and verify key FastAPI endpoints return data."""
    repo = "demo/repo"
    commit = "deadbeef"
    db_path = tmp_path / "codeintel.duckdb"
    gateway = _seed_db(db_path, repo=repo, commit=commit)
    app = _build_app(gateway, db_path, repo=repo, commit=commit)
    with TestClient(app) as client:
        _assert_function_summary(client)
        _assert_callgraph(client)
        _assert_tests_for_function(client)
        _assert_file_summary(client)
        _assert_architecture_endpoints(client)
        _assert_subsystem_filters(client)
        _assert_ide_hints(client)
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
    resp = client.get(
        "/function/callgraph", params={"goid_h128": 1, "direction": "both", "limit": 10}
    )
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
    _assert_dataset_listing(client)
    _assert_function_metrics_dataset(client)
    _assert_function_validation_dataset(client)


def _assert_dataset_listing(client: TestClient) -> None:
    resp = client.get("/datasets")
    datasets = _json_or_fail(resp)
    if not isinstance(datasets, list):
        pytest.fail("Datasets payload was not a list")
    dataset_rows = [cast("dict[str, object]", ds) for ds in datasets if isinstance(ds, dict)]
    if not dataset_rows:
        pytest.fail("Dataset rows were not objects")
    if not any(ds.get("name") == "function_metrics" for ds in dataset_rows):
        pytest.fail("function_metrics dataset missing from list")
    if not any(ds.get("name") == "function_validation" for ds in dataset_rows):
        pytest.fail("function_validation dataset missing from list")


def _assert_function_metrics_dataset(client: TestClient) -> None:
    resp = client.get("/datasets/function_metrics", params={"limit": 5})
    rows_payload = _expect_dict(_json_or_fail(resp))
    rows = cast("list[dict[str, Any]]", rows_payload["rows"])
    if not rows:
        pytest.fail("Expected dataset rows for function_metrics")
    if rows[0]["qualname"] != "pkg.mod.func":
        pytest.fail("Unexpected qualname in dataset rows")


def _assert_function_validation_dataset(client: TestClient) -> None:
    max_expected_limit = 2
    resp = client.get("/datasets/function_validation", params={"limit": 5})
    fv_payload = _expect_dict(_json_or_fail(resp))
    fv_rows = cast("list[dict[str, Any]]", fv_payload["rows"])
    if not fv_rows:
        pytest.fail("Expected dataset rows for function_validation")
    if fv_rows[0].get("issue") != "span_not_found":
        pytest.fail("Unexpected issue in function_validation rows")

    resp = client.get(
        "/datasets/function_validation",
        params={"limit": max_expected_limit, "offset": -5},
    )
    clamped = _expect_dict(_json_or_fail(resp))
    meta = cast("dict[str, Any]", clamped.get("meta", {}))
    messages = cast("list[dict[str, Any]]", meta.get("messages", []))
    codes = {m.get("code") for m in messages}
    if "offset_invalid" not in codes:
        pytest.fail(f"Expected offset_invalid message codes; got {codes}")
    if clamped.get("offset") != 0:
        pytest.fail("Expected offset to clamp to 0")
    if clamped.get("limit") != max_expected_limit:
        pytest.fail("Expected limit to remain at requested value when within max")


def _assert_architecture_endpoints(client: TestClient) -> None:
    expected_call_fan_out = 3
    expected_import_fan_in = 3
    func_resp = client.get("/architecture/function", params={"goid_h128": 1})
    func_payload = _expect_dict(_json_or_fail(func_resp))
    if not func_payload.get("found"):
        pytest.fail("Function architecture should be found")
    func_arch = cast("dict[str, Any]", func_payload["architecture"])
    if func_arch.get("call_fan_out") != expected_call_fan_out:
        pytest.fail("Unexpected call_fan_out in function architecture")

    mod_resp = client.get("/architecture/module", params={"module": "pkg.mod"})
    mod_payload = _expect_dict(_json_or_fail(mod_resp))
    if not mod_payload.get("found"):
        pytest.fail("Module architecture should be found")
    mod_arch = cast("dict[str, Any]", mod_payload["architecture"])
    if mod_arch.get("import_fan_in") != expected_import_fan_in:
        pytest.fail("Unexpected import_fan_in in module architecture")

    _assert_subsystem_endpoints(client)


def _assert_subsystem_endpoints(client: TestClient) -> None:
    subs_resp = client.get("/architecture/subsystems")
    subs_payload = _expect_dict(_json_or_fail(subs_resp))
    subsystems = cast("list[dict[str, Any]]", subs_payload.get("subsystems", []))
    if not subsystems:
        pytest.fail("Expected subsystem summaries")
    if subsystems[0].get("subsystem_id") != "subsysdemo":
        pytest.fail("Unexpected subsystem id")

    memberships_resp = client.get("/architecture/module-subsystems", params={"module": "pkg.mod"})
    membership_payload = _expect_dict(_json_or_fail(memberships_resp))
    memberships = cast("list[dict[str, Any]]", membership_payload.get("memberships", []))
    if not memberships:
        pytest.fail("Expected subsystem memberships")
    if memberships[0].get("subsystem_id") != "subsysdemo":
        pytest.fail("Unexpected subsystem mapping")

    detail_resp = client.get("/architecture/subsystem", params={"subsystem_id": "subsysdemo"})
    detail_payload = _expect_dict(_json_or_fail(detail_resp))
    if not detail_payload.get("found"):
        pytest.fail("Subsystem detail should be found")
    modules = cast("list[dict[str, Any]]", detail_payload.get("modules", []))
    if not modules or modules[0].get("module") != "pkg.mod":
        pytest.fail("Subsystem detail modules missing or incorrect")


def _assert_ide_hints(client: TestClient) -> None:
    resp = client.get("/ide/hints", params={"rel_path": "pkg/mod.py"})
    hints_payload = _expect_dict(_json_or_fail(resp))
    hints = cast("list[dict[str, Any]]", hints_payload.get("hints", []))
    if not hints_payload.get("found"):
        pytest.fail("IDE hints should be found")
    if not hints:
        pytest.fail("Expected IDE hint rows")
    first_hint = hints[0]
    if first_hint.get("subsystem_name") != "api_pkg":
        pytest.fail("IDE hint subsystem_name mismatch")


def _assert_subsystem_filters(client: TestClient) -> None:
    resp = client.get("/architecture/subsystems", params={"role": "api", "q": "api"})
    payload = _expect_dict(_json_or_fail(resp))
    subs = cast("list[dict[str, Any]]", payload.get("subsystems", []))
    if not subs:
        pytest.fail("Expected subsystem list with filters")
    if subs[0].get("subsystem_id") != "subsysdemo":
        pytest.fail("Filtered subsystem mismatch")


def test_http_backend_against_fastapi(tmp_path: Path) -> None:
    """HttpBackend exercises FastAPI app via ASGI transport."""
    repo = "demo/repo"
    commit = "deadbeef"
    db_path = tmp_path / "codeintel.duckdb"
    gateway = _seed_db(db_path, repo=repo, commit=commit)
    app = _build_app(gateway, db_path, repo=repo, commit=commit)
    with TestClient(app):
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        )
        backend = HttpBackend(
            base_url="http://testserver",
            repo=repo,
            commit=commit,
            timeout=1.0,
            limits=BackendLimits(default_limit=5, max_rows_per_call=10),
            client=client,
        )
        try:
            _assert_http_backend_round_trip(backend)
        finally:
            anyio.run(client.aclose)


def _assert_http_backend_round_trip(backend: HttpBackend) -> None:
    _assert_http_backend_metadata(backend)
    _assert_http_backend_architecture(backend)
    _assert_http_backend_datasets(backend)


def _assert_http_backend_metadata(backend: HttpBackend) -> None:
    summary = backend.get_function_summary(
        urn="goid:demo/repo#python:function:pkg.mod.func",
    )
    if not summary.found or summary.summary is None:
        pytest.fail("Expected function summary")
    if summary.summary["qualname"] != "pkg.mod.func":
        pytest.fail("Unexpected qualname from HttpBackend summary")

    neighbors = backend.get_callgraph_neighbors(goid_h128=1, direction="both", limit=5)
    if not neighbors.outgoing:
        pytest.fail("Expected outgoing neighbors")

    tests_payload = backend.get_tests_for_function(goid_h128=1, limit=5)
    if not tests_payload.tests:
        pytest.fail("Expected tests via HttpBackend")


def _assert_http_backend_architecture(backend: HttpBackend) -> None:
    func_arch = backend.get_function_architecture(goid_h128=1)
    if not func_arch.found or func_arch.architecture is None:
        pytest.fail("Function architecture missing")

    mod_arch = backend.get_module_architecture(module="pkg.mod")
    if not mod_arch.found or mod_arch.architecture is None:
        pytest.fail("Module architecture missing")

    subsystems = backend.list_subsystems(limit=5)
    if not subsystems.subsystems:
        pytest.fail("Expected subsystem summaries via HttpBackend")

    memberships = backend.get_module_subsystems(module="pkg.mod")
    if not memberships.found or not memberships.memberships:
        pytest.fail("Expected subsystem mappings")

    detail = backend.get_subsystem_modules(subsystem_id="subsysdemo")
    if not detail.found or detail.subsystem is None:
        pytest.fail("Expected subsystem detail via HttpBackend")
    if not detail.modules:
        pytest.fail("Expected subsystem module rows via HttpBackend")

    hints = backend.get_file_hints(rel_path="pkg/mod.py")
    if not hints.found or not hints.hints:
        pytest.fail("Expected IDE hints via HttpBackend")


def _assert_http_backend_datasets(backend: HttpBackend) -> None:
    datasets = backend.list_datasets()
    if not any(ds.name == "function_metrics" for ds in datasets):
        pytest.fail("Expected function_metrics dataset in HttpBackend list")

    rows = backend.read_dataset_rows(dataset_name="function_metrics", limit=5)
    if not rows.rows:
        pytest.fail("Expected dataset rows via HttpBackend")


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
