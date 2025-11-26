"""Integration smoke tests for FastAPI endpoints against a seeded DuckDB."""

from __future__ import annotations

from http import HTTPStatus
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
from codeintel.storage.gateway import StorageGateway
from tests._helpers.architecture import open_seeded_architecture_gateway


def _seed_db(db_path: Path, *, repo: str, commit: str) -> StorageGateway:
    """
    Open a DuckDB seeded with architecture data for FastAPI tests.

    Returns
    -------
    StorageGateway
        Gateway seeded with architecture data.
    """
    return open_seeded_architecture_gateway(
        repo=repo,
        commit=commit,
        db_path=db_path,
        strict_schema=True,
    )


def _build_app(gateway: StorageGateway, db_path: Path, *, repo: str, commit: str) -> FastAPI:
    """
    Construct a FastAPI app bound to the provided gateway connection.

    Returns
    -------
    FastAPI
        Configured application using the supplied backend.
    """
    cfg = ApiAppConfig(
        server=McpServerConfig(repo_root=db_path.parent, repo=repo, commit=commit, read_only=False),
        read_only=False,
    )

    def _loader() -> ApiAppConfig:
        return cfg

    def _backend_factory(_: ApiAppConfig, *, gateway: StorageGateway) -> BackendResource:
        backend = DuckDBBackend(gateway=gateway, repo=repo, commit=commit)
        return BackendResource(backend=backend, service=backend.service, close=lambda: None)

    return create_app(config_loader=_loader, backend_factory=_backend_factory, gateway=gateway)


def test_backend_registry_matches_gateway(tmp_path: Path) -> None:
    """Backend should mirror the gateway dataset registry."""
    repo = "demo/repo"
    commit = "deadbeef"
    gateway = _seed_db(tmp_path / "codeintel.duckdb", repo=repo, commit=commit)
    backend = DuckDBBackend(gateway=gateway, repo=repo, commit=commit)
    if backend.gateway.datasets.mapping != gateway.datasets.mapping:
        pytest.fail("Backend registry should match gateway mapping")
    gateway.close()


def test_fastapi_endpoints_smoke(architecture_gateway: StorageGateway, tmp_path: Path) -> None:
    """Seed a DuckDB and verify key FastAPI endpoints return data."""
    repo = "demo/repo"
    commit = "deadbeef"
    db_path = tmp_path / "codeintel.duckdb"
    app = _build_app(architecture_gateway, db_path, repo=repo, commit=commit)
    with TestClient(app) as client:
        _assert_function_summary(client)
        _assert_callgraph(client)
        _assert_tests_for_function(client)
        _assert_file_summary(client)
        _assert_architecture_endpoints(client)
        _assert_subsystem_filters(client)
        _assert_ide_hints(client)
        _assert_dataset_endpoints(client)


def test_health_reports_limits(architecture_gateway: StorageGateway, tmp_path: Path) -> None:
    """Health endpoint should surface backend limits for clients."""
    repo = "demo/repo"
    commit = "deadbeef"
    db_path = tmp_path / "health.duckdb"
    app = _build_app(architecture_gateway, db_path, repo=repo, commit=commit)
    with TestClient(app) as client:
        resp = client.get("/health")
        payload = _expect_dict(_json_or_fail(resp))
        limits = cast("dict[str, int] | None", payload.get("limits"))
        if limits is None:
            pytest.fail("Health payload missing limits")
        if limits.get("max_rows_per_call") != BackendLimits().max_rows_per_call:
            pytest.fail(f"Unexpected limits payload: {limits}")


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
            async def _close_client() -> None:
                await client.aclose()

            anyio.run(_close_client)


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


def test_callgraph_neighborhood_limits(tmp_path: Path) -> None:
    """Call graph neighborhood enforces limit validation and truncation metadata."""
    repo = "demo/repo"
    commit = "deadbeef"
    db_path = tmp_path / "codeintel.duckdb"
    gateway = _seed_db(db_path, repo=repo, commit=commit)
    app = _build_app(gateway, db_path, repo=repo, commit=commit)
    with TestClient(app) as client:
        bad = client.get(
            "/graph/call/neighborhood", params={"goid_h128": 1, "radius": 1, "max_nodes": -1}
        )
        if bad.status_code != HTTPStatus.BAD_REQUEST:
            pytest.fail(f"Expected 400 for invalid max_nodes, got {bad.status_code}")
        resp = client.get(
            "/graph/call/neighborhood", params={"goid_h128": 1, "radius": 1, "max_nodes": 0}
        )
        if resp.status_code != HTTPStatus.OK:
            pytest.fail(f"Expected 200 response, got {resp.status_code}")
        body = _expect_dict(_json_or_fail(resp))
        meta = body.get("meta")
        if not isinstance(meta, dict):
            pytest.fail("Expected meta object in response")
        applied_limit = meta.get("applied_limit")
        truncated = meta.get("truncated")
        if applied_limit != 0:
            pytest.fail(f"Expected applied_limit 0, got {applied_limit}")
        if truncated is not True:
            pytest.fail(f"Expected truncated True, got {truncated}")


def test_import_boundary_limits(tmp_path: Path) -> None:
    """Import boundary endpoint enforces limit validation and truncation metadata."""
    repo = "demo/repo"
    commit = "deadbeef"
    db_path = tmp_path / "codeintel.duckdb"
    gateway = _seed_db(db_path, repo=repo, commit=commit)
    app = _build_app(gateway, db_path, repo=repo, commit=commit)
    with TestClient(app) as client:
        bad = client.get("/graph/import/boundary", params={"subsystem_id": "sub1", "max_edges": -5})
        if bad.status_code != HTTPStatus.BAD_REQUEST:
            pytest.fail(f"Expected 400 for invalid max_edges, got {bad.status_code}")
        resp = client.get("/graph/import/boundary", params={"subsystem_id": "sub1", "max_edges": 0})
        if resp.status_code != HTTPStatus.OK:
            pytest.fail(f"Expected 200 response, got {resp.status_code}")
        body = _expect_dict(_json_or_fail(resp))
        meta = body.get("meta")
        if not isinstance(meta, dict):
            pytest.fail("Expected meta object in response")
        applied_limit = meta.get("applied_limit")
        truncated = meta.get("truncated")
        if applied_limit != 0:
            pytest.fail(f"Expected applied_limit 0, got {applied_limit}")
        if truncated is not True:
            pytest.fail(f"Expected truncated True, got {truncated}")


JsonPayload = dict[str, object] | list[object]


def _json_or_fail(resp: httpx.Response) -> JsonPayload:
    """
    Parse JSON payload or fail with a helpful message.

    Returns
    -------
    dict[str, object]
        Parsed JSON object payload when status is 200.
    """
    if resp.status_code != status.HTTP_200_OK:
        pytest.fail(f"Unexpected status {resp.status_code}: {resp.text}")
    data = resp.json()
    if not isinstance(data, (dict, list)):
        pytest.fail("Response payload is not JSON object/array")
    return data


def _expect_dict(payload: JsonPayload) -> dict[str, object]:
    """
    Validate that a JSON payload is a dict.

    Returns
    -------
    dict[str, object]
        Parsed JSON object payload.
    """
    if not isinstance(payload, dict):
        pytest.fail("Expected JSON object payload")
    return payload
