"""Tests for semantic HTTP routes served by create_serving_app()."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
from fastapi import status
from fastapi.testclient import TestClient

from codeintel.config.primitives import BuildPaths
from codeintel.serving.http.app import create_serving_app
from codeintel.serving.settings import ServingSettings
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.hamilton_harness_artifacts import HarnessArtifacts

if TYPE_CHECKING:
    from pathlib import Path


def _make_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA docs")
    con.execute("CREATE TABLE docs.v_demo (id INTEGER, label VARCHAR)")
    con.execute("INSERT INTO docs.v_demo VALUES (1, 'one'), (2, 'two'), (3, 'three')")
    con.close()


def _write_serving_artifacts(
    tmp_path: Path,
    *,
    registry_path: Path,
    manifest_path: Path,
    buildspec_path: Path,
) -> None:
    artifacts = HarnessArtifacts(
        repo_root=tmp_path,
        paths=BuildPaths.from_explicit(build_dir=tmp_path / "build"),
    )
    artifacts.write_semantic_registry(
        path=registry_path,
        views=[
            {
                "id": "demo.view",
                "kind": "view",
                "table_key": "docs.v_demo",
                "entity": "demo",
                "grain": "per_row",
                "description": "Demo view",
                "primary_key": ["id"],
                "columns": ["id", "label"],
                "joins": [],
                "defaults": {"limit": 200, "order_by": ["id"]},
                "sensitivity": "internal",
                "deprecated": False,
                "replaced_by": None,
            }
        ],
    )
    artifacts.write_schema_manifest(
        path=manifest_path,
        tables=[
            {
                "schema": "docs",
                "name": "v_demo",
                "table_key": "docs.v_demo",
                "primary_key": ["id"],
                "indexes": [],
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "label", "type": "VARCHAR", "nullable": True},
                ],
            }
        ],
    )
    artifacts.write_buildspec(
        path=buildspec_path,
        datasets=[{"table_key": "docs.v_demo", "schema_hash": "schema_v_demo"}],
    )


def _write_pointer(
    path: Path,
    *,
    db_path: Path,
    registry_path: Path,
    manifest_path: Path,
    buildspec_path: Path,
) -> None:
    pointer = {
        "db_path": str(db_path),
        "semantic_registry_path": str(registry_path),
        "schema_manifest_path": str(manifest_path),
        "buildspec_path": str(buildspec_path),
        "repo": "demo/repo",
        "commit": "deadbeef",
        "run_id": "run-1",
        "published_at": datetime.now(tz=UTC).isoformat(),
        "semantic_layer_version": "v123",
    }
    path.write_text(json.dumps(pointer, indent=2, sort_keys=True), encoding="utf-8")


def test_semantic_routes_end_to_end(tmp_path: Path) -> None:
    """Serve semantic endpoints from a snapshot and query results."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True, exist_ok=True)

    db_path = tmp_path / "codeintel.duckdb"
    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    buildspec_path = tmp_path / "buildspec.json"
    _make_db(db_path)
    _write_serving_artifacts(
        tmp_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )
    _write_pointer(
        serve_dir / "current.json",
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )

    settings = ServingSettings(serve_dir=serve_dir, pool_size=1, poll_interval_s=0.01)
    app = create_serving_app(settings=settings, mount_mcp=False)

    with TestClient(app) as client:
        views = client.get("/v1/semantic/views")
        expect_equal(views.status_code, status.HTTP_200_OK)
        expect_true(any(v["id"] == "demo.view" for v in views.json()["views"]))

        desc = client.get("/v1/semantic/views/demo.view")
        expect_equal(desc.status_code, status.HTTP_200_OK)
        expect_equal(desc.json()["table_key"], "docs.v_demo")

        query = client.post(
            "/v1/semantic/query",
            json={
                "view_id": "demo.view",
                "filters": [{"column": "id", "op": "gte", "value": 2}],
                "order_by": ["-id"],
                "limit": 10,
                "offset": 0,
            },
        )
        expect_equal(query.status_code, status.HTTP_200_OK)
        rows = query.json()["rows"]
        expect_equal([row["id"] for row in rows], [3, 2])


def test_semantic_route_invalid_filter_returns_400(tmp_path: Path) -> None:
    """Bad filters map to 400 rather than a server error."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True, exist_ok=True)

    db_path = tmp_path / "codeintel.duckdb"
    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    buildspec_path = tmp_path / "buildspec.json"
    _make_db(db_path)
    _write_serving_artifacts(
        tmp_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )
    _write_pointer(
        serve_dir / "current.json",
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )

    settings = ServingSettings(serve_dir=serve_dir, pool_size=1, poll_interval_s=0.01)
    app = create_serving_app(settings=settings, mount_mcp=False)

    with TestClient(app) as client:
        query = client.post(
            "/v1/semantic/query",
            json={
                "view_id": "demo.view",
                "filters": [{"column": "nope", "op": "eq", "value": 1}],
                "limit": 10,
                "offset": 0,
            },
        )
        expect_equal(query.status_code, status.HTTP_400_BAD_REQUEST)


def test_semantic_routes_support_correlation_id(tmp_path: Path) -> None:
    """All semantic routes include correlation IDs."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True, exist_ok=True)

    db_path = tmp_path / "codeintel.duckdb"
    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    buildspec_path = tmp_path / "buildspec.json"
    _make_db(db_path)
    _write_serving_artifacts(
        tmp_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )
    _write_pointer(
        serve_dir / "current.json",
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )

    settings = ServingSettings(serve_dir=serve_dir, pool_size=1, poll_interval_s=0.01)
    app = create_serving_app(settings=settings, mount_mcp=False)

    with TestClient(app) as client:
        correlation_id = "cid-test-123"
        views = client.get("/v1/semantic/views", headers={"X-Correlation-ID": correlation_id})
        expect_equal(views.status_code, status.HTTP_200_OK)
        expect_equal(views.headers.get("X-Correlation-ID"), correlation_id)

        missing = client.get(
            "/v1/semantic/views/nope.view", headers={"X-Correlation-ID": correlation_id}
        )
        expect_equal(missing.status_code, status.HTTP_404_NOT_FOUND)
        payload = missing.json()
        expect_equal(payload.get("correlation_id"), correlation_id)


def test_semantic_routes_support_optional_api_key(tmp_path: Path) -> None:
    """When an API key is configured, routes require it."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True, exist_ok=True)

    db_path = tmp_path / "codeintel.duckdb"
    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    buildspec_path = tmp_path / "buildspec.json"
    _make_db(db_path)
    _write_serving_artifacts(
        tmp_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )
    _write_pointer(
        serve_dir / "current.json",
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )

    settings = ServingSettings(
        serve_dir=serve_dir,
        pool_size=1,
        poll_interval_s=0.01,
        api_key="secret-key",
    )
    app = create_serving_app(settings=settings, mount_mcp=False)

    with TestClient(app) as client:
        denied = client.get("/v1/semantic/views")
        expect_equal(denied.status_code, status.HTTP_401_UNAUTHORIZED)

        ok = client.get("/v1/semantic/views", headers={"X-API-Key": "secret-key"})
        expect_equal(ok.status_code, status.HTTP_200_OK)
