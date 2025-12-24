"""Tests for export/streaming HTTP endpoints."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import status
from fastapi.testclient import TestClient

from codeintel.config.primitives import BuildPaths
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.http.app import create_serving_app
from codeintel.serving.settings import ServingSettings
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_true,
)
from tests._helpers.assertions.http_responses import assert_problem_detail_response
from tests._helpers.gateway import GatewayFactory
from tests._helpers.hamilton_harness_artifacts import HarnessArtifacts
from tests._helpers.schemas import ensure_production_schemas

if TYPE_CHECKING:
    from pathlib import Path


def _make_db(db_path: Path) -> None:
    """Create test database with sample data."""
    gateway = GatewayFactory().file_backed(db_path).open()
    try:
        ensure_production_schemas(gateway.con)
        gateway.con.execute(
            "CREATE TABLE docs.v_export_test (id INTEGER, name VARCHAR, value DOUBLE)"
        )
        gateway.con.execute(
            """
            INSERT INTO docs.v_export_test VALUES
            (1, 'alpha', 1.1),
            (2, 'beta', 2.2),
            (3, 'gamma', 3.3),
            (4, 'delta', 4.4),
            (5, 'epsilon', 5.5)
            """
        )
    finally:
        gateway.close()


def _write_pointer(
    path: Path,
    *,
    db_path: Path,
    registry_path: Path,
    manifest_path: Path,
    buildspec_path: Path,
) -> None:
    """Write serving snapshot pointer."""
    pointer = ServingSnapshotPointer(
        db_path=db_path,
        semantic_registry_path=registry_path,
        schema_manifest_path=manifest_path,
        buildspec_path=buildspec_path,
        repo="test/export",
        commit="abc123",
        run_id="run-export-1",
        published_at=datetime.now(tz=UTC),
        semantic_layer_version="v100",
    )
    path.write_text(pointer.to_json(), encoding="utf-8")


def _setup_serving_env(tmp_path: Path) -> ServingSettings:
    """Create serving environment.

    Returns
    -------
    ServingSettings
        Configured settings for test serving.
    """
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True, exist_ok=True)

    db_path = tmp_path / "export_test.duckdb"
    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    buildspec_path = tmp_path / "buildspec.json"

    _make_db(db_path)
    artifacts = HarnessArtifacts(
        repo_root=tmp_path,
        paths=BuildPaths.from_explicit(build_dir=tmp_path / "build"),
    )
    artifacts.write_semantic_registry(
        path=registry_path,
        views=[
            {
                "id": "export.test",
                "kind": "view",
                "table_key": "docs.v_export_test",
                "entity": "test",
                "grain": "per_row",
                "description": "Export test view",
                "primary_key": ["id"],
                "columns": ["id", "name", "value"],
                "joins": [],
                "defaults": {"limit": 200, "order_by": ["id"]},
                "sensitivity": "internal",
            }
        ],
    )
    artifacts.write_schema_manifest(
        path=manifest_path,
        tables=[
            {
                "schema": "docs",
                "name": "v_export_test",
                "table_key": "docs.v_export_test",
                "primary_key": ["id"],
                "indexes": [],
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "name", "type": "VARCHAR", "nullable": True},
                    {"name": "value", "type": "DOUBLE", "nullable": True},
                ],
            }
        ],
    )
    artifacts.write_buildspec(
        path=buildspec_path,
        datasets=[{"table_key": "docs.v_export_test", "schema_hash": "schema_export_test"}],
    )
    _write_pointer(
        serve_dir / "current.json",
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )

    return ServingSettings(serve_dir=serve_dir, pool_size=1, poll_interval_s=0.01)


def test_export_json_format(tmp_path: Path) -> None:
    """Export with format=json returns JSON response with rows."""
    settings = _setup_serving_env(tmp_path)
    app = create_serving_app(settings=settings, mount_mcp=False)

    with TestClient(app) as client:
        response = client.post(
            "/v1/export/semantic/export.test",
            json={"view_id": "export.test", "format": "json", "limit": 100},
        )

        expect_equal(response.status_code, status.HTTP_200_OK)
        data = response.json()
        expect_equal(data["view_id"], "export.test")
        expect_equal(data["count"], 5)
        expect_equal(len(data["rows"]), 5)


def test_export_jsonl_format(tmp_path: Path) -> None:
    """Export with format=jsonl returns newline-delimited JSON."""
    settings = _setup_serving_env(tmp_path)
    app = create_serving_app(settings=settings, mount_mcp=False)

    with TestClient(app) as client:
        response = client.post(
            "/v1/export/semantic/export.test",
            json={"view_id": "export.test", "format": "jsonl", "limit": 100},
        )

        expect_equal(response.status_code, status.HTTP_200_OK)
        expect_in("application/x-ndjson", response.headers.get("content-type", ""))

        lines = response.text.strip().split("\n")
        expect_equal(len(lines), 5)

        first_row = json.loads(lines[0])
        expect_equal(first_row["id"], 1)
        expect_equal(first_row["name"], "alpha")


def test_export_jsonl_content_disposition(tmp_path: Path) -> None:
    """JSONL export should include content-disposition header."""
    settings = _setup_serving_env(tmp_path)
    app = create_serving_app(settings=settings, mount_mcp=False)

    with TestClient(app) as client:
        response = client.post(
            "/v1/export/semantic/export.test",
            json={"view_id": "export.test", "format": "jsonl"},
        )

        expect_equal(response.status_code, status.HTTP_200_OK)
        content_disposition = response.headers.get("content-disposition", "")
        expect_in("export.test.jsonl", content_disposition)


def test_export_with_filters(tmp_path: Path) -> None:
    """Export should support filtering."""
    settings = _setup_serving_env(tmp_path)
    app = create_serving_app(settings=settings, mount_mcp=False)

    with TestClient(app) as client:
        response = client.post(
            "/v1/export/semantic/export.test",
            json={
                "view_id": "export.test",
                "format": "jsonl",
                "filters": [{"column": "id", "op": "gte", "value": 3}],
            },
        )

        expect_equal(response.status_code, status.HTTP_200_OK)

        lines = response.text.strip().split("\n")
        expect_equal(len(lines), 3)


def test_export_with_select_columns(tmp_path: Path) -> None:
    """Export should support column selection."""
    settings = _setup_serving_env(tmp_path)
    app = create_serving_app(settings=settings, mount_mcp=False)

    with TestClient(app) as client:
        response = client.post(
            "/v1/export/semantic/export.test",
            json={
                "view_id": "export.test",
                "format": "jsonl",
                "select": ["id", "name"],
            },
        )

        expect_equal(response.status_code, status.HTTP_200_OK)

        first_row = json.loads(response.text.strip().split("\n")[0])
        expect_true("id" in first_row)
        expect_true("name" in first_row)
        expect_true("value" not in first_row)


def test_export_with_order_by(tmp_path: Path) -> None:
    """Export should support ordering."""
    settings = _setup_serving_env(tmp_path)
    app = create_serving_app(settings=settings, mount_mcp=False)

    with TestClient(app) as client:
        response = client.post(
            "/v1/export/semantic/export.test",
            json={
                "view_id": "export.test",
                "format": "jsonl",
                "order_by": ["-id"],
            },
        )

        expect_equal(response.status_code, status.HTTP_200_OK)

        lines = response.text.strip().split("\n")
        first_row = json.loads(lines[0])
        last_row = json.loads(lines[-1])
        expect_equal(first_row["id"], 5)
        expect_equal(last_row["id"], 1)


def test_export_view_not_found(tmp_path: Path) -> None:
    """Export with unknown view returns 404."""
    settings = _setup_serving_env(tmp_path)
    app = create_serving_app(settings=settings, mount_mcp=False)

    with TestClient(app) as client:
        response = client.post(
            "/v1/export/semantic/nonexistent.view",
            json={"view_id": "nonexistent.view", "format": "json"},
        )

        assert_problem_detail_response(response, status_code=status.HTTP_404_NOT_FOUND)
        body = response.json()
        expect_in("not found", body.get("title", "").lower())


def test_export_invalid_filter_column(tmp_path: Path) -> None:
    """Export with invalid filter column returns 400."""
    settings = _setup_serving_env(tmp_path)
    app = create_serving_app(settings=settings, mount_mcp=False)

    with TestClient(app) as client:
        response = client.post(
            "/v1/export/semantic/export.test",
            json={
                "view_id": "export.test",
                "format": "json",
                "filters": [{"column": "nonexistent", "op": "eq", "value": 1}],
            },
        )

        assert_problem_detail_response(response, status_code=status.HTTP_400_BAD_REQUEST)


def test_export_respects_api_key(tmp_path: Path) -> None:
    """Export endpoints should require API key when configured."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True, exist_ok=True)

    db_path = tmp_path / "export_test.duckdb"
    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    buildspec_path = tmp_path / "buildspec.json"

    _make_db(db_path)
    artifacts = HarnessArtifacts(
        repo_root=tmp_path,
        paths=BuildPaths.from_explicit(build_dir=tmp_path / "build"),
    )
    artifacts.write_semantic_registry(
        path=registry_path,
        views=[
            {
                "id": "export.test",
                "kind": "view",
                "table_key": "docs.v_export_test",
                "entity": "test",
                "grain": "per_row",
                "description": "Export test view",
                "primary_key": ["id"],
                "columns": ["id", "name", "value"],
                "joins": [],
                "defaults": {"limit": 200, "order_by": ["id"]},
                "sensitivity": "internal",
            }
        ],
    )
    artifacts.write_schema_manifest(
        path=manifest_path,
        tables=[
            {
                "schema": "docs",
                "name": "v_export_test",
                "table_key": "docs.v_export_test",
                "primary_key": ["id"],
                "indexes": [],
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "name", "type": "VARCHAR", "nullable": True},
                    {"name": "value", "type": "DOUBLE", "nullable": True},
                ],
            }
        ],
    )
    artifacts.write_buildspec(
        path=buildspec_path,
        datasets=[{"table_key": "docs.v_export_test", "schema_hash": "schema_export_test"}],
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
        api_key="export-secret",
    )
    app = create_serving_app(settings=settings, mount_mcp=False)

    with TestClient(app) as client:
        denied = client.post(
            "/v1/export/semantic/export.test",
            json={"view_id": "export.test", "format": "json"},
        )
        expect_equal(denied.status_code, status.HTTP_401_UNAUTHORIZED)

        allowed = client.post(
            "/v1/export/semantic/export.test",
            json={"view_id": "export.test", "format": "json"},
            headers={"X-API-Key": "export-secret"},
        )
        expect_equal(allowed.status_code, status.HTTP_200_OK)


def test_export_v1_route(tmp_path: Path) -> None:
    """Export endpoints are versioned under /v1."""
    settings = _setup_serving_env(tmp_path)
    app = create_serving_app(settings=settings, mount_mcp=False)

    with TestClient(app) as client:
        response = client.post(
            "/v1/export/semantic/export.test",
            json={"view_id": "export.test", "format": "json"},
        )

        expect_equal(response.status_code, status.HTTP_200_OK)
        expect_equal(response.json()["count"], 5)
