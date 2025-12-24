"""Tests for semantic HTTP routes served by create_serving_app()."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import status
from fastapi.testclient import TestClient

from codeintel.serving.http.app import create_serving_app
from codeintel.serving.settings import ServingSettings
from tests._helpers.assertions import assert_http_success
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory

if TYPE_CHECKING:
    from pathlib import Path


def test_semantic_routes_end_to_end(tmp_path: Path) -> None:
    """Serve semantic endpoints from a snapshot and query results."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True, exist_ok=True)
    ServingSnapshotFactory(tmp_path).demo_snapshot(pointer_path=serve_dir / "current.json")

    settings = ServingSettings(serve_dir=serve_dir, pool_size=1, poll_interval_s=0.01)
    app = create_serving_app(settings=settings, mount_mcp=False)

    with TestClient(app) as client:
        views = assert_http_success(client, "/v1/semantic/views")
        expect_true(any(v["id"] == "demo.view" for v in views["views"]))

        desc = assert_http_success(client, "/v1/semantic/views/demo.view")
        expect_equal(desc["table_key"], "docs.v_demo")

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
    ServingSnapshotFactory(tmp_path).demo_snapshot(pointer_path=serve_dir / "current.json")

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
    ServingSnapshotFactory(tmp_path).demo_snapshot(pointer_path=serve_dir / "current.json")

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
    ServingSnapshotFactory(tmp_path).demo_snapshot(pointer_path=serve_dir / "current.json")

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
