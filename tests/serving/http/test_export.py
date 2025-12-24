"""Tests for export/streaming HTTP endpoints."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fastapi import status

from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_true,
)
from tests._helpers.assertions.http_responses import assert_problem_detail_response
from tests._helpers.gateway import GatewayFactory
from tests._helpers.harnesses.serving_app import ServingAppHarness
from tests._helpers.schemas import ensure_production_schemas
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory, SnapshotArtifacts

if TYPE_CHECKING:
    from pathlib import Path

    from tests._helpers.serving_snapshot_factory import ServingSnapshot


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


def _export_views() -> list[dict[str, object]]:
    return [
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
    ]


def _export_tables() -> list[dict[str, object]]:
    return [
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
    ]


def _make_snapshot(factory: ServingSnapshotFactory) -> ServingSnapshot:
    artifacts = SnapshotArtifacts(
        views=_export_views(),
        tables=_export_tables(),
        db_setup=_make_db,
    )
    return factory.make_snapshot(artifacts=artifacts)


def _make_harness(factory: ServingSnapshotFactory) -> ServingAppHarness:
    snapshot = _make_snapshot(factory)
    return ServingAppHarness.from_snapshot(snapshot)


def test_export_json_format(serving_snapshot_factory: ServingSnapshotFactory) -> None:
    """Export with format=json returns JSON response with rows."""
    harness = _make_harness(serving_snapshot_factory)
    with harness.http_client(mount_mcp=False) as client:
        response = client.post(
            "/v1/export/semantic/export.test",
            json={"view_id": "export.test", "format": "json", "limit": 100},
        )

        expect_equal(response.status_code, status.HTTP_200_OK)
        data = response.json()
        expect_equal(data["view_id"], "export.test")
        expect_equal(data["count"], 5)
        expect_equal(len(data["rows"]), 5)


def test_export_jsonl_format(serving_snapshot_factory: ServingSnapshotFactory) -> None:
    """Export with format=jsonl returns newline-delimited JSON."""
    harness = _make_harness(serving_snapshot_factory)
    with harness.http_client(mount_mcp=False) as client:
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


def test_export_jsonl_content_disposition(serving_snapshot_factory: ServingSnapshotFactory) -> None:
    """JSONL export should include content-disposition header."""
    harness = _make_harness(serving_snapshot_factory)
    with harness.http_client(mount_mcp=False) as client:
        response = client.post(
            "/v1/export/semantic/export.test",
            json={"view_id": "export.test", "format": "jsonl"},
        )

        expect_equal(response.status_code, status.HTTP_200_OK)
        content_disposition = response.headers.get("content-disposition", "")
        expect_in("export.test.jsonl", content_disposition)


def test_export_with_filters(serving_snapshot_factory: ServingSnapshotFactory) -> None:
    """Export should support filtering."""
    harness = _make_harness(serving_snapshot_factory)
    with harness.http_client(mount_mcp=False) as client:
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


def test_export_with_select_columns(serving_snapshot_factory: ServingSnapshotFactory) -> None:
    """Export should support column selection."""
    harness = _make_harness(serving_snapshot_factory)
    with harness.http_client(mount_mcp=False) as client:
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


def test_export_with_order_by(serving_snapshot_factory: ServingSnapshotFactory) -> None:
    """Export should support ordering."""
    harness = _make_harness(serving_snapshot_factory)
    with harness.http_client(mount_mcp=False) as client:
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


def test_export_view_not_found(serving_snapshot_factory: ServingSnapshotFactory) -> None:
    """Export with unknown view returns 404."""
    harness = _make_harness(serving_snapshot_factory)
    with harness.http_client(mount_mcp=False) as client:
        response = client.post(
            "/v1/export/semantic/nonexistent.view",
            json={"view_id": "nonexistent.view", "format": "json"},
        )

        assert_problem_detail_response(response, status_code=status.HTTP_404_NOT_FOUND)
        body = response.json()
        expect_in("not found", body.get("title", "").lower())


def test_export_invalid_filter_column(serving_snapshot_factory: ServingSnapshotFactory) -> None:
    """Export with invalid filter column returns 400."""
    harness = _make_harness(serving_snapshot_factory)
    with harness.http_client(mount_mcp=False) as client:
        response = client.post(
            "/v1/export/semantic/export.test",
            json={
                "view_id": "export.test",
                "format": "json",
                "filters": [{"column": "nonexistent", "op": "eq", "value": 1}],
            },
        )

        assert_problem_detail_response(response, status_code=status.HTTP_400_BAD_REQUEST)


def test_export_respects_api_key(serving_snapshot_factory: ServingSnapshotFactory) -> None:
    """Export endpoints should require API key when configured."""
    harness = _make_harness(serving_snapshot_factory)
    with harness.http_client(
        mount_mcp=False,
        settings_overrides={"api_key": "export-secret"},
    ) as client:
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


def test_export_v1_route(serving_snapshot_factory: ServingSnapshotFactory) -> None:
    """Export endpoints are versioned under /v1."""
    harness = _make_harness(serving_snapshot_factory)
    with harness.http_client(mount_mcp=False) as client:
        response = client.post(
            "/v1/export/semantic/export.test",
            json={"view_id": "export.test", "format": "json"},
        )

        expect_equal(response.status_code, status.HTTP_200_OK)
        expect_equal(response.json()["count"], 5)
