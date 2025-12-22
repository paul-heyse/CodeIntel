"""Tests for canonical SQL fingerprint generation on the MCP serving surface."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
import pytest
from fastmcp.client import Client

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.mcp_payloads import extract_payload

if TYPE_CHECKING:
    from pathlib import Path


_SHA256_HEX_LENGTH = 64


def _make_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA docs")
    con.execute("CREATE TABLE docs.v_demo (id INTEGER, label VARCHAR)")
    con.execute("INSERT INTO docs.v_demo VALUES (1, 'one'), (2, 'two'), (3, 'three')")
    con.close()


def _write_registry(path: Path) -> None:
    registry = {
        "version": "v1",
        "views": [
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
    }
    path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")


def _write_schema_manifest(path: Path) -> None:
    manifest = {
        "version": "v1",
        "tables": [
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
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _write_buildspec(path: Path) -> None:
    buildspec = {
        "spec_version": 1,
        "targets": [],
        "datasets": [{"table_key": "docs.v_demo", "schema_hash": "schema_v_demo"}],
    }
    path.write_text(json.dumps(buildspec, indent=2, sort_keys=True), encoding="utf-8")


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


def _setup_test_snapshot(tmp_path: Path) -> Path:
    db_path = tmp_path / "codeintel.duckdb"
    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    buildspec_path = tmp_path / "buildspec.json"
    pointer_path = tmp_path / "current.json"

    _make_db(db_path)
    _write_registry(registry_path)
    _write_schema_manifest(manifest_path)
    _write_buildspec(buildspec_path)
    _write_pointer(
        pointer_path,
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )
    return pointer_path


def _is_sha256_hex(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) != _SHA256_HEX_LENGTH:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


@pytest.mark.anyio
async def test_mcp_sql_fingerprint_is_stable_for_same_request(tmp_path: Path) -> None:
    """Return stable fingerprint for identical semantic_query inputs."""
    pointer_path = _setup_test_snapshot(tmp_path)

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        settings = ServingSettings(
            serve_dir=tmp_path,
            hot_swap=False,
            pool_size=1,
            poll_interval_s=0.01,
            result_engine="pandas",
            schema_enforcement="strict",
            mcp_mask_errors=False,
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            args = {"request": {"view_id": "demo.view", "pagination": {"limit": 2, "offset": 0}}}
            first = extract_payload(await client.call_tool("semantic_query", args))
            second = extract_payload(await client.call_tool("semantic_query", args))

            fp1 = first.get("sql_fingerprint")
            fp2 = second.get("sql_fingerprint")
            if not _is_sha256_hex(fp1) or not _is_sha256_hex(fp2):
                pytest.fail("Expected semantic_query.sql_fingerprint to be a SHA256 hex digest")
            if fp1 != fp2:
                pytest.fail("Expected sql_fingerprint to be stable for identical requests")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_sql_fingerprint_changes_when_limit_changes(tmp_path: Path) -> None:
    """Change fingerprint when compiled SQL changes (e.g., different LIMIT)."""
    pointer_path = _setup_test_snapshot(tmp_path)

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        settings = ServingSettings(
            serve_dir=tmp_path,
            hot_swap=False,
            pool_size=1,
            poll_interval_s=0.01,
            result_engine="pandas",
            schema_enforcement="strict",
            mcp_mask_errors=False,
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            first = extract_payload(
                await client.call_tool(
                    "semantic_query",
                    {
                        "request": {
                            "view_id": "demo.view",
                            "pagination": {"limit": 2, "offset": 0},
                        }
                    },
                )
            )
            second = extract_payload(
                await client.call_tool(
                    "semantic_query",
                    {
                        "request": {
                            "view_id": "demo.view",
                            "pagination": {"limit": 3, "offset": 0},
                        }
                    },
                )
            )

            fp1 = first.get("sql_fingerprint")
            fp2 = second.get("sql_fingerprint")
            if not _is_sha256_hex(fp1) or not _is_sha256_hex(fp2):
                pytest.fail("Expected sql_fingerprint to be present for both queries")
            if fp1 == fp2:
                pytest.fail("Expected sql_fingerprint to differ when SQL changes")
    finally:
        await manager.stop()
