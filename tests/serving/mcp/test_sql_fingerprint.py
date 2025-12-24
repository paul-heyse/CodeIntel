"""Tests for canonical SQL fingerprint generation on the MCP serving surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastmcp.client import Client

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.mcp_payloads import extract_payload
from tests._helpers.serving_snapshots import setup_demo_snapshot

if TYPE_CHECKING:
    from pathlib import Path


_SHA256_HEX_LENGTH = 64


def _setup_demo_snapshot(tmp_path: Path) -> Path:
    snapshot = setup_demo_snapshot(tmp_path)
    return snapshot.pointer_path


def _is_sha256_hex(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) != _SHA256_HEX_LENGTH:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


@pytest.mark.anyio
async def test_mcp_sql_fingerprint_is_stable_for_same_request(tmp_path: Path) -> None:
    """Return stable fingerprint for identical semantic_query inputs."""
    pointer_path = _setup_demo_snapshot(tmp_path)

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
    pointer_path = _setup_demo_snapshot(tmp_path)

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
