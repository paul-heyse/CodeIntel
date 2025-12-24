"""Tests for FastMCP sampling integration (ctx.sample) on the serving surface."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_package_version
from typing import TYPE_CHECKING

import pytest
from fastmcp.client import Client

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.mcp_payloads import extract_payload
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.types import CreateMessageRequestParams, SamplingMessage


def _setup_demo_snapshot(tmp_path: Path) -> Path:
    snapshot = ServingSnapshotFactory(tmp_path, serve_dir=tmp_path).demo_snapshot(row_count=30)
    return snapshot.pointer_path


def _runtime_version(name: str) -> str:
    try:
        return get_package_version(name)
    except PackageNotFoundError:
        return "not-installed"


@pytest.mark.anyio
async def test_mcp_sampling_opt_in_adds_summary(tmp_path: Path) -> None:
    """Include a summary only when sampling is enabled and supported."""
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
            mcp_enable_sampling=True,
            mcp_sample_threshold=1,
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        def sampling_handler(
            _messages: list[SamplingMessage],
            _params: CreateMessageRequestParams,
            _context: object,
        ) -> str:
            return f"summary(runtime_sqlglot={_runtime_version('sqlglot')})"

        async with Client(mcp, sampling_handler=sampling_handler) as client:
            payload = extract_payload(
                await client.call_tool(
                    "semantic_query",
                    {"request": {"view_id": "demo.view"}},
                )
            )
            summary = payload.get("summary")
            if not isinstance(summary, str) or "summary(" not in summary:
                pytest.fail("Expected semantic_query to include sampling summary when enabled")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_sampling_disabled_does_not_sample(tmp_path: Path) -> None:
    """Avoid calling ctx.sample when server-side sampling is disabled."""
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
            mcp_enable_sampling=False,
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        def sampling_handler(
            _messages: list[SamplingMessage],
            _params: CreateMessageRequestParams,
            _context: object,
        ) -> str:
            return "summary(should_not_be_used)"

        async with Client(mcp, sampling_handler=sampling_handler) as client:
            payload = extract_payload(
                await client.call_tool(
                    "semantic_query",
                    {"request": {"view_id": "demo.view"}},
                )
            )
            if payload.get("summary") is not None:
                pytest.fail("Expected semantic_query to omit summary when sampling is disabled")
    finally:
        await manager.stop()
