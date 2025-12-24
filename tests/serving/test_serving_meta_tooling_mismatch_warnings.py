"""Tests for runtime vs snapshot tooling mismatch warnings in serving_meta."""

from __future__ import annotations

import json
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
from tests._helpers.assertions import assert_target_ok
from tests._helpers.assertions.expectation_assertions import expect_true
from tests._helpers.harnesses.serving_harness import ServingTargetHarness
from tests._helpers.mcp_payloads import extract_payload

if TYPE_CHECKING:
    from pathlib import Path


def _publish_serving_snapshot(serving_target_harness: ServingTargetHarness) -> Path:
    records = serving_target_harness.run_targets()
    assert_target_ok(records["serving_artifacts"])
    serving_target_harness.publish_snapshot(run_id="run-1")
    serve_dir = serving_target_harness.harness.ctx.build_paths.build_dir / "serving"
    return serve_dir / "current.json"


@pytest.mark.anyio
async def test_serving_meta_includes_tool_version_mismatch_warning(
    serving_target_harness: ServingTargetHarness,
) -> None:
    """Return mismatch warnings when snapshot tool versions differ from runtime."""
    try:
        runtime_sqlglot = get_package_version("sqlglot")
    except PackageNotFoundError:
        pytest.skip("sqlglot not installed in runtime environment")

    pointer_path = _publish_serving_snapshot(serving_target_harness)
    serve_dir = pointer_path.parent

    (serve_dir / "environment.json").write_text(
        json.dumps({"tools": {"sqlglot": "0.0.0"}}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        settings = ServingSettings(
            serve_dir=serve_dir,
            hot_swap=False,
            pool_size=1,
            poll_interval_s=0.01,
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            payload = extract_payload(await client.call_tool("serving_meta", {}))
            warnings = payload.get("warnings")
            expected = f"tool-version-mismatch: sqlglot snapshot=0.0.0 runtime={runtime_sqlglot}"
            if not isinstance(warnings, list):
                pytest.fail("Expected serving_meta.warnings to be a list")
            expect_true(expected in warnings, message="Expected mismatch warning for sqlglot")
    finally:
        await manager.stop()
