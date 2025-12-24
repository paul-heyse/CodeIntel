"""Tests for runtime vs snapshot tooling mismatch warnings in serving_meta."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_package_version

import pytest

from tests._helpers.assertions.expectation_assertions import expect_true
from tests._helpers.harnesses.serving_app import ServingAppHarness
from tests._helpers.mcp_payloads import extract_payload
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory


@pytest.mark.anyio
async def test_serving_meta_includes_tool_version_mismatch_warning(
    serving_snapshot_factory: ServingSnapshotFactory,
) -> None:
    """Return mismatch warnings when snapshot tool versions differ from runtime."""
    try:
        runtime_sqlglot = get_package_version("sqlglot")
    except PackageNotFoundError:
        pytest.skip("sqlglot not installed in runtime environment")

    snapshot = serving_snapshot_factory.demo_snapshot()
    serve_dir = snapshot.serve_dir

    (serve_dir / "environment.json").write_text(
        json.dumps({"tools": {"sqlglot": "0.0.0"}}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    harness = ServingAppHarness.from_snapshot(snapshot)
    async with harness.mcp_client(settings_overrides={"hot_swap": False}) as client:
        payload = extract_payload(await client.call_tool("serving_meta", {}))
        warnings = payload.get("warnings")
        expected = f"tool-version-mismatch: sqlglot snapshot=0.0.0 runtime={runtime_sqlglot}"
        if not isinstance(warnings, list):
            pytest.fail("Expected serving_meta.warnings to be a list")
        expect_true(expected in warnings, message="Expected mismatch warning for sqlglot")
