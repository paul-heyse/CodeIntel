"""Tests for canonical SQL fingerprint generation on the MCP serving surface."""

from __future__ import annotations

import pytest

from tests._helpers.harnesses.serving_app import ServingAppHarness, ServingSettingsOverrides
from tests._helpers.mcp_payloads import extract_payload
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory

_SHA256_HEX_LENGTH = 64


def _is_sha256_hex(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) != _SHA256_HEX_LENGTH:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


@pytest.mark.anyio
async def test_mcp_sql_fingerprint_is_stable_for_same_request(
    serving_snapshot_factory: ServingSnapshotFactory,
) -> None:
    """Return stable fingerprint for identical semantic_query inputs."""
    snapshot = serving_snapshot_factory.demo_snapshot()
    harness = ServingAppHarness.from_snapshot(snapshot)
    settings_overrides: ServingSettingsOverrides = {
        "hot_swap": False,
        "result_engine": "pandas",
        "schema_enforcement": "strict",
        "mcp_mask_errors": False,
    }
    async with harness.mcp_client(settings_overrides=settings_overrides) as client:
        args = {"request": {"view_id": "demo.view", "pagination": {"limit": 2, "offset": 0}}}
        first = extract_payload(await client.call_tool("semantic_query", args))
        second = extract_payload(await client.call_tool("semantic_query", args))

        fp1 = first.get("sql_fingerprint")
        fp2 = second.get("sql_fingerprint")
        if not _is_sha256_hex(fp1) or not _is_sha256_hex(fp2):
            pytest.fail("Expected semantic_query.sql_fingerprint to be a SHA256 hex digest")
        if fp1 != fp2:
            pytest.fail("Expected sql_fingerprint to be stable for identical requests")


@pytest.mark.anyio
async def test_mcp_sql_fingerprint_changes_when_limit_changes(
    serving_snapshot_factory: ServingSnapshotFactory,
) -> None:
    """Change fingerprint when compiled SQL changes (e.g., different LIMIT)."""
    snapshot = serving_snapshot_factory.demo_snapshot()
    harness = ServingAppHarness.from_snapshot(snapshot)
    settings_overrides: ServingSettingsOverrides = {
        "hot_swap": False,
        "result_engine": "pandas",
        "schema_enforcement": "strict",
        "mcp_mask_errors": False,
    }
    async with harness.mcp_client(settings_overrides=settings_overrides) as client:
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
