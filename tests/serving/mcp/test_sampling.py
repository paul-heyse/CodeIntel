"""Tests for FastMCP sampling integration (ctx.sample) on the serving surface."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_package_version
from typing import TYPE_CHECKING

import pytest

from tests._helpers.harnesses.serving_app import ServingAppHarness, ServingSettingsOverrides
from tests._helpers.mcp_payloads import extract_payload
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory

if TYPE_CHECKING:
    from mcp.types import CreateMessageRequestParams, SamplingMessage


def _runtime_version(name: str) -> str:
    try:
        return get_package_version(name)
    except PackageNotFoundError:
        return "not-installed"


@pytest.mark.anyio
async def test_mcp_sampling_opt_in_adds_summary(
    serving_snapshot_factory: ServingSnapshotFactory,
) -> None:
    """Include a summary only when sampling is enabled and supported."""
    snapshot = serving_snapshot_factory.demo_snapshot(row_count=30)
    harness = ServingAppHarness.from_snapshot(snapshot)
    settings_overrides: ServingSettingsOverrides = {
        "hot_swap": False,
        "result_engine": "polars",
        "schema_enforcement": "strict",
        "mcp_mask_errors": False,
        "mcp_enable_sampling": True,
        "mcp_sample_threshold": 1,
    }

    def sampling_handler(
        _messages: list[SamplingMessage],
        _params: CreateMessageRequestParams,
        _context: object,
    ) -> str:
        return f"summary(runtime_sqlglot={_runtime_version('sqlglot')})"

    async with harness.mcp_client(
        settings_overrides=settings_overrides,
        client_kwargs={"sampling_handler": sampling_handler},
    ) as client:
        payload = extract_payload(
            await client.call_tool(
                "semantic_query",
                {"request": {"view_id": "demo.view"}},
            )
        )
        summary = payload.get("summary")
        if not isinstance(summary, str) or "summary(" not in summary:
            pytest.fail("Expected semantic_query to include sampling summary when enabled")


@pytest.mark.anyio
async def test_mcp_sampling_disabled_does_not_sample(
    serving_snapshot_factory: ServingSnapshotFactory,
) -> None:
    """Avoid calling ctx.sample when server-side sampling is disabled."""
    snapshot = serving_snapshot_factory.demo_snapshot(row_count=30)
    harness = ServingAppHarness.from_snapshot(snapshot)
    settings_overrides: ServingSettingsOverrides = {
        "hot_swap": False,
        "result_engine": "polars",
        "schema_enforcement": "strict",
        "mcp_mask_errors": False,
        "mcp_enable_sampling": False,
    }

    def sampling_handler(
        _messages: list[SamplingMessage],
        _params: CreateMessageRequestParams,
        _context: object,
    ) -> str:
        return "summary(should_not_be_used)"

    async with harness.mcp_client(
        settings_overrides=settings_overrides,
        client_kwargs={"sampling_handler": sampling_handler},
    ) as client:
        payload = extract_payload(
            await client.call_tool(
                "semantic_query",
                {"request": {"view_id": "demo.view"}},
            )
        )
        if payload.get("summary") is not None:
            pytest.fail("Expected semantic_query to omit summary when sampling is disabled")
