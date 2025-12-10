"""Tests for AstExtractPlugin module wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.plugins.ast_extract import AstExtractPlugin
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.fakes.ingestion_plugins import (
    StepCallCapture,
    make_recording_adapter_factories,
    make_recording_step_factory,
)
from tests._helpers.ingestion import TargetContextConfig, build_target_context_for_plugin
from tests.ingestion.plugins._wiring import run_sync_plugin_wiring_scenario


def _make_plugin(
    capture: StepCallCapture,
    *,
    table_key: str = "core.ast_nodes",
    result: StepResult | None = None,
) -> AstExtractPlugin:
    storage_factory, discovery_factory = make_recording_adapter_factories(capture)
    step_factory = make_recording_step_factory(capture, table_key=table_key, result=result)
    return AstExtractPlugin(
        storage_adapter_factory=storage_factory,
        discovery_adapter_factory=discovery_factory,
        step_factory=step_factory,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("scenario", ["resources", "db_fallback", "gateway_failure"])
async def test_ast_extract_wiring_scenarios(tmp_path: Path, scenario: str) -> None:
    """Shared wiring coverage for AstExtractPlugin."""
    await run_sync_plugin_wiring_scenario(
        scenario,
        lambda capture: _make_plugin(capture),
        tmp_path,
        table_key="core.ast_nodes",
    )


@pytest.mark.anyio
async def test_execute_custom_result(tmp_path: Path) -> None:
    """Smoke test to ensure custom StepResult values are propagated."""
    capture = StepCallCapture()
    plugin = _make_plugin(capture, result=StepResult.ok(table_counts={"core.ast_nodes": 2}))

    ctx = build_target_context_for_plugin(
        plugin,
        tmp_path,
        config=TargetContextConfig(),
    )
    result = await plugin.execute(ctx)

    expect_true(result.success)
    expect_equal(result.row_counts.get("core.ast_nodes"), 2)
