"""Tests for AstExtractPlugin module wiring."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.plugins.ast_extract import AstExtractPlugin, StepFactory
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.fakes.ingestion_plugins import (
    StepCallCapture,
    make_recording_adapter_factories,
    make_recording_step_factory,
)
from tests._helpers.ingestion import (
    TargetContextConfig,
    build_target_context_for_plugin,
    make_resource_case_params,
)
from tests.ingestion.plugins._wiring import run_sync_plugin_wiring_scenario


def _make_plugin(
    capture: StepCallCapture,
    *,
    table_key: str = "core.ast_nodes",
    result: StepResult | None = None,
) -> AstExtractPlugin:
    storage_factory, discovery_factory = make_recording_adapter_factories(capture)
    step_factory = cast(
        "StepFactory",
        make_recording_step_factory(capture, table_key=table_key, result=result),
    )
    return AstExtractPlugin(
        storage_adapter_factory=storage_factory,
        discovery_adapter_factory=discovery_factory,
        step_factory=step_factory,
    )


RESOURCE_CASES = make_resource_case_params()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "options",
    [params for _, params in RESOURCE_CASES],
    ids=[name for name, _ in RESOURCE_CASES],
)
async def test_ast_extract_wiring_scenarios(
    tmp_path: Path, options: dict[str, bool], ingestion_gateway
) -> None:
    """Shared wiring coverage for AstExtractPlugin."""
    await run_sync_plugin_wiring_scenario(
        _make_plugin,
        tmp_path,
        table_key="core.ast_nodes",
        options=options,
        gateway=ingestion_gateway,
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
