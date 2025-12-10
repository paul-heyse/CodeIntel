"""Tests for DocstringsIngestPlugin and basic fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.plugins.docstrings_plugin import DocstringsIngestPlugin, StepFactory
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.assertions.logging_assertions import assert_logged
from tests._helpers.fakes.contexts import TargetResourceOverrides
from tests._helpers.fakes.ingestion_plugins import (
    StepCallCapture,
    make_recording_adapter_factories,
    make_recording_step_factory,
)
from tests._helpers.fakes.recording_gateways import FailingGateway
from tests._helpers.ingestion import (
    TargetContextConfig,
    build_target_context_for_plugin,
    make_resource_case_params,
)
from tests.ingestion.plugins._wiring import run_sync_plugin_wiring_scenario


def _make_plugin(
    capture: StepCallCapture,
    *,
    table_key: str = "core.docstrings",
    result: StepResult | None = None,
) -> DocstringsIngestPlugin:
    storage_factory, discovery_factory = make_recording_adapter_factories(capture)
    step_factory = cast(
        "StepFactory",
        make_recording_step_factory(capture, table_key=table_key, result=result),
    )
    return DocstringsIngestPlugin(
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
async def test_docstrings_wiring_scenarios(
    tmp_path: Path, options: dict[str, bool], ingestion_gateway
) -> None:
    """Shared wiring coverage for DocstringsIngestPlugin."""
    await run_sync_plugin_wiring_scenario(
        _make_plugin,
        tmp_path,
        table_key="core.docstrings",
        options=options,
        gateway=ingestion_gateway,
    )


@pytest.mark.anyio
async def test_gateway_errors_log_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, ingestion_gateway
) -> None:
    """Database lookup failures should be logged and yield an empty module set."""
    capture = StepCallCapture()
    plugin = _make_plugin(capture)
    failing_gateway = FailingGateway(ingestion_gateway, "db down")
    caplog.set_level("WARNING")

    ctx = build_target_context_for_plugin(
        plugin,
        tmp_path,
        config=TargetContextConfig(
            gateway=failing_gateway,
            resources=TargetResourceOverrides(modules=()),
        ),
    )
    result = await plugin.execute(ctx)

    expect_true(result.success)
    expect_equal(capture.modules, [])
    assert_logged(caplog.records, level="WARNING", containing="gateway error")


@pytest.mark.anyio
async def test_execute_logs_step_errors(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Step errors should be logged with warnings while returning counts."""
    capture = StepCallCapture()
    error_result = StepResult(
        table_counts={"core.docstrings": 2},
        errors=["missing docstring in pkg/doc.py", "parse error in pkg/unicode/delta.py"],
    )
    plugin = _make_plugin(capture, result=error_result)
    caplog.set_level("WARNING")

    ctx = build_target_context_for_plugin(
        plugin,
        tmp_path,
        config=TargetContextConfig(
            resources=TargetResourceOverrides(
                modules=("pkg/doc.py", "pkg/unicode/delta.py"),
            ),
        ),
    )
    result = await plugin.execute(ctx)

    expect_true(result.success)
    expect_equal(result.row_counts, {"core.docstrings": 2})
    assert_logged(caplog.records, level="WARNING", containing="missing docstring in pkg/doc.py")
    assert_logged(caplog.records, level="WARNING", containing="pkg/unicode/delta.py")
