"""Registry and resources edge-case tests."""

from __future__ import annotations

from codeintel.build.resources import TargetExecution, TargetResources
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_true,
)

_DURATION_THRESHOLD_MS = 5000


def test_target_resources_and_execution_helpers() -> None:
    """Resource and execution helpers return expected values."""
    resources = TargetResources(tools=("tool",))
    expect_true(resources.requires_any_tool())
    expect_false(TargetResources().requires_any_tool())

    execution = TargetExecution(
        cpu_intensive=True, io_intensive=True, memory_intensive=True, max_runtime_ms=10000
    )
    expect_equal(execution.estimated_duration_ms(), 10000)

    execution_light = TargetExecution(
        cpu_intensive=True, io_intensive=False, memory_intensive=False, max_runtime_ms=60000
    )
    expect_true(execution_light.estimated_duration_ms() > _DURATION_THRESHOLD_MS)
