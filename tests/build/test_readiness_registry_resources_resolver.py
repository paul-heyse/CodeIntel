"""Registry and resources edge-case tests."""

from __future__ import annotations

from codeintel.build.resources import TargetExecution, TargetResources
from codeintel.build.targets import get_target_by_table
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_true,
)
from tests._helpers.catalog import make_target_descriptor
from tests._helpers.contracts import contract_for_keys

_DURATION_THRESHOLD_MS = 5000


def test_registry_get_target_by_table() -> None:
    """get_target_by_table returns producer target for table."""
    target = make_target_descriptor(
        name="producer",
        module="analytics",
        contract=contract_for_keys(("core.produced",)),
    )

    found = get_target_by_table("core.produced", targets=(target,))

    expect_true(found is target)


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
