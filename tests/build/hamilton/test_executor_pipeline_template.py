"""Executor pipeline template tests.

This module validates that the reusable executor pipeline template in
``codeintel.build.hamilton.templates.materialize_template`` can be instantiated via
Hamilton's ``@subdag`` decorator and produces correct TargetRunRecords for
success, failure, and skip scenarios.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType
from typing import TYPE_CHECKING, cast

import hamilton.driver as h_driver
from hamilton.function_modifiers import source, subdag, tag, value

from codeintel.build.contracts import OutputContract
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.templates import materialize_template
from codeintel.build.hamilton.templates.materialize_template import executor_materialize
from codeintel.build.targets import OutputTarget, TargetGraph
from tests._helpers.assertions import assert_record_row_counts, assert_target_ok
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness

if TYPE_CHECKING:
    from typing import Protocol

    class _EphemeralExtractModule(Protocol):
        t__goids__extract: object
        t__goids: object


# Keep types available for Hamilton's runtime type resolution
_HAMILTON_TYPE_HINTS = (TargetRunRecord,)


def _make_env(harness: HamiltonBuildHarness) -> BuildEnv:
    """Create a BuildEnv for testing.

    Returns
    -------
    BuildEnv
        Build environment configured for testing.
    """
    return replace(harness.build_env(), force_targets=frozenset({"goids"}))


def _make_graph() -> TargetGraph:
    """Create a minimal TargetGraph with goids target.

    Returns
    -------
    TargetGraph
        Target graph with goids target registered.
    """
    graph = TargetGraph()
    graph.register(
        OutputTarget(
            name="goids",
            module="graphs",
            contract=OutputContract.simple(table_keys=("core.goids", "core.goid_crosswalk")),
        )
    )
    return graph


def test_execution_result_contract() -> None:
    """Verify ExecutionResult exposes the executor boundary fields."""
    result = ExecutionResult.ok(table_counts={"core.goids": 10})
    expect_true(hasattr(result, "success"), message="ExecutionResult should have success attr")
    expect_true(
        hasattr(result, "table_counts"), message="ExecutionResult should have table_counts attr"
    )
    expect_true(hasattr(result, "error"), message="ExecutionResult should have error attr")


def test_executor_materialize_success(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Verify executor_materialize produces succeeded record on success."""
    env = _make_env(build_harness)
    graph = _make_graph()

    compute_result = ExecutionResult.ok(table_counts={"core.goids": 100, "core.goid_crosswalk": 50})

    record = executor_materialize(env, graph, "goids", compute_result)

    assert_target_ok(record)
    expect_equal(record.target, expected="goids", label="record.target")
    assert_record_row_counts(
        record,
        {
            "core.goids": 100,
            "core.goid_crosswalk": 50,
        },
    )


def test_executor_materialize_failure(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Verify executor_materialize produces failed record on failure."""
    env = _make_env(build_harness)
    graph = _make_graph()

    compute_result = ExecutionResult.failed("GOID extraction failed: syntax error")

    record = executor_materialize(env, graph, "goids", compute_result)

    assert_target_ok(record, expected_status="failed")
    expect_equal(record.target, expected="goids", label="record.target")
    expect_true(
        record.error is not None and "GOID extraction failed" in record.error,
        message=f"Expected error message containing 'GOID extraction failed', got: {record.error}",
    )


def test_executor_materialize_failure_default_error(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Verify executor_materialize uses default error message when error is None."""
    env = _make_env(build_harness)
    graph = _make_graph()

    compute_result = ExecutionResult(success=False)

    record = executor_materialize(env, graph, "goids", compute_result)

    assert_target_ok(record, expected_status="failed")
    expect_true(
        record.error is not None and "goids computation failed" in record.error,
        message=f"Expected default error message, got: {record.error}",
    )


def _build_subdag_module(compute_result: ExecutionResult) -> ModuleType:
    """Build an ephemeral Hamilton module using materialize_template via @subdag.

    Returns
    -------
    ModuleType
        Ephemeral Hamilton module with executor_pipeline wired via @subdag.
    """
    mod = ModuleType("tests.build.hamilton._executor_pipeline_case")
    mod.__doc__ = "Ephemeral module for testing materialize_template via @subdag."
    sys.modules[mod.__name__] = mod

    # Capture compute_result in closure
    captured_result = compute_result

    @tag(domain="graphs", target="goids", node_type="tool")
    def t__goids__extract(env: BuildEnv) -> ExecutionResult:
        """Return the captured compute result.

        Returns
        -------
        ExecutionResult
            Result for testing.
        """
        # Use env to satisfy Hamilton's requirement for inputs
        _ = env
        return captured_result

    @tag(domain="graphs", target="goids", node_type="materialize")
    @subdag(
        materialize_template,
        inputs={
            "env": source("env"),
            "graph": source("graph"),
            "target_name": value("goids"),
            "compute_result": source("t__goids__extract"),
        },
    )
    def t__goids(executor_record: TargetRunRecord) -> TargetRunRecord:
        """Return the subDAG-produced record.

        Returns
        -------
        TargetRunRecord
            Target execution record produced by the executor pipeline.
        """
        return executor_record

    # Set module ownership for Hamilton discovery
    t__goids__extract.__module__ = mod.__name__
    t__goids.__module__ = mod.__name__

    module_namespace = cast("_EphemeralExtractModule", mod)
    module_namespace.t__goids__extract = t__goids__extract
    module_namespace.t__goids = t__goids
    return mod


def test_executor_pipeline_via_subdag_success(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Verify materialize_template works via @subdag with successful compute."""
    env = _make_env(build_harness)
    graph = _make_graph()

    compute_result = ExecutionResult.ok(table_counts={"core.goids": 42, "core.goid_crosswalk": 21})
    module = _build_subdag_module(compute_result)

    driver = h_driver.Builder().with_modules(module).build()
    results = driver.execute(["t__goids"], inputs={"env": env, "graph": graph})
    record = cast("TargetRunRecord", results["t__goids"])

    assert_target_ok(record)
    expect_equal(record.target, expected="goids", label="record.target")
    assert_record_row_counts(record, {"core.goids": 42})


def test_executor_pipeline_via_subdag_failure(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Verify materialize_template works via @subdag with failed compute."""
    env = _make_env(build_harness)
    graph = _make_graph()

    compute_result = ExecutionResult.failed("Test failure")
    module = _build_subdag_module(compute_result)

    driver = h_driver.Builder().with_modules(module).build()
    results = driver.execute(["t__goids"], inputs={"env": env, "graph": graph})
    record = cast("TargetRunRecord", results["t__goids"])

    assert_target_ok(record, expected_status="failed")
    expect_true(
        record.error is not None and "Test failure" in record.error,
        message=f"Expected error to contain 'Test failure', got: {record.error}",
    )
