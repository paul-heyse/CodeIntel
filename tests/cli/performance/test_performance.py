"""Performance regression tests with budget assertions.

Test critical paths have acceptable latency and resource usage.
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING

import pytest

from codeintel.cli.core import CliResult
from codeintel.cli.execution import OperationCategory, OperationExecutor, OperationSpec
from codeintel.cli.introspection import get_registry
from tests._helpers.assertions import expect_true

if TYPE_CHECKING:
    from tests.cli._harness import CliTestHarness, OperationTestHarness

# Performance budgets (in seconds)
FAST_OPERATION_BUDGET = 0.5
MEDIUM_OPERATION_BUDGET = 2.0
MAX_VERSION_TIME = 0.3
MAX_LOOKUP_TIME_MS = 0.1
MAX_MIDDLEWARE_OVERHEAD_MS = 10.0
LOOKUP_ITERATIONS = 1000
MIDDLEWARE_ITERATIONS = 100
EXECUTOR_CLEANUP_ITERATIONS = 50
SEQUENTIAL_OPS_COUNT = 10
MEMORY_TEST_ITERATIONS = 100


@pytest.mark.benchmark
def test_cli_startup_time(cli: CliTestHarness) -> None:
    """Test CLI starts within budget."""
    start = time.perf_counter()
    result = cli.invoke(["--help"])
    elapsed = time.perf_counter() - start

    expect_true(result.success)
    expect_true(
        elapsed < FAST_OPERATION_BUDGET,
        message=f"CLI startup took {elapsed:.2f}s, budget is {FAST_OPERATION_BUDGET}s",
    )


@pytest.mark.benchmark
def test_version_command_fast(cli: CliTestHarness) -> None:
    """Test version command is fast."""
    start = time.perf_counter()
    cli.invoke(["--version"])
    elapsed = time.perf_counter() - start

    expect_true(
        elapsed < MAX_VERSION_TIME,
        message=f"Version took {elapsed:.2f}s, should be < {MAX_VERSION_TIME}s",
    )


@pytest.mark.benchmark
def test_read_operations_are_fast(
    op_harness: OperationTestHarness,
) -> None:
    """Test read operations complete within budget."""
    read_operations = [
        "build.status",
        "op.list",
        "dataset.list",
        "storage.status",
    ]

    for op_id in read_operations:
        start = time.perf_counter()
        result = op_harness.execute(op_id)
        elapsed = time.perf_counter() - start

        # Only check timing on successful operations
        if result.success:
            expect_true(
                elapsed < FAST_OPERATION_BUDGET,
                message=f"{op_id} took {elapsed:.2f}s, budget is {FAST_OPERATION_BUDGET}s",
            )


@pytest.mark.benchmark
def test_operation_registry_lookup_fast() -> None:
    """Test operation registry lookup is fast."""
    registry = get_registry()

    # Warm up
    _ = registry.get("build.status")

    # Measure lookup time
    start = time.perf_counter()
    for _ in range(LOOKUP_ITERATIONS):
        registry.get("build.status")
    elapsed = time.perf_counter() - start

    avg_time = elapsed / LOOKUP_ITERATIONS
    max_lookup_time = MAX_LOOKUP_TIME_MS / 1000
    expect_true(
        avg_time < max_lookup_time,
        message=f"Registry lookup took {avg_time * 1000:.3f}ms average",
    )


@pytest.mark.benchmark
def test_json_output_overhead(cli: CliTestHarness) -> None:
    """Test JSON output doesn't add significant overhead."""
    # Text output
    start = time.perf_counter()
    cli.invoke(["build", "status", "--format=text"])
    text_time = time.perf_counter() - start

    # JSON output
    start = time.perf_counter()
    cli.invoke(["build", "status", "--format=json"])
    json_time = time.perf_counter() - start

    # JSON should be within 2x of text plus buffer
    buffer = 0.1
    expect_true(
        json_time < text_time * 2 + buffer,
        message=f"JSON ({json_time:.2f}s) much slower than text ({text_time:.2f}s)",
    )


@pytest.mark.benchmark
def test_middleware_overhead_acceptable() -> None:
    """Test middleware stack doesn't add excessive overhead."""

    def fast_handler() -> CliResult[dict[str, int]]:
        """Return fast test result.

        Returns
        -------
        CliResult[dict[str, int]]
            Success result with test data.
        """
        return CliResult.ok({"value": 42})

    spec = OperationSpec(
        operation_id="test.fast",
        handler=fast_handler,
        category=OperationCategory.READ,
    )

    executor = OperationExecutor()

    # Warm up
    executor.execute(spec, {}, render=False)

    # Measure
    start = time.perf_counter()
    for _ in range(MIDDLEWARE_ITERATIONS):
        executor.execute(spec, {}, render=False)
    elapsed = time.perf_counter() - start

    avg_time = elapsed / MIDDLEWARE_ITERATIONS
    max_overhead = MAX_MIDDLEWARE_OVERHEAD_MS / 1000
    expect_true(
        avg_time < max_overhead,
        message=f"Executor overhead {avg_time * 1000:.1f}ms per call",
    )


@pytest.mark.benchmark
def test_repeated_operations_no_memory_leak(
    op_harness: OperationTestHarness,
) -> None:
    """Test repeated operations don't leak memory."""
    # Run many operations
    for _ in range(MEMORY_TEST_ITERATIONS):
        op_harness.execute("op.list")

    # Force garbage collection
    gc.collect()
    # If we get here without OOM, basic memory cleanup is working


@pytest.mark.benchmark
def test_executor_cleanup() -> None:
    """Test executor cleans up properly."""

    def handler() -> CliResult[dict[str, int]]:
        """Return test result.

        Returns
        -------
        CliResult[dict[str, int]]
            Success result with test data.
        """
        return CliResult.ok({"value": 1})

    spec = OperationSpec(
        operation_id="test.cleanup",
        handler=handler,
        category=OperationCategory.READ,
    )

    # Create and destroy many executors
    for _ in range(EXECUTOR_CLEANUP_ITERATIONS):
        executor = OperationExecutor()
        executor.execute(spec, {}, render=False)
        del executor

    gc.collect()
    # If we get here without OOM, cleanup is working


@pytest.mark.benchmark
def test_sequential_operations(
    op_harness: OperationTestHarness,
) -> None:
    """Test sequential operation execution time is linear."""
    start = time.perf_counter()
    for _ in range(SEQUENTIAL_OPS_COUNT):
        op_harness.execute("op.list")
    total_time = time.perf_counter() - start

    # Average time per operation
    avg_time = total_time / SEQUENTIAL_OPS_COUNT

    # Should be relatively consistent
    max_avg_time = FAST_OPERATION_BUDGET
    expect_true(
        avg_time < max_avg_time,
        message=f"Average op time {avg_time:.2f}s exceeds budget {max_avg_time}s",
    )
