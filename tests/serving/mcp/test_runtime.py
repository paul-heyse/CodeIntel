"""Tests for MCP runtime utilities."""

from __future__ import annotations

import asyncio
import time
from typing import cast

import anyio
import pytest

from codeintel.serving.mcp.runtime import QueryLimiter
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


def test_query_limiter_max_concurrent() -> None:
    """Verify max_concurrent property returns configured value."""
    limiter = QueryLimiter(max_concurrent=3)
    expect_equal(limiter.max_concurrent, 3)


@pytest.mark.anyio
async def test_query_limiter_run_sync_function() -> None:
    """Verify run() executes sync function and returns result."""
    limiter = QueryLimiter(max_concurrent=2)

    def add(a: int, b: int) -> int:
        return a + b

    result = await limiter.run(add, 2, 3)
    expected = 5
    expect_equal(result, expected)


@pytest.mark.anyio
async def test_query_limiter_concurrency() -> None:
    """Verify limiter blocks when at capacity."""
    limiter = QueryLimiter(max_concurrent=1)
    call_order: list[str] = []

    def slow_task(n: int) -> int:
        call_order.append(f"start_{n}")
        time.sleep(0.05)  # Brief sleep to ensure serialization
        call_order.append(f"end_{n}")
        return n

    # Run two tasks concurrently with limit=1
    results = await asyncio.gather(
        limiter.run(slow_task, 1),
        limiter.run(slow_task, 2),
    )

    # Both tasks should complete
    expected_results = [1, 2]
    expect_equal(sorted(cast(tuple[int, int], results)), expected_results)

    # With max_concurrent=1, tasks should serialize:
    # One task must fully complete before the other starts
    expected_count = 4
    expect_equal(len(call_order), expected_count)

    # First task starts and ends before second task starts
    first_start = call_order[0]
    first_end = call_order[1]
    expect_true(first_start.startswith("start_"), message="First entry should be start")
    expect_true(first_end.startswith("end_"), message="Second entry should be end")
    # Extract the number and verify same task started and ended
    first_num = first_start.split("_")[1]
    expect_equal(first_end, f"end_{first_num}")


@pytest.mark.anyio
async def test_query_limiter_run_async_function() -> None:
    """Verify run_async() executes async function with limiting."""
    limiter = QueryLimiter(max_concurrent=2)

    async def async_add(a: int, b: int) -> int:
        await anyio.sleep(0.01)
        return a + b

    result = await limiter.run_async(async_add, 5, 7)
    expected = 12
    expect_equal(result, expected)


@pytest.mark.anyio
async def test_query_limiter_run_async_concurrency() -> None:
    """Verify run_async limits concurrent async tasks."""
    limiter = QueryLimiter(max_concurrent=1)
    execution_order: list[str] = []

    async def async_task(name: str) -> str:
        execution_order.append(f"start_{name}")
        await anyio.sleep(0.05)
        execution_order.append(f"end_{name}")
        return name

    # Run two async tasks concurrently with limit=1
    results = await asyncio.gather(
        limiter.run_async(async_task, "a"),
        limiter.run_async(async_task, "b"),
    )

    expect_equal(sorted(cast(tuple[str, str], results)), ["a", "b"])
    expected_count = 4
    expect_equal(len(execution_order), expected_count)

    # Verify serialization: first task completes before second starts
    first_start = execution_order[0]
    first_end = execution_order[1]
    expect_true(first_start.startswith("start_"), message="First entry should be start")
    expect_true(first_end.startswith("end_"), message="Second entry should be end")
    first_name = first_start.split("_")[1]
    expect_equal(first_end, f"end_{first_name}")


@pytest.mark.anyio
async def test_query_limiter_higher_concurrency() -> None:
    """Verify limiter allows configured concurrency level."""
    max_allowed = 3
    limiter = QueryLimiter(max_concurrent=max_allowed)
    concurrent_count = 0
    max_concurrent_seen = 0

    async def track_concurrent() -> None:
        nonlocal concurrent_count, max_concurrent_seen
        concurrent_count += 1
        max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
        await anyio.sleep(0.02)
        concurrent_count -= 1

    # Run 5 tasks with limit=3
    task_count = 5
    await asyncio.gather(*[limiter.run_async(track_concurrent) for _ in range(task_count)])

    # Should never exceed configured max
    expect_true(
        max_concurrent_seen <= max_allowed,
        message=f"Should not exceed {max_allowed} concurrent",
    )
    # Should reach at least 2 concurrent (may not hit exactly 3 due to timing)
    min_expected = 2
    expect_true(
        max_concurrent_seen >= min_expected,
        message=f"Should reach at least {min_expected} concurrent",
    )
