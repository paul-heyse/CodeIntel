"""Tests for serving layer request context management.

This module tests the RequestContext context variable management, ensuring
proper context propagation and isolation across concurrent operations.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal

from codeintel.serving.context import (
    RequestContext,
    get_current_request_context,
    reset_current_request_context,
    set_current_request_context,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)

if TYPE_CHECKING:
    from contextvars import Token


def test_request_context_default_is_none() -> None:
    """Verify get_current_request_context returns None when not set."""
    token = set_current_request_context(RequestContext(correlation_id="temp", transport="http"))
    reset_current_request_context(token)

    result = get_current_request_context()
    expect_true(result is None)


def test_set_and_get_request_context() -> None:
    """Verify set/get round-trip preserves context values."""
    ctx = RequestContext(
        correlation_id="test-123",
        transport="http",
        operation="get_function_summary",
        dataset="analytics.goid_risk_factors",
        repo="demo/repo",
        commit="deadbeef",
        client_id="test-client",
        user_agent="pytest/1.0",
    )

    token = set_current_request_context(ctx)
    try:
        retrieved = get_current_request_context()
        retrieved = expect_is_not_none(retrieved)
        expect_equal(retrieved.correlation_id, "test-123")
        expect_equal(retrieved.transport, "http")
        expect_equal(retrieved.operation, "get_function_summary")
        expect_equal(retrieved.dataset, "analytics.goid_risk_factors")
        expect_equal(retrieved.repo, "demo/repo")
        expect_equal(retrieved.commit, "deadbeef")
        expect_equal(retrieved.client_id, "test-client")
        expect_equal(retrieved.user_agent, "pytest/1.0")
    finally:
        reset_current_request_context(token)


def test_reset_request_context_restores_previous() -> None:
    """Verify reset_current_request_context restores the previous value."""
    ctx1 = RequestContext(correlation_id="ctx-1", transport="http")
    token1 = set_current_request_context(ctx1)

    ctx2 = RequestContext(correlation_id="ctx-2", transport="mcp")
    token2 = set_current_request_context(ctx2)

    current = get_current_request_context()
    current = expect_is_not_none(current)
    expect_equal(current.correlation_id, "ctx-2")

    reset_current_request_context(token2)

    current = get_current_request_context()
    current = expect_is_not_none(current)
    expect_equal(current.correlation_id, "ctx-1")

    reset_current_request_context(token1)


def test_request_context_with_optional_fields() -> None:
    """Verify RequestContext works with optional fields as None."""
    ctx = RequestContext(
        correlation_id="minimal-ctx",
        transport="cli",
    )

    expect_true(ctx.operation is None)
    expect_true(ctx.dataset is None)
    expect_true(ctx.repo is None)
    expect_true(ctx.commit is None)
    expect_true(ctx.snapshot is None)
    expect_true(ctx.graph_scope is None)
    expect_true(ctx.client_id is None)
    expect_true(ctx.user_agent is None)


def test_request_context_with_snapshot_and_graph_scope() -> None:
    """Verify RequestContext can hold snapshot and graph_scope objects."""
    mock_snapshot = {"repo": "test", "commit": "abc"}
    mock_graph_scope = {"subsystem_id": "core"}

    ctx = RequestContext(
        correlation_id="with-objects",
        transport="http",
        snapshot=mock_snapshot,
        graph_scope=mock_graph_scope,
    )

    expect_equal(ctx.snapshot, mock_snapshot)
    expect_equal(ctx.graph_scope, mock_graph_scope)


def test_context_isolation_across_threads() -> None:
    """Verify context is isolated between threads."""
    results: dict[str, str | None] = {}

    def worker(thread_id: str, correlation_id: str) -> None:
        ctx = RequestContext(correlation_id=correlation_id, transport="http")
        token = set_current_request_context(ctx)
        try:
            time.sleep(0.01)
            current = get_current_request_context()
            results[thread_id] = current.correlation_id if current else None
        finally:
            reset_current_request_context(token)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(worker, "thread-1", "corr-1"),
            executor.submit(worker, "thread-2", "corr-2"),
            executor.submit(worker, "thread-3", "corr-3"),
        ]
        for f in futures:
            f.result()

    expect_equal(results["thread-1"], "corr-1")
    expect_equal(results["thread-2"], "corr-2")
    expect_equal(results["thread-3"], "corr-3")


def test_context_not_visible_in_other_thread() -> None:
    """Verify context set in one thread is not visible in another."""
    main_ctx = RequestContext(correlation_id="main-thread", transport="http")
    token = set_current_request_context(main_ctx)

    other_thread_result: list[RequestContext | None] = []

    def other_thread_worker() -> None:
        ctx = get_current_request_context()
        other_thread_result.append(ctx)

    try:
        thread = threading.Thread(target=other_thread_worker)
        thread.start()
        thread.join()

        expect_equal(len(other_thread_result), 1)
        expect_true(other_thread_result[0] is None)
    finally:
        reset_current_request_context(token)


def test_context_propagation_in_async_task() -> None:
    """Verify context propagates to async tasks correctly."""

    async def async_worker() -> RequestContext | None:
        await asyncio.sleep(0)
        return get_current_request_context()

    async def main() -> RequestContext | None:
        ctx = RequestContext(correlation_id="async-main", transport="http")
        token = set_current_request_context(ctx)
        try:
            return await async_worker()
        finally:
            reset_current_request_context(token)

    result = asyncio.run(main())
    result = expect_is_not_none(result)
    expect_equal(result.correlation_id, "async-main")


def test_context_isolation_across_async_tasks() -> None:
    """Verify context is isolated between concurrent async tasks."""
    results: dict[str, str | None] = {}

    async def worker(task_id: str, correlation_id: str) -> None:
        ctx = RequestContext(correlation_id=correlation_id, transport="http")
        token = set_current_request_context(ctx)
        try:
            await asyncio.sleep(0.01)
            current = get_current_request_context()
            results[task_id] = current.correlation_id if current else None
        finally:
            reset_current_request_context(token)

    async def main() -> None:
        await asyncio.gather(
            worker("task-1", "corr-1"),
            worker("task-2", "corr-2"),
            worker("task-3", "corr-3"),
        )

    asyncio.run(main())

    expect_equal(results["task-1"], "corr-1")
    expect_equal(results["task-2"], "corr-2")
    expect_equal(results["task-3"], "corr-3")


def test_nested_context_stacking() -> None:
    """Verify nested contexts can be stacked and unwound correctly."""
    tokens: list[Token[RequestContext | None]] = []

    for i in range(3):
        ctx = RequestContext(correlation_id=f"level-{i}", transport="http")
        tokens.append(set_current_request_context(ctx))

    current = get_current_request_context()
    current = expect_is_not_none(current)
    expect_equal(current.correlation_id, "level-2")

    reset_current_request_context(tokens[2])
    current = get_current_request_context()
    current = expect_is_not_none(current)
    expect_equal(current.correlation_id, "level-1")

    reset_current_request_context(tokens[1])
    current = get_current_request_context()
    current = expect_is_not_none(current)
    expect_equal(current.correlation_id, "level-0")

    reset_current_request_context(tokens[0])
    current = get_current_request_context()
    expect_true(current is None)


def test_context_manager_pattern() -> None:
    """Verify context can be used in a context-manager-like pattern."""

    class ContextScope:
        """Helper for context-manager usage of RequestContext."""

        def __init__(self, correlation_id: str, transport: Literal["http", "mcp", "cli"]) -> None:
            """Initialize scope."""
            self.ctx = RequestContext(correlation_id=correlation_id, transport=transport)
            self.token: Token[RequestContext | None] | None = None

        def __enter__(self) -> RequestContext:
            """
            Enter scope and set context.

            Returns
            -------
            RequestContext
                The context that was set.
            """
            self.token = set_current_request_context(self.ctx)
            return self.ctx

        def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
            """Exit scope and reset context."""
            if self.token is not None:
                reset_current_request_context(self.token)

    expect_true(get_current_request_context() is None)

    with ContextScope("scoped-ctx", "http") as ctx:
        current = get_current_request_context()
        current = expect_is_not_none(current)
        expect_true(current is ctx)
        expect_equal(ctx.correlation_id, "scoped-ctx")

    expect_true(get_current_request_context() is None)


def test_all_transport_types() -> None:
    """Verify all supported transport types work correctly."""
    transports: list[Literal["http", "mcp", "cli"]] = ["http", "mcp", "cli"]
    for transport in transports:
        ctx = RequestContext(correlation_id=f"test-{transport}", transport=transport)
        token = set_current_request_context(ctx)
        try:
            current = get_current_request_context()
            current = expect_is_not_none(current)
            expect_equal(current.transport, transport)
        finally:
            reset_current_request_context(token)
