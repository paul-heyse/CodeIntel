"""Tests for serving layer request context management.

This module tests the RequestContext context variable management, ensuring
proper context propagation and isolation across concurrent operations.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import Token
from typing import Literal

from codeintel.serving.context import (
    RequestContext,
    get_current_request_context,
    reset_current_request_context,
    set_current_request_context,
)

# =============================================================================
# Basic Context Operations
# =============================================================================


def test_request_context_default_is_none() -> None:
    """Verify get_current_request_context returns None when not set."""
    # Reset any existing context first
    token = set_current_request_context(RequestContext(correlation_id="temp", transport="http"))
    reset_current_request_context(token)

    result = get_current_request_context()
    assert result is None


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
        assert retrieved is not None
        assert retrieved.correlation_id == "test-123"
        assert retrieved.transport == "http"
        assert retrieved.operation == "get_function_summary"
        assert retrieved.dataset == "analytics.goid_risk_factors"
        assert retrieved.repo == "demo/repo"
        assert retrieved.commit == "deadbeef"
        assert retrieved.client_id == "test-client"
        assert retrieved.user_agent == "pytest/1.0"
    finally:
        reset_current_request_context(token)


def test_reset_request_context_restores_previous() -> None:
    """Verify reset_current_request_context restores the previous value."""
    # Set initial context
    ctx1 = RequestContext(correlation_id="ctx-1", transport="http")
    token1 = set_current_request_context(ctx1)

    # Set nested context
    ctx2 = RequestContext(correlation_id="ctx-2", transport="mcp")
    token2 = set_current_request_context(ctx2)

    # Verify nested context is active
    current = get_current_request_context()
    assert current is not None
    assert current.correlation_id == "ctx-2"

    # Reset nested context
    reset_current_request_context(token2)

    # Verify original context is restored
    current = get_current_request_context()
    assert current is not None
    assert current.correlation_id == "ctx-1"

    # Clean up
    reset_current_request_context(token1)


def test_request_context_with_optional_fields() -> None:
    """Verify RequestContext works with optional fields as None."""
    ctx = RequestContext(
        correlation_id="minimal-ctx",
        transport="cli",
    )

    assert ctx.operation is None
    assert ctx.dataset is None
    assert ctx.repo is None
    assert ctx.commit is None
    assert ctx.snapshot is None
    assert ctx.graph_scope is None
    assert ctx.client_id is None
    assert ctx.user_agent is None


def test_request_context_with_snapshot_and_graph_scope() -> None:
    """Verify RequestContext can hold snapshot and graph_scope objects."""
    # Use arbitrary objects to test Any fields
    mock_snapshot = {"repo": "test", "commit": "abc"}
    mock_graph_scope = {"subsystem_id": "core"}

    ctx = RequestContext(
        correlation_id="with-objects",
        transport="http",
        snapshot=mock_snapshot,
        graph_scope=mock_graph_scope,
    )

    assert ctx.snapshot == mock_snapshot
    assert ctx.graph_scope == mock_graph_scope


# =============================================================================
# Thread Isolation Tests
# =============================================================================


def test_context_isolation_across_threads() -> None:
    """Verify context is isolated between threads."""
    results: dict[str, str | None] = {}

    def worker(thread_id: str, correlation_id: str) -> None:
        ctx = RequestContext(correlation_id=correlation_id, transport="http")
        token = set_current_request_context(ctx)
        try:
            # Small sleep to allow interleaving
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

    # Each thread should have seen its own context
    assert results["thread-1"] == "corr-1"
    assert results["thread-2"] == "corr-2"
    assert results["thread-3"] == "corr-3"


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

        # Other thread should not see main thread's context
        assert len(other_thread_result) == 1
        assert other_thread_result[0] is None
    finally:
        reset_current_request_context(token)


# =============================================================================
# Async Context Tests
# =============================================================================


def test_context_propagation_in_async_task() -> None:
    """Verify context propagates to async tasks correctly."""

    async def async_worker() -> RequestContext | None:
        await asyncio.sleep(0)  # Yield to event loop
        return get_current_request_context()

    async def main() -> RequestContext | None:
        ctx = RequestContext(correlation_id="async-main", transport="http")
        token = set_current_request_context(ctx)
        try:
            return await async_worker()
        finally:
            reset_current_request_context(token)

    result = asyncio.run(main())
    assert result is not None
    assert result.correlation_id == "async-main"


def test_context_isolation_across_async_tasks() -> None:
    """Verify context is isolated between concurrent async tasks."""
    results: dict[str, str | None] = {}

    async def worker(task_id: str, correlation_id: str) -> None:
        ctx = RequestContext(correlation_id=correlation_id, transport="http")
        token = set_current_request_context(ctx)
        try:
            await asyncio.sleep(0.01)  # Allow interleaving
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

    # Each task should have seen its own context
    assert results["task-1"] == "corr-1"
    assert results["task-2"] == "corr-2"
    assert results["task-3"] == "corr-3"


# =============================================================================
# Nested Context Tests
# =============================================================================


def test_nested_context_stacking() -> None:
    """Verify nested contexts can be stacked and unwound correctly."""
    tokens: list[Token[RequestContext | None]] = []

    # Stack 3 contexts
    for i in range(3):
        ctx = RequestContext(correlation_id=f"level-{i}", transport="http")
        tokens.append(set_current_request_context(ctx))

    # Verify topmost is active
    current = get_current_request_context()
    assert current is not None
    assert current.correlation_id == "level-2"

    # Unwind and verify each level
    reset_current_request_context(tokens[2])
    current = get_current_request_context()
    assert current is not None
    assert current.correlation_id == "level-1"

    reset_current_request_context(tokens[1])
    current = get_current_request_context()
    assert current is not None
    assert current.correlation_id == "level-0"

    reset_current_request_context(tokens[0])
    current = get_current_request_context()
    assert current is None


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

    # Verify clean state
    assert get_current_request_context() is None

    with ContextScope("scoped-ctx", "http") as ctx:
        current = get_current_request_context()
        assert current is not None
        assert current is ctx
        assert ctx.correlation_id == "scoped-ctx"

    # Verify cleanup
    assert get_current_request_context() is None


# =============================================================================
# Transport Type Tests
# =============================================================================


def test_all_transport_types() -> None:
    """Verify all supported transport types work correctly."""
    transports: list[Literal["http", "mcp", "cli"]] = ["http", "mcp", "cli"]
    for transport in transports:
        ctx = RequestContext(correlation_id=f"test-{transport}", transport=transport)
        token = set_current_request_context(ctx)
        try:
            current = get_current_request_context()
            assert current is not None
            assert current.transport == transport
        finally:
            reset_current_request_context(token)
