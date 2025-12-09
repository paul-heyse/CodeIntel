"""Async-aware middleware for CLI operations.

Provide middleware protocols and implementations that work with
both sync and async handlers.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from codeintel.cli.async_types import ProgressEvent, ProgressState, get_handler_type
from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.resilience import RetryPolicy
from codeintel.cli.results import CliResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from opentelemetry.trace import Span

# Optional opentelemetry import
try:
    from opentelemetry import trace as otel_trace

    _HAS_OPENTELEMETRY = True
except ImportError:
    _HAS_OPENTELEMETRY = False
    otel_trace = None  # type: ignore[assignment]


@dataclass
class AsyncMiddlewareContext:
    """Context passed through async middleware chain.

    Parameters
    ----------
    operation_id
        Operation identifier.
    params
        Operation parameters.
    handler_type
        Type of handler ('sync', 'async', 'streaming').
    metadata
        Additional context metadata.
    """

    operation_id: str
    params: dict[str, Any]
    handler_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class AsyncOperationMiddleware(ABC):
    """Base class for async-aware middleware.

    Middleware can wrap both sync and async operations, providing
    cross-cutting concerns like tracing, logging, and retry handling.
    """

    @abstractmethod
    async def before(self, ctx: AsyncMiddlewareContext) -> AsyncMiddlewareContext:
        """Execute before operation.

        Parameters
        ----------
        ctx
            Middleware context.

        Returns
        -------
        AsyncMiddlewareContext
            Modified context.
        """
        ...

    @abstractmethod
    async def after(
        self,
        ctx: AsyncMiddlewareContext,
        result: CliResult[Any],
        error: Exception | None = None,
    ) -> CliResult[Any]:
        """Execute after operation.

        Parameters
        ----------
        ctx
            Middleware context.
        result
            Operation result.
        error
            Exception if operation failed.

        Returns
        -------
        CliResult[Any]
            Modified result.
        """
        ...


class AsyncTracingMiddleware(AsyncOperationMiddleware):
    """Middleware for async-aware tracing.

    Create OpenTelemetry spans for operations with proper async context.
    """

    def __init__(self) -> None:
        """Initialize tracing middleware."""
        self._spans: dict[str, Span] = {}

    async def before(self, ctx: AsyncMiddlewareContext) -> AsyncMiddlewareContext:
        """Start trace span.

        Parameters
        ----------
        ctx
            Middleware context.

        Returns
        -------
        AsyncMiddlewareContext
            Context with span added to metadata.
        """
        if not _HAS_OPENTELEMETRY or otel_trace is None:
            return ctx

        tracer = otel_trace.get_tracer("codeintel.cli")
        span = tracer.start_span(f"cli.{ctx.operation_id}")
        span.set_attribute("operation.id", ctx.operation_id)
        span.set_attribute("operation.handler_type", ctx.handler_type)
        span.set_attribute("operation.params_count", len(ctx.params))

        ctx.metadata["span"] = span
        self._spans[ctx.operation_id] = span

        return ctx

    async def after(
        self,
        ctx: AsyncMiddlewareContext,
        result: CliResult[Any],
        error: Exception | None = None,
    ) -> CliResult[Any]:
        """End trace span.

        Parameters
        ----------
        ctx
            Middleware context.
        result
            Operation result.
        error
            Exception if operation failed.

        Returns
        -------
        CliResult[Any]
            Unmodified result.
        """
        span = ctx.metadata.get("span")
        if span is not None:
            if error is not None:
                span.record_exception(error)
                span.set_attribute(key="error", value=True)
            else:
                span.set_attribute(key="success", value=result.success)
            span.end()

        # Cleanup
        self._spans.pop(ctx.operation_id, None)

        return result


class AsyncResilienceMiddleware(AsyncOperationMiddleware):
    """Middleware for async retry handling.

    Apply retry policy with exponential backoff for async operations.
    Uses the unified RetryPolicy from the resilience module.

    Parameters
    ----------
    policy
        Retry policy to apply.
    """

    def __init__(self, policy: RetryPolicy | None = None) -> None:
        """Initialize resilience middleware."""
        self._policy = policy or RetryPolicy()

    async def before(self, ctx: AsyncMiddlewareContext) -> AsyncMiddlewareContext:
        """Add retry state to context.

        Parameters
        ----------
        ctx
            Middleware context.

        Returns
        -------
        AsyncMiddlewareContext
            Context with retry state.
        """
        ctx.metadata["retry_attempt"] = 0
        ctx.metadata["retry_policy"] = self._policy
        return ctx

    async def after(
        self,
        ctx: AsyncMiddlewareContext,
        result: CliResult[Any],
        error: Exception | None = None,
    ) -> CliResult[Any]:
        """Handle retry logic.

        Parameters
        ----------
        ctx
            Middleware context.
        result
            Operation result.
        error
            Exception if operation failed.

        Returns
        -------
        CliResult[Any]
            Result (may have retried).
        """
        # If no error, return as-is
        if error is None:
            return result

        policy: RetryPolicy = ctx.metadata.get("retry_policy", self._policy)
        attempt: int = ctx.metadata.get("retry_attempt", 0)

        # Check if we should retry
        if attempt >= policy.max_attempts:
            return result

        if not policy.is_retryable(error):
            return result

        # Update attempt count for next middleware call
        ctx.metadata["retry_attempt"] = attempt + 1

        return result


@asynccontextmanager
async def async_middleware_context[T](
    handler: Callable[..., CliResult[T]],
    params: dict[str, Any],
    *,
    middleware: list[AsyncOperationMiddleware] | None = None,
    operation_id: str = "unknown",
) -> AsyncGenerator[AsyncMiddlewareContext]:
    """Create a context manager for middleware execution.

    Parameters
    ----------
    handler
        Handler function.
    params
        Operation parameters.
    middleware
        List of middleware to apply.
    operation_id
        Operation identifier.

    Yields
    ------
    AsyncMiddlewareContext
        The middleware context.
    """
    handler_type = get_handler_type(handler)
    ctx = AsyncMiddlewareContext(
        operation_id=operation_id,
        params=params,
        handler_type=handler_type,
    )

    middleware = middleware or []

    # Execute before hooks
    for mw in middleware:
        ctx = await mw.before(ctx)

    try:
        yield ctx
    finally:
        # After hooks executed by caller
        pass


class AsyncProgressMiddleware(AsyncOperationMiddleware):
    """Middleware for progress reporting.

    Emit progress events during operation execution.

    Parameters
    ----------
    callback
        Callback to receive progress events.
    """

    def __init__(self, callback: Callable[[ProgressEvent], None] | None = None) -> None:
        """Initialize progress middleware."""
        self._callback = callback

    async def before(self, ctx: AsyncMiddlewareContext) -> AsyncMiddlewareContext:
        """Emit start progress event.

        Parameters
        ----------
        ctx
            Middleware context.

        Returns
        -------
        AsyncMiddlewareContext
            Context with progress callback.
        """
        if self._callback:
            event = ProgressEvent(
                operation_id=ctx.operation_id,
                state=ProgressState.RUNNING,
                progress=0.0,
                message=f"Starting {ctx.operation_id}",
            )
            self._callback(event)
            ctx.metadata["progress_callback"] = self._callback

        return ctx

    async def after(
        self,
        ctx: AsyncMiddlewareContext,
        result: CliResult[Any],
        error: Exception | None = None,
    ) -> CliResult[Any]:
        """Emit completion progress event.

        Parameters
        ----------
        ctx
            Middleware context.
        result
            Operation result.
        error
            Exception if operation failed.

        Returns
        -------
        CliResult[Any]
            Unmodified result.
        """
        if self._callback:
            state = ProgressState.FAILED if error or not result.success else ProgressState.COMPLETED
            event = ProgressEvent(
                operation_id=ctx.operation_id,
                state=state,
                progress=1.0 if result.success else None,
                message="Completed" if result.success else "Failed",
            )
            self._callback(event)

        return result


async def run_with_middleware[T](
    handler: Callable[..., CliResult[T]],
    params: dict[str, Any],
    *,
    middleware: list[AsyncOperationMiddleware] | None = None,
    operation_id: str = "unknown",
) -> CliResult[T]:
    """Run a handler with middleware.

    Parameters
    ----------
    handler
        Handler function.
    params
        Operation parameters.
    middleware
        List of middleware to apply.
    operation_id
        Operation identifier.

    Returns
    -------
    CliResult[T]
        Handler result.
    """
    handler_type = get_handler_type(handler)
    ctx = AsyncMiddlewareContext(
        operation_id=operation_id,
        params=params,
        handler_type=handler_type,
    )

    middleware = middleware or []
    error: Exception | None = None

    # Execute before hooks
    for mw in middleware:
        ctx = await mw.before(ctx)

    # Execute handler
    try:
        if handler_type == "async":
            result: CliResult[T] = await handler(**params)  # type: ignore[misc]
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: handler(**params))
    except Exception as e:  # noqa: BLE001 - need to catch for middleware
        error = e
        result = CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:operation/error",
                title="Operation Error",
                detail=str(e),
                status=500,
            ),
        )

    # Execute after hooks (reverse order)
    for mw in reversed(middleware):
        result = await mw.after(ctx, result, error)

    return result


__all__ = [
    "AsyncMiddlewareContext",
    "AsyncOperationMiddleware",
    "AsyncProgressMiddleware",
    "AsyncResilienceMiddleware",
    "AsyncTracingMiddleware",
    "async_middleware_context",
    "run_with_middleware",
]
