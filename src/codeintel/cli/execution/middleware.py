"""Unified middleware for CLI operation execution.

Provide middleware protocol and implementations that work with
both sync and async handlers through a unified interface.
"""

from __future__ import annotations

import contextlib
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from codeintel.cli.execution.context import ExecutionContext
from codeintel.cli.execution.types import ProgressEvent, ProgressState

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from codeintel.cli.core import CliResult

# Optional opentelemetry import
try:
    from opentelemetry import trace as otel_trace

    _HAS_OPENTELEMETRY = True
except ImportError:
    _HAS_OPENTELEMETRY = False
    otel_trace = None  # type: ignore[assignment]


LOG = logging.getLogger(__name__)


class Middleware(ABC):
    """Base class for middleware supporting both sync and async.

    Middleware can implement sync-only, async-only, or both interfaces.
    The executor will call the appropriate methods based on handler type.
    Default async methods delegate to sync counterparts.

    All middleware receives ExecutionContext, enabling rich cross-cutting
    concerns with access to operation ID, params, tracing, progress, etc.
    """

    @abstractmethod
    def before_invoke(
        self,
        ctx: ExecutionContext,
    ) -> dict[str, Any]:
        """Execute before operation (sync).

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        dict[str, Any]
            Context data to pass to after_invoke.
        """
        ...

    async def before_invoke_async(
        self,
        ctx: ExecutionContext,
    ) -> dict[str, Any]:
        """Execute before operation (async).

        Default delegates to sync version.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        dict[str, Any]
            Context data to pass to after_invoke.
        """
        return self.before_invoke(ctx)

    @abstractmethod
    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: CliResult[Any],
        mw_context: dict[str, Any],
    ) -> CliResult[Any]:
        """Execute after successful operation (sync).

        Parameters
        ----------
        ctx
            Execution context.
        result
            Operation result.
        mw_context
            Context from before_invoke.

        Returns
        -------
        CliResult[Any]
            Possibly modified result.
        """
        ...

    async def after_invoke_async(
        self,
        ctx: ExecutionContext,
        result: CliResult[Any],
        mw_context: dict[str, Any],
    ) -> CliResult[Any]:
        """Execute after successful operation (async).

        Default delegates to sync version.

        Parameters
        ----------
        ctx
            Execution context.
        result
            Operation result.
        mw_context
            Context from before_invoke.

        Returns
        -------
        CliResult[Any]
            Possibly modified result.
        """
        return self.after_invoke(ctx, result, mw_context)

    @abstractmethod
    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_context: dict[str, Any],
    ) -> Exception | None:
        """Handle operation error (sync).

        Parameters
        ----------
        ctx
            Execution context.
        exc
            Exception that occurred.
        mw_context
            Context from before_invoke.

        Returns
        -------
        Exception | None
            Exception to raise, or None to suppress.
        """
        ...

    async def on_error_async(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_context: dict[str, Any],
    ) -> Exception | None:
        """Handle operation error (async).

        Default delegates to sync version.

        Parameters
        ----------
        ctx
            Execution context.
        exc
            Exception that occurred.
        mw_context
            Context from before_invoke.

        Returns
        -------
        Exception | None
            Exception to raise, or None to suppress.
        """
        return self.on_error(ctx, exc, mw_context)


class LoggingMiddleware(Middleware):
    """Log operation execution details."""

    def __init__(self, *, log_params: bool = True) -> None:
        """Initialize logging middleware.

        Parameters
        ----------
        log_params
            Whether to log operation parameters (may contain sensitive data).
        """
        self._log_params = log_params

    def before_invoke(
        self,
        ctx: ExecutionContext,
    ) -> dict[str, Any]:
        """Log operation start.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        dict[str, Any]
            Context with start time.
        """
        extra: dict[str, Any] = {"op_id": ctx.operation_id}
        if self._log_params:
            extra["params"] = ctx.params
        LOG.info("Starting operation", extra=extra)
        return {"start_time": time.monotonic()}

    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: CliResult[Any],
        mw_context: dict[str, Any],
    ) -> CliResult[Any]:
        """Log operation completion.

        Parameters
        ----------
        ctx
            Execution context.
        result
            Operation result.
        mw_context
            Context from before_invoke.

        Returns
        -------
        CliResult[Any]
            Unmodified result.
        """
        _ = self  # Instance method conforming to interface
        duration = time.monotonic() - mw_context["start_time"]
        extra: dict[str, Any] = {
            "op_id": ctx.operation_id,
            "duration_seconds": duration,
            "success": result.success,
        }
        LOG.info("Operation completed", extra=extra)
        return result

    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_context: dict[str, Any],
    ) -> Exception | None:
        """Log operation error.

        Parameters
        ----------
        ctx
            Execution context.
        exc
            Exception that occurred.
        mw_context
            Context from before_invoke.

        Returns
        -------
        Exception
            The original exception.
        """
        _ = self  # Instance method conforming to interface
        duration = time.monotonic() - mw_context.get("start_time", 0)
        extra: dict[str, Any] = {
            "op_id": ctx.operation_id,
            "duration_seconds": duration,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        LOG.error("Operation failed", extra=extra)
        return exc


class MetricsMiddleware(Middleware):
    """Collect operation metrics."""

    def __init__(self) -> None:
        """Initialize the metrics middleware."""
        self._operation_count: dict[str, int] = {}
        self._operation_errors: dict[str, int] = {}
        self._operation_durations: dict[str, list[float]] = {}

    def before_invoke(
        self,
        ctx: ExecutionContext,
    ) -> dict[str, Any]:
        """Record operation start.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        dict[str, Any]
            Context with start time.
        """
        if ctx.operation_id not in self._operation_count:
            self._operation_count[ctx.operation_id] = 0
        return {"start_time": time.monotonic()}

    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: CliResult[Any],
        mw_context: dict[str, Any],
    ) -> CliResult[Any]:
        """Record operation success.

        Parameters
        ----------
        ctx
            Execution context.
        result
            Operation result.
        mw_context
            Context from before_invoke.

        Returns
        -------
        CliResult[Any]
            Unmodified result.
        """
        duration = time.monotonic() - mw_context["start_time"]
        self._operation_count[ctx.operation_id] = self._operation_count.get(ctx.operation_id, 0) + 1
        if ctx.operation_id not in self._operation_durations:
            self._operation_durations[ctx.operation_id] = []
        self._operation_durations[ctx.operation_id].append(duration)
        return result

    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_context: dict[str, Any],
    ) -> Exception | None:
        """Record operation error.

        Parameters
        ----------
        ctx
            Execution context.
        exc
            Exception that occurred.
        mw_context
            Context from before_invoke (unused).

        Returns
        -------
        Exception
            The original exception.
        """
        del mw_context  # Unused
        self._operation_errors[ctx.operation_id] = (
            self._operation_errors.get(ctx.operation_id, 0) + 1
        )
        return exc

    def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics.

        Returns
        -------
        dict[str, Any]
            Metrics summary including counts, errors, and durations.
        """
        return {
            "operation_counts": dict(self._operation_count),
            "operation_errors": dict(self._operation_errors),
            "operation_durations": {
                op_id: {
                    "count": len(durations),
                    "total": sum(durations),
                    "avg": sum(durations) / len(durations) if durations else 0,
                }
                for op_id, durations in self._operation_durations.items()
            },
        }


class TimingMiddleware(Middleware):
    """Track operation timing."""

    def before_invoke(
        self,
        ctx: ExecutionContext,
    ) -> dict[str, Any]:
        """Record start time.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        dict[str, Any]
            Context with start time.
        """
        # Store operation_id for potential use in after_invoke
        _ = (self, ctx)  # Conform to interface
        return {"start_time": time.monotonic(), "started_at": datetime.now(UTC)}

    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: CliResult[Any],
        mw_context: dict[str, Any],
    ) -> CliResult[Any]:
        """Record end time.

        Parameters
        ----------
        ctx
            Execution context.
        result
            Operation result.
        mw_context
            Context from before_invoke.

        Returns
        -------
        CliResult[Any]
            Unmodified result.
        """
        _ = self  # Instance method conforming to interface
        duration = time.monotonic() - mw_context["start_time"]
        ctx.metadata["duration_seconds"] = duration
        ctx.metadata["started_at"] = mw_context["started_at"]
        ctx.metadata["ended_at"] = datetime.now(UTC)
        return result

    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_context: dict[str, Any],
    ) -> Exception | None:
        """Record timing on error.

        Parameters
        ----------
        ctx
            Execution context.
        exc
            Exception that occurred.
        mw_context
            Context from before_invoke.

        Returns
        -------
        Exception
            The original exception.
        """
        _ = self  # Instance method conforming to interface
        duration = time.monotonic() - mw_context["start_time"]
        ctx.metadata["duration_seconds"] = duration
        ctx.metadata["ended_at"] = datetime.now(UTC)
        return exc


class TracingMiddleware(Middleware):
    """OpenTelemetry tracing middleware."""

    def __init__(self) -> None:
        """Initialize tracing middleware."""
        self._spans: dict[str, Any] = {}

    def before_invoke(
        self,
        ctx: ExecutionContext,
    ) -> dict[str, Any]:
        """Start trace span.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        dict[str, Any]
            Context with span.
        """
        if not _HAS_OPENTELEMETRY or otel_trace is None:
            return {}

        tracer = otel_trace.get_tracer("codeintel.cli")
        span = tracer.start_span(f"cli.{ctx.operation_id}")
        span.set_attribute("operation.id", ctx.operation_id)
        span.set_attribute("operation.params_count", len(ctx.params))

        self._spans[ctx.operation_id] = span
        return {"span": span}

    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: CliResult[Any],
        mw_context: dict[str, Any],
    ) -> CliResult[Any]:
        """End trace span.

        Parameters
        ----------
        ctx
            Execution context.
        result
            Operation result.
        mw_context
            Context from before_invoke.

        Returns
        -------
        CliResult[Any]
            Unmodified result.
        """
        span = mw_context.get("span")
        if span is not None:
            span.set_attribute("success", result.success)
            span.end()
            self._spans.pop(ctx.operation_id, None)
        return result

    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_context: dict[str, Any],
    ) -> Exception | None:
        """Record error in span.

        Parameters
        ----------
        ctx
            Execution context.
        exc
            Exception that occurred.
        mw_context
            Context from before_invoke.

        Returns
        -------
        Exception
            The original exception.
        """
        span = mw_context.get("span")
        if span is not None:
            span.record_exception(exc)
            span.set_attribute(key="error", value=True)
            span.end()
            self._spans.pop(ctx.operation_id, None)
        return exc


class ProgressMiddleware(Middleware):
    """Progress reporting middleware."""

    def __init__(
        self,
        callback: Callable[[ProgressEvent], None] | None = None,
    ) -> None:
        """Initialize progress middleware.

        Parameters
        ----------
        callback
            Callback to receive progress events.
        """
        self._callback = callback

    def before_invoke(
        self,
        ctx: ExecutionContext,
    ) -> dict[str, Any]:
        """Emit start progress event.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        dict[str, Any]
            Empty context.
        """
        if self._callback is not None:
            event = ProgressEvent(
                operation_id=ctx.operation_id,
                state=ProgressState.RUNNING,
                progress=0.0,
                message=f"Starting {ctx.operation_id}",
            )
            self._callback(event)
        return {}

    def after_invoke(
        self,
        ctx: ExecutionContext,
        result: CliResult[Any],
        mw_context: dict[str, Any],
    ) -> CliResult[Any]:
        """Emit completion progress event.

        Parameters
        ----------
        ctx
            Execution context.
        result
            Operation result.
        mw_context
            Context from before_invoke (unused).

        Returns
        -------
        CliResult[Any]
            Unmodified result.
        """
        del mw_context  # Unused
        if self._callback is not None:
            state = ProgressState.COMPLETED if result.success else ProgressState.FAILED
            event = ProgressEvent(
                operation_id=ctx.operation_id,
                state=state,
                progress=1.0 if result.success else None,
                message="Completed" if result.success else "Failed",
            )
            self._callback(event)
        return result

    def on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_context: dict[str, Any],
    ) -> Exception | None:
        """Emit failure progress event.

        Parameters
        ----------
        ctx
            Execution context.
        exc
            Exception that occurred.
        mw_context
            Context from before_invoke (unused).

        Returns
        -------
        Exception
            The original exception.
        """
        del mw_context  # Unused
        if self._callback is not None:
            event = ProgressEvent(
                operation_id=ctx.operation_id,
                state=ProgressState.FAILED,
                message=str(exc),
            )
            self._callback(event)
        return exc


@dataclass
class MiddlewareStack:
    """Stack of middleware for unified execution."""

    middleware: list[Middleware] = field(default_factory=list)

    def add(self, mw: Middleware) -> None:
        """Add middleware to the stack.

        Parameters
        ----------
        mw
            Middleware to add.
        """
        self.middleware.append(mw)

    def execute_before(self, ctx: ExecutionContext) -> list[dict[str, Any]]:
        """Execute all before_invoke hooks (sync).

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[dict[str, Any]]
            List of middleware contexts.
        """
        contexts: list[dict[str, Any]] = []
        for mw in self.middleware:
            mw_ctx = mw.before_invoke(ctx)
            contexts.append(mw_ctx)
        return contexts

    async def execute_before_async(self, ctx: ExecutionContext) -> list[dict[str, Any]]:
        """Execute all before_invoke hooks (async).

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        list[dict[str, Any]]
            List of middleware contexts.
        """
        contexts: list[dict[str, Any]] = []
        for mw in self.middleware:
            mw_ctx = await mw.before_invoke_async(ctx)
            contexts.append(mw_ctx)
        return contexts

    def execute_after(
        self,
        ctx: ExecutionContext,
        result: CliResult[Any],
        mw_contexts: list[dict[str, Any]],
    ) -> CliResult[Any]:
        """Execute all after_invoke hooks (sync, reverse order).

        Parameters
        ----------
        ctx
            Execution context.
        result
            Operation result.
        mw_contexts
            Middleware contexts from before_invoke.

        Returns
        -------
        CliResult[Any]
            Final result after all middleware.
        """
        for mw, mw_ctx in zip(reversed(self.middleware), reversed(mw_contexts), strict=False):
            result = mw.after_invoke(ctx, result, mw_ctx)
        return result

    async def execute_after_async(
        self,
        ctx: ExecutionContext,
        result: CliResult[Any],
        mw_contexts: list[dict[str, Any]],
    ) -> CliResult[Any]:
        """Execute all after_invoke hooks (async, reverse order).

        Parameters
        ----------
        ctx
            Execution context.
        result
            Operation result.
        mw_contexts
            Middleware contexts from before_invoke.

        Returns
        -------
        CliResult[Any]
            Final result after all middleware.
        """
        for mw, mw_ctx in zip(reversed(self.middleware), reversed(mw_contexts), strict=False):
            result = await mw.after_invoke_async(ctx, result, mw_ctx)
        return result

    def execute_on_error(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_contexts: list[dict[str, Any]],
    ) -> Exception | None:
        """Execute all on_error hooks (sync, reverse order).

        Parameters
        ----------
        ctx
            Execution context.
        exc
            Exception that occurred.
        mw_contexts
            Middleware contexts from before_invoke.

        Returns
        -------
        Exception | None
            Final exception or None if suppressed.
        """
        final_exc: Exception | None = exc
        for mw, mw_ctx in zip(reversed(self.middleware), reversed(mw_contexts), strict=False):
            with contextlib.suppress(Exception):
                final_exc = mw.on_error(ctx, exc, mw_ctx)
        return final_exc

    async def execute_on_error_async(
        self,
        ctx: ExecutionContext,
        exc: Exception,
        mw_contexts: list[dict[str, Any]],
    ) -> Exception | None:
        """Execute all on_error hooks (async, reverse order).

        Parameters
        ----------
        ctx
            Execution context.
        exc
            Exception that occurred.
        mw_contexts
            Middleware contexts from before_invoke.

        Returns
        -------
        Exception | None
            Final exception or None if suppressed.
        """
        final_exc: Exception | None = exc
        for mw, mw_ctx in zip(reversed(self.middleware), reversed(mw_contexts), strict=False):
            with contextlib.suppress(Exception):
                final_exc = await mw.on_error_async(ctx, exc, mw_ctx)
        return final_exc

    @contextmanager
    def wrap(
        self,
        ctx: ExecutionContext,
    ) -> Iterator[list[dict[str, Any]]]:
        """Wrap operation execution with middleware (sync).

        Parameters
        ----------
        ctx
            Execution context.

        Yields
        ------
        list[dict[str, Any]]
            Middleware contexts.
        """
        mw_contexts = self.execute_before(ctx)
        try:
            yield mw_contexts
        except Exception as exc:
            final_exc = self.execute_on_error(ctx, exc, mw_contexts)
            if final_exc is not None:
                raise final_exc from exc
        # Note: after_invoke is called by the caller after getting result

    @asynccontextmanager
    async def wrap_async(
        self,
        ctx: ExecutionContext,
    ) -> AsyncGenerator[list[dict[str, Any]]]:
        """Wrap operation execution with middleware (async).

        Parameters
        ----------
        ctx
            Execution context.

        Yields
        ------
        list[dict[str, Any]]
            Middleware contexts.
        """
        mw_contexts = await self.execute_before_async(ctx)
        try:
            yield mw_contexts
        except Exception as exc:
            final_exc = await self.execute_on_error_async(ctx, exc, mw_contexts)
            if final_exc is not None:
                raise final_exc from exc
        # Note: after_invoke is called by the caller after getting result


# Global middleware stack
_MIDDLEWARE_STACK: MiddlewareStack | None = None


def get_middleware_stack() -> MiddlewareStack:
    """Get the global middleware stack.

    Returns
    -------
    MiddlewareStack
        Global middleware stack instance.
    """
    global _MIDDLEWARE_STACK  # noqa: PLW0603
    if _MIDDLEWARE_STACK is None:
        _MIDDLEWARE_STACK = MiddlewareStack()
    return _MIDDLEWARE_STACK


def configure_default_middleware() -> None:
    """Configure default middleware (logging)."""
    stack = get_middleware_stack()
    if not any(isinstance(mw, LoggingMiddleware) for mw in stack.middleware):
        stack.add(LoggingMiddleware())


__all__ = [
    "LoggingMiddleware",
    "MetricsMiddleware",
    "Middleware",
    "MiddlewareStack",
    "ProgressMiddleware",
    "TimingMiddleware",
    "TracingMiddleware",
    "configure_default_middleware",
    "get_middleware_stack",
]
