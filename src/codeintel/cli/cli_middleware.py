"""Middleware pattern for CLI operation execution.

Middleware components intercept operation execution to provide
cross-cutting concerns like logging, metrics, and tracing.
"""

from __future__ import annotations

import contextlib
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

LOG = logging.getLogger(__name__)


class OperationMiddleware(ABC):
    """Base class for operation execution middleware."""

    @abstractmethod
    def before_invoke(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute before operation invocation.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        dict[str, Any]
            Context data to pass to after_invoke.
        """
        ...

    @abstractmethod
    def after_invoke(
        self,
        op_id: str,
        result: object,
        context: dict[str, Any],
    ) -> None:
        """Execute after successful operation invocation.

        Parameters
        ----------
        op_id
            Operation identifier.
        result
            Operation result.
        context
            Context from before_invoke.
        """
        ...

    @abstractmethod
    def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """Execute on operation error.

        Parameters
        ----------
        op_id
            Operation identifier.
        exc
            Exception that occurred.
        context
            Context from before_invoke.
        """
        ...


class LoggingMiddleware(OperationMiddleware):
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
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Log operation start.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        dict[str, Any]
            Context with start time.
        """
        extra: dict[str, Any] = {"op_id": op_id}
        if self._log_params:
            extra["params"] = params
        LOG.info("Starting operation", extra=extra)
        return {"start_time": time.monotonic()}

    def after_invoke(
        self,
        op_id: str,
        result: object,
        context: dict[str, Any],
    ) -> None:
        """Log operation completion.

        Parameters
        ----------
        op_id
            Operation identifier.
        result
            Operation result.
        context
            Context from before_invoke.
        """
        del result  # Unused but required by protocol
        duration = time.monotonic() - context["start_time"]
        extra: dict[str, Any] = {"op_id": op_id, "duration_seconds": duration}
        if self._log_params:
            extra["middleware"] = "logging"
        LOG.info("Operation completed", extra=extra)

    def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """Log operation error.

        Parameters
        ----------
        op_id
            Operation identifier.
        exc
            Exception that occurred.
        context
            Context from before_invoke.
        """
        duration = time.monotonic() - context.get("start_time", 0)
        extra: dict[str, Any] = {
            "op_id": op_id,
            "duration_seconds": duration,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        if self._log_params:
            extra["middleware"] = "logging"
        LOG.error("Operation failed", extra=extra)


class MetricsMiddleware(OperationMiddleware):
    """Collect operation metrics.

    This middleware tracks operation counts, errors, and durations
    for observability and monitoring purposes.
    """

    def __init__(self) -> None:
        """Initialize the metrics middleware."""
        self._operation_count: dict[str, int] = {}
        self._operation_errors: dict[str, int] = {}
        self._operation_durations: dict[str, list[float]] = {}

    def before_invoke(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Record operation start.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        dict[str, Any]
            Context with start time and operation.
        """
        del params  # Unused but required by protocol
        # Initialize tracking for this operation if needed
        if op_id not in self._operation_count:
            self._operation_count[op_id] = 0
        return {"start_time": time.monotonic()}

    def after_invoke(
        self,
        op_id: str,
        result: object,
        context: dict[str, Any],
    ) -> None:
        """Record operation success.

        Parameters
        ----------
        op_id
            Operation identifier.
        result
            Operation result.
        context
            Context from before_invoke.
        """
        del result  # Unused but required by protocol
        duration = time.monotonic() - context["start_time"]
        self._operation_count[op_id] = self._operation_count.get(op_id, 0) + 1
        if op_id not in self._operation_durations:
            self._operation_durations[op_id] = []
        self._operation_durations[op_id].append(duration)

    def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """Record operation error.

        Parameters
        ----------
        op_id
            Operation identifier.
        exc
            Exception that occurred.
        context
            Context from before_invoke.
        """
        del exc, context  # Unused but required by protocol
        self._operation_errors[op_id] = self._operation_errors.get(op_id, 0) + 1

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


@dataclass
class MiddlewareStack:
    """Stack of middleware to execute around operations."""

    middleware: list[OperationMiddleware] = field(default_factory=list)

    def add(self, mw: OperationMiddleware) -> None:
        """Add middleware to the stack.

        Parameters
        ----------
        mw
            Middleware to add.
        """
        self.middleware.append(mw)

    @contextmanager
    def wrap(self, op_id: str, params: dict[str, Any]) -> Iterator[None]:
        """Wrap operation execution with middleware.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Yields
        ------
        None
            Control to the wrapped operation.
        """
        contexts: list[dict[str, Any]] = []

        # Before hooks
        for mw in self.middleware:
            ctx = mw.before_invoke(op_id, params)
            contexts.append(ctx)

        try:
            yield
        except Exception as exc:
            # Error hooks (reverse order)
            for mw, ctx in zip(reversed(self.middleware), reversed(contexts), strict=False):
                with contextlib.suppress(Exception):
                    mw.on_error(op_id, exc, ctx)
            raise
        else:
            # After hooks (reverse order)
            for mw, ctx in zip(reversed(self.middleware), reversed(contexts), strict=False):
                mw.after_invoke(op_id, None, ctx)


# Global middleware stack
_MIDDLEWARE_STACK = MiddlewareStack()


def get_middleware_stack() -> MiddlewareStack:
    """Get the global middleware stack.

    Returns
    -------
    MiddlewareStack
        Global middleware stack instance.
    """
    return _MIDDLEWARE_STACK


def configure_default_middleware() -> None:
    """Configure default middleware (logging)."""
    stack = get_middleware_stack()
    # Only add if not already configured
    if not any(isinstance(mw, LoggingMiddleware) for mw in stack.middleware):
        stack.add(LoggingMiddleware())


__all__ = [
    "LoggingMiddleware",
    "MetricsMiddleware",
    "MiddlewareStack",
    "OperationMiddleware",
    "configure_default_middleware",
    "get_middleware_stack",
]
