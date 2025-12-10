"""Logging middleware for operations.

Provides structured logging of operation execution with operation_id,
duration, and success/failure status.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeintel.operations.middleware.base import BaseMiddleware

if TYPE_CHECKING:
    from codeintel.operations.base import OperationSpec
    from codeintel.operations.context import OpContext
    from codeintel.operations.result import Result


LOG = logging.getLogger(__name__)


class LoggingMiddleware(BaseMiddleware):
    """Add structured logging to operation execution.

    Logs operation start, completion, and errors with relevant context.

    Example
    -------
    >>> middleware = LoggingMiddleware()
    >>> # Logs: "Starting operation jobs.list"
    >>> # Logs: "Completed jobs.list in 0.05s (success=True)"
    """

    def before(
        self,
        spec: OperationSpec,
        params: object,
        ctx: OpContext,
    ) -> None:
        """Log operation start.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.
        ctx
            Operation context.
        """
        _ = (self, params, ctx)  # Acknowledge for signature compatibility
        LOG.info(
            "Starting operation %s",
            spec.operation_id,
            extra={
                "operation_id": spec.operation_id,
                "group": spec.group,
            },
        )

    def after(
        self,
        spec: OperationSpec,
        params: object,
        ctx: OpContext,
        result: Result[object],
        duration: float,
    ) -> None:
        """Log operation completion.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.
        ctx
            Operation context.
        result
            Operation result.
        duration
            Execution duration in seconds.
        """
        _ = (self, params, ctx)  # Acknowledge for signature compatibility
        LOG.info(
            "Completed %s in %.3fs (success=%s)",
            spec.operation_id,
            duration,
            result.success,
            extra={
                "operation_id": spec.operation_id,
                "duration_seconds": duration,
                "success": result.success,
            },
        )

    def on_error(
        self,
        spec: OperationSpec,
        params: object,
        ctx: OpContext,
        error: Exception,
        duration: float,
    ) -> None:
        """Log operation error.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.
        ctx
            Operation context.
        error
            The exception raised.
        duration
            Execution duration in seconds.
        """
        _ = (self, params, ctx)  # Acknowledge for signature compatibility
        LOG.error(
            "Failed %s after %.3fs: %s (%s)",
            spec.operation_id,
            duration,
            error,
            type(error).__name__,
            extra={
                "operation_id": spec.operation_id,
                "duration_seconds": duration,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )


__all__ = [
    "LoggingMiddleware",
]
