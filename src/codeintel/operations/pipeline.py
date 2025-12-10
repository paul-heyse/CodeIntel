"""Operation execution pipeline.

The pipeline executes operations with middleware for cross-cutting concerns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.operations.middleware.base import BaseMiddleware
from codeintel.operations.middleware.errors import ErrorHandlingMiddleware
from codeintel.operations.middleware.logging import LoggingMiddleware
from codeintel.operations.middleware.telemetry import TelemetryMiddleware
from codeintel.operations.middleware.validation import ValidationMiddleware
from codeintel.operations.result import Result

if TYPE_CHECKING:
    from codeintel.operations.base import Operation, OperationSpec
    from codeintel.operations.context import OpContext


@dataclass
class OperationPipeline:
    """Execute operations with middleware.

    The pipeline wraps operation execution with before/after/on_error
    hooks from each middleware in order.

    Parameters
    ----------
    middleware
        List of middleware to apply.
    error_handler
        Middleware for converting exceptions to Results.

    Example
    -------
    >>> from codeintel.operations.pipeline import OperationPipeline
    >>> pipeline = get_default_pipeline()
    >>> result = pipeline.execute(operation, params, ctx, spec)
    """

    middleware: list[BaseMiddleware] = field(default_factory=list)
    error_handler: ErrorHandlingMiddleware = field(default_factory=ErrorHandlingMiddleware)

    def execute[P, R](
        self,
        operation: Operation[P, R],
        params: P,
        ctx: OpContext,
        spec: OperationSpec,
    ) -> Result[R]:
        """Execute an operation with middleware.

        Parameters
        ----------
        operation
            The operation instance to execute.
        params
            Operation parameters.
        ctx
            Operation context.
        spec
            Operation specification.

        Returns
        -------
        Result[R]
            Operation result (success or failure).
        """
        # Set operation ID on context
        ctx = ctx.with_operation(spec.operation_id)

        # Record start time
        start_time = time.monotonic()

        try:
            # Run before hooks
            for mw in self.middleware:
                mw.before(spec, params, ctx)

            # Execute the operation
            result = operation.execute(params, ctx)

        except Exception as exc:  # noqa: BLE001 - pipeline must handle arbitrary operation failures
            # Calculate duration
            duration = time.monotonic() - start_time

            # Run on_error hooks
            for mw in self.middleware:
                mw.on_error(spec, params, ctx, exc, duration)

            # Convert exception to Result
            return self.error_handler.handle_exception(spec, params, ctx, exc)

        else:
            # Calculate duration
            duration = time.monotonic() - start_time

            # Run after hooks
            for mw in self.middleware:
                mw.after(spec, params, ctx, result, duration)

            return result


# Default pipeline configuration
_DEFAULT_PIPELINE: OperationPipeline | None = None


def get_default_pipeline() -> OperationPipeline:
    """Get the default pipeline with standard middleware.

    Returns a pipeline configured with:
    - LoggingMiddleware
    - TelemetryMiddleware
    - ValidationMiddleware
    - ErrorHandlingMiddleware

    Returns
    -------
    OperationPipeline
        The default pipeline.
    """
    global _DEFAULT_PIPELINE  # noqa: PLW0603

    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = OperationPipeline(
            middleware=[
                LoggingMiddleware(),
                TelemetryMiddleware(),
                ValidationMiddleware(),
            ],
            error_handler=ErrorHandlingMiddleware(),
        )

    return _DEFAULT_PIPELINE


def create_pipeline(
    *,
    include_logging: bool = True,
    include_telemetry: bool = True,
    include_validation: bool = True,
) -> OperationPipeline:
    """Create a customized pipeline.

    Parameters
    ----------
    include_logging
        Include logging middleware.
    include_telemetry
        Include telemetry middleware.
    include_validation
        Include validation middleware.

    Returns
    -------
    OperationPipeline
        Customized pipeline.
    """
    middleware: list[BaseMiddleware] = []

    if include_logging:
        middleware.append(LoggingMiddleware())

    if include_telemetry:
        middleware.append(TelemetryMiddleware())

    if include_validation:
        middleware.append(ValidationMiddleware())

    return OperationPipeline(
        middleware=middleware,
        error_handler=ErrorHandlingMiddleware(),
    )


__all__ = [
    "OperationPipeline",
    "create_pipeline",
    "get_default_pipeline",
]
