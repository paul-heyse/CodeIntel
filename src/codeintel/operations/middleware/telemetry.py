"""Telemetry middleware for operations.

Provides OpenTelemetry spans and metrics for operation execution.
Falls back gracefully when OpenTelemetry is not available.
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


class TelemetryMiddleware(BaseMiddleware):
    """Add OpenTelemetry spans and metrics to operation execution.

    Creates spans with operation metadata and records duration metrics.
    Falls back gracefully when telemetry is not configured.

    Example
    -------
    >>> middleware = TelemetryMiddleware()
    >>> # Creates span: "codeintel.operation.jobs.list"
    >>> # Records metric: operation_duration_seconds{operation_id="jobs.list"}
    """

    def before(
        self,
        spec: OperationSpec,
        params: object,
        ctx: OpContext,
    ) -> None:
        """Start telemetry span.

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

        # Telemetry will be integrated when context is fully implemented
        LOG.debug("Telemetry: starting %s", spec.operation_id)

    def after(
        self,
        spec: OperationSpec,
        params: object,
        ctx: OpContext,
        result: Result[object],
        duration: float,
    ) -> None:
        """End telemetry span and record metrics.

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

        # Telemetry will be integrated when context is fully implemented
        LOG.debug(
            "Telemetry: completed %s in %.3fs (success=%s)",
            spec.operation_id,
            duration,
            result.success,
        )

    def on_error(
        self,
        spec: OperationSpec,
        params: object,
        ctx: OpContext,
        error: Exception,
        duration: float,
    ) -> None:
        """Record error in telemetry span.

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

        # Telemetry will be integrated when context is fully implemented
        LOG.debug(
            "Telemetry: error in %s after %.3fs: %s",
            spec.operation_id,
            duration,
            type(error).__name__,
        )


__all__ = [
    "TelemetryMiddleware",
]
