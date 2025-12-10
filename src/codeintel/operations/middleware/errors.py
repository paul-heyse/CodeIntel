"""Error handling middleware for operations.

Converts exceptions to Result.fail() with ProblemDetail.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from codeintel.operations.errors.factory import (
    fail_internal,
    fail_validation,
)
from codeintel.operations.middleware.base import BaseMiddleware
from codeintel.operations.middleware.validation import ValidationError
from codeintel.operations.result import Result

if TYPE_CHECKING:
    from codeintel.operations.base import OperationSpec
    from codeintel.operations.context import OpContext


LOG = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseMiddleware):
    """Convert exceptions to Result.fail() with structured errors.

    This middleware is special - it modifies the pipeline execution
    to catch exceptions and convert them to failed Results.

    Supported exception types:
    - ValidationError -> validation error

    Example
    -------
    >>> middleware = ErrorHandlingMiddleware()
    >>> # ValidationError -> Result.fail(validation_error(...))
    >>> # Exception -> Result.fail(internal_error(...))
    """

    def handle_exception[P, R](
        self,
        spec: OperationSpec,
        params: P,
        ctx: OpContext,
        error: Exception,
    ) -> Result[R]:
        """Convert an exception to a failed Result.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.
        ctx
            Operation context.
        error
            The exception to convert.

        Returns
        -------
        Result[R]
            Failed result with appropriate error.
        """
        _ = (self, params, ctx)  # Acknowledge for signature compatibility

        if isinstance(error, ValidationError):
            return cast("Result[R]", fail_validation(str(error), param=error.param))

        LOG.error(
            "Unhandled exception in %s: %s (%s)",
            spec.operation_id,
            error,
            type(error).__name__,
        )
        return cast(
            "Result[R]",
            fail_internal(
                str(error),
                operation_id=spec.operation_id,
                cause=error,
            ),
        )


__all__ = [
    "ErrorHandlingMiddleware",
]
