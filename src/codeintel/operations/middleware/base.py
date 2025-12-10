"""Base middleware protocol for operations.

Middleware wraps operation execution to add cross-cutting concerns
like logging, metrics, validation, and error handling.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from codeintel.operations.base import OperationSpec
    from codeintel.operations.context import OpContext
    from codeintel.operations.result import Result


class OperationMiddleware(Protocol):
    """Protocol for operation middleware.

    Middleware is called before and after operation execution,
    and on errors. Each middleware can modify context, params,
    or results.

    Example
    -------
    >>> class TimingMiddleware:
    ...     def before(self, spec, params, ctx):
    ...         print(f"Starting {spec.operation_id}")
    ...
    ...     def after(self, spec, params, ctx, result, duration):
    ...         print(f"Completed in {duration:.2f}s")
    ...
    ...     def on_error(self, spec, params, ctx, error, duration):
    ...         print(f"Failed after {duration:.2f}s")
    """

    @abstractmethod
    def before(
        self,
        spec: OperationSpec,
        params: object,
        ctx: OpContext,
    ) -> None:
        """Execute before operation runs.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.
        ctx
            Operation context.
        """
        ...

    @abstractmethod
    def after(
        self,
        spec: OperationSpec,
        params: object,
        ctx: OpContext,
        result: Result[object],
        duration: float,
    ) -> None:
        """Execute after operation completes successfully.

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
        ...

    @abstractmethod
    def on_error(
        self,
        spec: OperationSpec,
        params: object,
        ctx: OpContext,
        error: Exception,
        duration: float,
    ) -> None:
        """Execute when operation raises an exception.

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
        ...


class BaseMiddleware:
    """Base class for middleware with default no-op implementations.

    Subclass this and override only the methods you need.
    """

    def before(
        self,
        spec: OperationSpec,
        params: object,
        ctx: OpContext,
    ) -> None:
        """Execute before operation runs.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.
        ctx
            Operation context.
        """
        # Acknowledge parameters for signature compatibility
        _ = (self, spec, params, ctx)

    def after(
        self,
        spec: OperationSpec,
        params: object,
        ctx: OpContext,
        result: Result[object],
        duration: float,
    ) -> None:
        """Execute after operation completes successfully.

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
        # Acknowledge parameters for signature compatibility
        _ = (self, spec, params, ctx, result, duration)

    def on_error(
        self,
        spec: OperationSpec,
        params: object,
        ctx: OpContext,
        error: Exception,
        duration: float,
    ) -> None:
        """Execute when operation raises an exception.

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
        # Acknowledge parameters for signature compatibility
        _ = (self, spec, params, ctx, error, duration)


__all__ = [
    "BaseMiddleware",
    "OperationMiddleware",
]
