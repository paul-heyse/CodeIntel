"""Validation middleware for operations.

Validates operation parameters before execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.operations.middleware.base import BaseMiddleware

if TYPE_CHECKING:
    from codeintel.operations.base import OperationSpec
    from codeintel.operations.context import OpContext


class ValidationError(Exception):
    """Raised when parameter validation fails."""

    def __init__(self, message: str, param: str | None = None) -> None:
        self.param = param
        super().__init__(message)


@runtime_checkable
class Validatable(Protocol):
    """Protocol for params with a validate method."""

    def validate(self) -> None:
        """Validate the parameters."""
        ...


class ValidationMiddleware(BaseMiddleware):
    """Validate operation parameters before execution.

    Calls the params.validate() method if available, raising
    ValidationError on failure.

    Example
    -------
    >>> @dataclass(frozen=True)
    ... class MyParams:
    ...     name: str
    ...
    ...     def validate(self) -> None:
    ...         if not self.name:
    ...             raise ValidationError("name is required", param="name")
    """

    def before(
        self,
        spec: OperationSpec,
        params: object,
        ctx: OpContext,
    ) -> None:
        """Validate parameters before operation runs.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.
        ctx
            Operation context.

        Notes
        -----
        If params implements Validatable protocol (has a validate() method),
        it will be called and may raise ValidationError if validation fails.
        """
        _ = (self, spec, ctx)  # Acknowledge for signature compatibility

        # Call validate() if params implements the Validatable protocol
        if isinstance(params, Validatable):
            params.validate()


__all__ = [
    "ValidationError",
    "ValidationMiddleware",
]
