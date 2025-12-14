"""Result protocol definitions.

This module provides the ResultProtocol for types that represent
operation outcomes with success/failure semantics.
"""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable


@runtime_checkable
class ResultProtocol(Protocol):
    """Protocol for operation result types.

    Implementations provide consistent success/failure/duration semantics
    across all modules.

    Examples
    --------
    >>> class MyResult:
    ...     @property
    ...     def success(self) -> bool:
    ...         return True
    ...     @property
    ...     def error(self) -> str | None:
    ...         return None
    ...     @property
    ...     def duration_s(self) -> float:
    ...         return 0.0
    """

    @property
    def success(self) -> bool:
        """Return whether the operation succeeded.

        Returns
        -------
        bool
            True if operation completed successfully.
        """
        ...

    @property
    def error(self) -> str | None:
        """Return error message if operation failed.

        Returns
        -------
        str | None
            Error message, or None if operation succeeded.
        """
        ...

    @property
    def duration_s(self) -> float:
        """Return operation duration in seconds.

        Returns
        -------
        float
            Duration in seconds.
        """
        ...

    @classmethod
    def ok(cls, **kwargs: object) -> Self:
        """Create a success result.

        Parameters
        ----------
        **kwargs
            Additional result-specific fields.

        Returns
        -------
        Self
            Success result instance.
        """
        ...

    @classmethod
    def fail(cls, error: str, **kwargs: object) -> Self:
        """Create a failure result.

        Parameters
        ----------
        error
            Error message describing the failure.
        **kwargs
            Additional result-specific fields.

        Returns
        -------
        Self
            Failure result instance.
        """
        ...


__all__ = [
    "ResultProtocol",
]
