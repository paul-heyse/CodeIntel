"""Base result types for port operations.

This module defines protocol types for query and batch operation results.
Package-specific implementations can extend these protocols with additional
fields while maintaining compatibility with generic result handling.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseQueryResult(Protocol):
    """Protocol for query result types.

    Implementations must provide a row_count attribute indicating
    the number of rows returned by the query.

    Attributes
    ----------
    row_count
        Number of rows in the result set.
    """

    @property
    def row_count(self) -> int:
        """Return the number of rows in the result."""
        ...


@runtime_checkable
class BaseBatchResult(Protocol):
    """Protocol for batch operation result types.

    Implementations must provide a success indicator and an
    affected row count.

    Attributes
    ----------
    rows_affected
        Number of rows affected by the batch operation.
    """

    @property
    def rows_affected(self) -> int:
        """Return the number of rows affected."""
        ...


__all__ = [
    "BaseBatchResult",
    "BaseQueryResult",
]

