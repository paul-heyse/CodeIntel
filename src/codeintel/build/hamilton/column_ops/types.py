"""Protocol types for column-level feature operations."""

from __future__ import annotations

from typing import Protocol


class Column(Protocol):
    """Column-like object supporting arithmetic operations."""

    def __add__(self, other: object, /) -> Column:
        """Return a column representing addition."""
        ...

    def __radd__(self, other: object, /) -> Column:
        """Return a column representing right-side addition."""
        ...

    def __sub__(self, other: object, /) -> Column:
        """Return a column representing subtraction."""
        ...

    def __rsub__(self, other: object, /) -> Column:
        """Return a column representing right-side subtraction."""
        ...

    def __mul__(self, other: object, /) -> Column:
        """Return a column representing multiplication."""
        ...

    def __rmul__(self, other: object, /) -> Column:
        """Return a column representing right-side multiplication."""
        ...

    def __truediv__(self, other: object, /) -> Column:
        """Return a column representing true division."""
        ...

    def __rtruediv__(self, other: object, /) -> Column:
        """Return a column representing right-side true division."""
        ...


__all__ = ["Column"]
