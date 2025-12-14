"""Query builder protocol and implementation.

This module provides a protocol and base implementation
for building queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Self, runtime_checkable


@runtime_checkable
class QueryBuilderProtocol(Protocol):
    """Protocol for query builders.

    Query builders provide a fluent API for constructing queries.
    """

    def where(self, field: str, value: object) -> Self:
        """Add a where clause.

        Parameters
        ----------
        field
            Field name.
        value
            Value to match.

        Returns
        -------
        Self
            Self for chaining.
        """
        ...

    def order_by(self, field: str, *, descending: bool = False) -> Self:
        """Add an order by clause.

        Parameters
        ----------
        field
            Field to order by.
        descending
            Whether to order descending.

        Returns
        -------
        Self
            Self for chaining.
        """
        ...

    def limit(self, count: int) -> Self:
        """Limit the number of results.

        Parameters
        ----------
        count
            Maximum results.

        Returns
        -------
        Self
            Self for chaining.
        """
        ...

    def build(self) -> object:
        """Build the query.

        Returns
        -------
        object
            The constructed query.
        """
        ...


@dataclass
class QueryBuilder:
    """Simple query builder implementation.

    Builds a query as a dictionary of parameters.

    Examples
    --------
    >>> query = (
    ...     QueryBuilder("users")
    ...     .where("status", "active")
    ...     .order_by("created_at", descending=True)
    ...     .limit(10)
    ...     .build()
    ... )
    """

    table: str
    _wheres: list[tuple[str, object]] = field(default_factory=list)
    _order_by: tuple[str, bool] | None = None
    _limit: int | None = None

    def where(self, field_name: str, value: object) -> Self:
        """Add a where clause.

        Parameters
        ----------
        field_name
            Field name.
        value
            Value to match.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._wheres.append((field_name, value))
        return self

    def order_by(self, field_name: str, *, descending: bool = False) -> Self:
        """Add an order by clause.

        Parameters
        ----------
        field_name
            Field to order by.
        descending
            Whether to order descending.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._order_by = (field_name, descending)
        return self

    def limit(self, count: int) -> Self:
        """Limit the number of results.

        Parameters
        ----------
        count
            Maximum results.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._limit = count
        return self

    def build(self) -> dict[str, object]:
        """Build the query as a dictionary.

        Returns
        -------
        dict[str, object]
            Query parameters.
        """
        result: dict[str, object] = {
            "table": self.table,
            "filters": dict(self._wheres),
        }

        if self._order_by is not None:
            result["order_by"] = self._order_by[0]
            result["descending"] = self._order_by[1]

        if self._limit is not None:
            result["limit"] = self._limit

        return result


__all__ = [
    "QueryBuilder",
    "QueryBuilderProtocol",
]
