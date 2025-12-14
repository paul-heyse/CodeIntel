"""Unified pagination types.

This module provides standardized pagination types that work
across all repository implementations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Pagination:
    """Pagination parameters for list operations.

    Attributes
    ----------
    limit
        Maximum number of items to return.
    offset
        Number of items to skip.
    order_by
        Field to order by.
    descending
        If True, order descending.

    Examples
    --------
    >>> pagination = Pagination(limit=10, offset=20)
    >>> pagination.page_number
    2
    """

    limit: int = 50
    offset: int = 0
    order_by: str | None = None
    descending: bool = False

    def __post_init__(self) -> None:
        """Validate pagination parameters.

        Raises
        ------
        ValueError
            If limit or offset is negative.
        """
        if self.limit < 0:
            msg = "limit must be non-negative"
            raise ValueError(msg)
        if self.offset < 0:
            msg = "offset must be non-negative"
            raise ValueError(msg)

    @property
    def page_number(self) -> int:
        """Calculate the current page number (0-indexed).

        Returns
        -------
        int
            Current page number.
        """
        if self.limit == 0:
            return 0
        return self.offset // self.limit

    @classmethod
    def for_page(cls, page: int, page_size: int = 50) -> Pagination:
        """Create pagination for a specific page.

        Parameters
        ----------
        page
            Page number (0-indexed).
        page_size
            Number of items per page.

        Returns
        -------
        Pagination
            Pagination for the specified page.

        Examples
        --------
        >>> pagination = Pagination.for_page(2, page_size=10)
        >>> pagination.offset
        20
        """
        return cls(limit=page_size, offset=page * page_size)

    def next_page(self) -> Pagination:
        """Return pagination for the next page.

        Returns
        -------
        Pagination
            Pagination for the next page.
        """
        return Pagination(
            limit=self.limit,
            offset=self.offset + self.limit,
            order_by=self.order_by,
            descending=self.descending,
        )

    def with_limit(self, limit: int) -> Pagination:
        """Return a copy with a different limit.

        Parameters
        ----------
        limit
            New limit value.

        Returns
        -------
        Pagination
            New pagination with updated limit.
        """
        return Pagination(
            limit=limit,
            offset=self.offset,
            order_by=self.order_by,
            descending=self.descending,
        )

    def with_order(self, order_by: str, *, descending: bool = False) -> Pagination:
        """Return a copy with ordering.

        Parameters
        ----------
        order_by
            Field to order by.
        descending
            If True, order descending.

        Returns
        -------
        Pagination
            New pagination with ordering.
        """
        return Pagination(
            limit=self.limit,
            offset=self.offset,
            order_by=order_by,
            descending=descending,
        )


@dataclass
class PagedResult[T]:
    """Result of a paginated query.

    Attributes
    ----------
    items
        The items in this page.
    total
        Total number of items matching the query.
    limit
        Maximum number of items per page.
    offset
        Number of items skipped.
    truncated
        Whether more items exist beyond this page.

    Examples
    --------
    >>> result = PagedResult(items=[1, 2, 3], total=100, limit=10, offset=0)
    >>> result.has_more
    True
    >>> result.total_pages
    10
    """

    items: list[T] = field(default_factory=list)
    total: int | None = None
    limit: int = 50
    offset: int = 0
    truncated: bool = False

    @property
    def count(self) -> int:
        """Return the number of items in this page.

        Returns
        -------
        int
            Number of items.
        """
        return len(self.items)

    @property
    def has_more(self) -> bool:
        """Check if more items exist beyond this page.

        Returns
        -------
        bool
            True if more items exist.
        """
        if self.truncated:
            return True
        if self.total is not None:
            return self.offset + len(self.items) < self.total
        return False

    @property
    def total_pages(self) -> int | None:
        """Calculate total number of pages.

        Returns
        -------
        int | None
            Total pages, or None if total is unknown.
        """
        if self.total is None or self.limit == 0:
            return None
        return (self.total + self.limit - 1) // self.limit

    @property
    def page_number(self) -> int:
        """Return the current page number (0-indexed).

        Returns
        -------
        int
            Current page number.
        """
        if self.limit == 0:
            return 0
        return self.offset // self.limit

    @property
    def is_empty(self) -> bool:
        """Check if the result is empty.

        Returns
        -------
        bool
            True if no items.
        """
        return len(self.items) == 0

    def map[U](self, func: Callable[[T], U]) -> PagedResult[U]:
        """Transform items using a function.

        Parameters
        ----------
        func
            Function to apply to each item.

        Returns
        -------
        PagedResult[U]
            New result with transformed items.
        """
        return PagedResult(
            items=[func(item) for item in self.items],
            total=self.total,
            limit=self.limit,
            offset=self.offset,
            truncated=self.truncated,
        )

    def __iter__(self) -> Iterator[T]:
        """Iterate over items in this page.

        Returns
        -------
        Iterator[T]
            Iterator over items.
        """
        return iter(self.items)

    def __len__(self) -> int:
        """Return the number of items in this page.

        Returns
        -------
        int
            Number of items.
        """
        return len(self.items)

    @classmethod
    def empty(cls, *, limit: int = 50) -> PagedResult[T]:
        """Create an empty result.

        Parameters
        ----------
        limit
            Page size for the empty result.

        Returns
        -------
        PagedResult[T]
            Empty paged result.
        """
        return cls(items=[], total=0, limit=limit, offset=0)

    @classmethod
    def from_items(
        cls,
        items: list[T],
        *,
        limit: int | None = None,
        detect_truncation: bool = True,
    ) -> PagedResult[T]:
        """Create result from a list of items with truncation detection.

        If limit is provided and items length exceeds it, the result
        is truncated and marked as such.

        Parameters
        ----------
        items
            List of items.
        limit
            Optional limit to apply.
        detect_truncation
            If True, detect truncation when items exceed limit.

        Returns
        -------
        PagedResult[T]
            Paged result.
        """
        if limit is not None and detect_truncation and len(items) > limit:
            return cls(
                items=items[:limit],
                total=None,
                limit=limit,
                offset=0,
                truncated=True,
            )
        return cls(
            items=items,
            total=len(items),
            limit=limit or len(items),
            offset=0,
            truncated=False,
        )


__all__ = [
    "PagedResult",
    "Pagination",
]
