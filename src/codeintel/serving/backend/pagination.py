"""Pagination utilities and safety limits for serving backends.

This module provides:
- Standardized pagination handling (PaginatedFetch, paginate_items)
- Limit/offset clamping with messaging (clamp_limit, clamp_offset)
- Backend configuration (BackendLimits)

Every bounded operation should use these utilities to ensure consistent
truncation detection and safe limit handling across all serving endpoints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from codeintel.serving import domain_models as dm

Message = dm.Message
ResponseMeta = dm.ResponseMeta

# =============================================================================
# Backend Configuration
# =============================================================================


@dataclass(frozen=True)
class BackendLimits:
    """
    Safety limits applied uniformly across backends.

    Parameters
    ----------
    default_limit
        Default number of rows when no limit is specified.
    max_rows_per_call
        Maximum rows allowed for any single call.
    """

    default_limit: int = 50
    max_rows_per_call: int = 500

    @classmethod
    def from_config(cls, cfg: object) -> BackendLimits:
        """
        Build limits from configuration objects.

        Parameters
        ----------
        cfg
            Object with optional ``default_limit`` and ``max_rows_per_call`` attributes.

        Returns
        -------
        BackendLimits
            Limits derived from the provided configuration.
        """
        default = getattr(cfg, "default_limit", cls.default_limit)
        maximum = getattr(cfg, "max_rows_per_call", cls.max_rows_per_call)
        return cls(default_limit=int(default), max_rows_per_call=int(maximum))


# =============================================================================
# Pagination Types
# =============================================================================


@dataclass(frozen=True)
class PaginatedFetch[T]:
    """
    Result of a paginated fetch operation with metadata.

    Encapsulates the items returned plus pagination metadata that should be
    propagated to response envelopes. This ensures consistent truncation
    handling across all list endpoints.

    Parameters
    ----------
    items
        The fetched items (may be truncated if limit was applied).
    applied_limit
        The effective limit used for the query (None if unbounded).
    truncated
        Whether more items exist beyond the applied limit.
    messages
        Any messages generated during pagination (e.g., limit clamping warnings).

    Examples
    --------
    >>> result = PaginatedFetch(items=[1, 2, 3], applied_limit=3, truncated=True)
    >>> result.to_response_meta().truncated
    True
    """

    items: list[T]
    applied_limit: int | None
    truncated: bool
    messages: list[Message] = field(default_factory=list)

    def to_response_meta(self) -> ResponseMeta:
        """
        Convert pagination state to a ResponseMeta for response envelopes.

        Returns
        -------
        ResponseMeta
            Metadata with applied_limit, truncated, and messages populated.
        """
        return ResponseMeta(
            applied_limit=self.applied_limit,
            truncated=self.truncated,
            messages=self.messages,
        )

    @property
    def count(self) -> int:
        """Return the number of items in this page."""
        return len(self.items)

    def map(self, fn: object) -> PaginatedFetch[object]:
        """
        Transform items while preserving pagination metadata.

        Parameters
        ----------
        fn
            Callable to apply to each item.

        Returns
        -------
        PaginatedFetch
            New instance with transformed items.

        Raises
        ------
        TypeError
            When fn is not callable.
        """
        if not callable(fn):
            message = "fn must be callable"
            raise TypeError(message)
        typed_fn: Callable[[T], object] = fn
        return PaginatedFetch(
            items=[typed_fn(item) for item in self.items],
            applied_limit=self.applied_limit,
            truncated=self.truncated,
            messages=list(self.messages),
        )


@dataclass(frozen=True)
class LimitClamp:
    """
    Result of clamping a user-provided limit against backend constraints.

    Parameters
    ----------
    applied
        The effective limit after clamping.
    requested
        The original limit requested by the caller.
    has_error
        Whether the requested limit was invalid (e.g., negative).
    messages
        Warning or error messages from limit validation.
    """

    applied: int | None
    requested: int | None
    has_error: bool = False
    messages: list[Message] = field(default_factory=list)

    def limit_or_default(self, default: int) -> int:
        """Return applied limit or fallback to default when None.

        Parameters
        ----------
        default
            Fallback value when applied is None.

        Returns
        -------
        int
            The applied limit if set, otherwise the default.
        """
        return self.applied if self.applied is not None else default


def clamp_limit(
    requested: int | None,
    *,
    default: int,
    max_limit: int,
) -> LimitClamp:
    """
    Clamp a user-provided limit against backend constraints.

    Parameters
    ----------
    requested
        The limit requested by the caller (None means use default).
    default
        Default limit when none is requested.
    max_limit
        Maximum allowed limit.

    Returns
    -------
    LimitClamp
        Clamping result with effective limit and any warning messages.

    Examples
    --------
    >>> result = clamp_limit(None, default=10, max_limit=100)
    >>> result.applied
    10

    >>> result = clamp_limit(500, default=10, max_limit=100)
    >>> result.applied
    100
    >>> len(result.messages)
    1
    """
    messages: list[Message] = []

    if requested is None:
        return LimitClamp(applied=default, requested=None, messages=messages)

    if requested < 0:
        messages.append(
            Message(
                code="limit_invalid",
                severity="error",
                detail=f"Limit must be non-negative, got {requested}",
            )
        )
        return LimitClamp(
            applied=default,
            requested=requested,
            has_error=True,
            messages=messages,
        )

    if requested > max_limit:
        messages.append(
            Message(
                code="limit_clamped",
                severity="warning",
                detail=f"Requested limit {requested} exceeds max {max_limit}; clamped",
            )
        )
        return LimitClamp(applied=max_limit, requested=requested, messages=messages)

    return LimitClamp(applied=requested, requested=requested, messages=messages)


@dataclass(frozen=True)
class OffsetClamp:
    """
    Result of clamping offset to non-negative.

    Parameters
    ----------
    applied
        The effective offset after clamping.
    requested
        The original offset requested.
    has_error
        Whether the requested offset was invalid.
    messages
        Any error messages from validation.
    """

    applied: int
    requested: int
    has_error: bool = False
    messages: list[Message] = field(default_factory=list)


def clamp_offset(offset: int) -> OffsetClamp:
    """
    Clamp an offset to a non-negative value.

    Parameters
    ----------
    offset
        Requested offset value.

    Returns
    -------
    OffsetClamp
        Applied offset and any validation messages.
    """
    if offset < 0:
        return OffsetClamp(
            applied=0,
            requested=offset,
            messages=[
                Message(
                    code="offset_invalid",
                    severity="error",
                    detail="offset must be non-negative",
                    context={"requested": offset},
                )
            ],
            has_error=True,
        )
    return OffsetClamp(applied=offset, requested=offset)


def paginate_items[T](
    items: list[T],
    *,
    limit: int | None,
    detect_truncation: bool = True,
) -> PaginatedFetch[T]:
    """
    Apply pagination to an in-memory item list.

    When detect_truncation is True, fetches limit+1 items to detect if more
    exist beyond the page boundary.

    Parameters
    ----------
    items
        Full list of items to paginate.
    limit
        Maximum items to return (None for unbounded).
    detect_truncation
        Whether to check for items beyond the limit.

    Returns
    -------
    PaginatedFetch[T]
        Paginated result with truncation metadata.
    """
    if limit is None:
        return PaginatedFetch(
            items=items,
            applied_limit=None,
            truncated=False,
        )

    truncated = detect_truncation and len(items) > limit
    page_items = items[:limit]

    return PaginatedFetch(
        items=page_items,
        applied_limit=limit,
        truncated=truncated,
    )


__all__ = [
    "BackendLimits",
    "LimitClamp",
    "OffsetClamp",
    "PaginatedFetch",
    "clamp_limit",
    "clamp_offset",
    "paginate_items",
]
