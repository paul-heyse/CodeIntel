"""Pagination utilities and safety limits for serving backends.

This module provides:
- Standardized pagination handling (PaginatedFetch, paginate_items)
- Limit/offset clamping with messaging (clamp_limit_value, clamp_offset_value)
- Backend configuration (BackendLimits)

Every bounded operation should use these utilities to ensure consistent
truncation detection and safe limit handling across all serving endpoints.

Note
----
This module consolidates the previous ``limits.py`` module. The legacy
``ClampResult`` and ``clamp_limit_value`` functions are preserved for
backward compatibility.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from codeintel.serving.mcp.models import Message, ResponseMeta


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
# Legacy Clamping (preserved for backward compatibility)
# =============================================================================


@dataclass(frozen=True)
class ClampResult:
    """
    Result of clamping limit/offset values with messaging.

    This is the legacy result type. New code should prefer ``LimitClamp``.

    Parameters
    ----------
    applied
        The effective value after clamping (always an int).
    messages
        Warning or error messages from validation.
    has_error
        Whether validation failed.
    """

    applied: int
    messages: list[Message] = field(default_factory=list)
    has_error: bool = False


def clamp_limit_value(
    requested: int | None,
    *,
    default: int,
    max_limit: int,
) -> ClampResult:
    """
    Clamp a requested limit to safe bounds, returning warnings instead of raising.

    Parameters
    ----------
    requested
        Requested limit value; ``None`` means "use default".
    default
        Default limit to apply when none is requested.
    max_limit
        Maximum rows allowed for any call.

    Returns
    -------
    ClampResult
        Applied limit plus any informational or error messages.
    """
    messages: list[Message] = []
    limit = default if requested is None else requested

    if limit < 0:
        messages.append(
            Message(
                code="limit_invalid",
                severity="error",
                detail="limit must be non-negative",
                context={"requested": limit},
            )
        )
        return ClampResult(applied=0, messages=messages, has_error=True)

    if limit > max_limit:
        messages.append(
            Message(
                code="limit_clamped",
                severity="warning",
                detail=f"Requested {limit} rows; delivering {max_limit} (max allowed).",
                context={"requested": limit, "applied": max_limit, "max": max_limit},
            )
        )
        limit = max_limit

    return ClampResult(applied=limit, messages=messages, has_error=False)


def clamp_offset_value(offset: int) -> ClampResult:
    """
    Clamp an offset to a non-negative value, returning messaging instead of raising.

    Parameters
    ----------
    offset
        Requested offset value.

    Returns
    -------
    ClampResult
        Applied offset and any validation messages.
    """
    if offset < 0:
        return ClampResult(
            applied=0,
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
    return ClampResult(applied=offset)


# =============================================================================
# Modern Pagination Types
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
    "ClampResult",
    "LimitClamp",
    "PaginatedFetch",
    "clamp_limit",
    "clamp_limit_value",
    "clamp_offset_value",
    "paginate_items",
]
