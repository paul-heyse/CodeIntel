"""HTTP mixin helper utilities for limit clamping and error handling.

This module provides utilities for HTTP service mixins that handle pagination
and limit clamping. The helpers consolidate repetitive patterns found across
multiple service files.

Example Usage
-------------
Instead of::

    def list_high_risk_functions(self, *, limit: int | None = None) -> dm.HighRiskFunctionsResult:
        def _run() -> HighRiskFunctionsResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return HighRiskFunctionsResponse(functions=[], ...)
            # ... HTTP call

Use::

    def list_high_risk_functions(self, *, limit: int | None = None) -> dm.HighRiskFunctionsResult:
        def _run() -> HighRiskFunctionsResponse:
            clamped = clamp_limits(self.limits, limit)
            if clamped.has_error:
                return HighRiskFunctionsResponse(
                    functions=[], meta=ResponseMeta(messages=clamped.messages)
                )
            # ... HTTP call with clamped.applied_limit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.serving.backend.pagination import clamp_limit, clamp_offset

if TYPE_CHECKING:
    from codeintel.serving import domain_models as dm
    from codeintel.serving.backend import BackendLimits


@dataclass
class ClampedLimits:
    """Result of limit/offset clamping.

    Attributes
    ----------
    applied_limit
        The clamped limit value to use for the query.
    applied_offset
        The clamped offset value (0 if not provided).
    messages
        Warning or error messages from clamping.
    has_error
        True if clamping detected an error condition.
    """

    applied_limit: int
    applied_offset: int = 0
    messages: list[dm.Message] = field(default_factory=list)
    has_error: bool = False


def clamp_limits(
    limits: BackendLimits,
    limit: int | None,
    offset: int | None = None,
) -> ClampedLimits:
    """
    Apply limit/offset clamping with error detection.

    This function consolidates the repetitive limit clamping pattern found
    in HTTP service mixins.

    Parameters
    ----------
    limits
        Backend limits configuration.
    limit
        Requested limit (None uses default).
    offset
        Requested offset (None defaults to 0).

    Returns
    -------
    ClampedLimits
        Clamped values, messages, and error status.

    Examples
    --------
    >>> from codeintel.serving.backend import BackendLimits
    >>> limits = BackendLimits(default_limit=50, max_rows_per_call=1000)
    >>> result = clamp_limits(limits, None)
    >>> result.applied_limit
    50
    >>> result.has_error
    False
    """
    applied_limit = limits.default_limit if limit is None else limit
    clamp = clamp_limit(applied_limit, default=applied_limit, max_limit=limits.max_rows_per_call)

    messages: list[dm.Message] = list(clamp.messages)
    applied_offset = 0
    has_error = clamp.has_error

    if offset is not None:
        offset_clamp = clamp_offset(offset)
        messages.extend(offset_clamp.messages)
        applied_offset = offset_clamp.applied
        if offset_clamp.has_error:
            has_error = True

    return ClampedLimits(
        applied_limit=clamp.applied,
        applied_offset=applied_offset,
        messages=messages,
        has_error=has_error,
    )


__all__ = ["ClampedLimits", "clamp_limits"]
