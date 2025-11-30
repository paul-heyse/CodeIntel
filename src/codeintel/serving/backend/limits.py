"""Shared safety limits and clamping helpers for serving backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.serving.mcp.models import Message


@dataclass(frozen=True)
class BackendLimits:
    """Safety limits applied uniformly across backends."""

    default_limit: int = 50
    max_rows_per_call: int = 500

    @classmethod
    def from_config(cls, cfg: object) -> BackendLimits:
        """
        Build limits from configuration objects exposing default_limit/max_rows_per_call.

        Parameters
        ----------
        cfg:
            Object with optional `default_limit` and `max_rows_per_call` attributes.

        Returns
        -------
        BackendLimits
            Limits derived from the provided configuration.
        """
        default = getattr(cfg, "default_limit", cls.default_limit)
        maximum = getattr(cfg, "max_rows_per_call", cls.max_rows_per_call)
        return cls(default_limit=int(default), max_rows_per_call=int(maximum))


@dataclass(frozen=True)
class ClampResult:
    """Result of clamping limit/offset values with messaging."""

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
    requested:
        Requested limit value; ``None`` means "use default".
    default:
        Default limit to apply when none is requested.
    max_limit:
        Maximum rows allowed for any call.

    Returns
    -------
    ClampResult
        Applied limit plus any informational or error messages.
    """
    message_cls = _get_message_class()
    messages: list[Message] = []
    limit = default if requested is None else requested

    if limit < 0:
        messages.append(
            message_cls(
                code="limit_invalid",
                severity="error",
                detail="limit must be non-negative",
                context={"requested": limit},
            )
        )
        return ClampResult(applied=0, messages=messages, has_error=True)

    if limit > max_limit:
        messages.append(
            message_cls(
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
    offset:
        Requested offset value.

    Returns
    -------
    ClampResult
        Applied offset and any validation messages.
    """
    message_cls = _get_message_class()
    if offset < 0:
        return ClampResult(
            applied=0,
            messages=[
                message_cls(
                    code="offset_invalid",
                    severity="error",
                    detail="offset must be non-negative",
                    context={"requested": offset},
                )
            ],
            has_error=True,
        )
    return ClampResult(applied=offset)


def _get_message_class() -> type[Message]:
    """
    Return the Message model without importing at module import time.

    Returns
    -------
    type[Message]
        Message model used for clamp results.
    """
    module = import_module("codeintel.serving.mcp.models")
    return module.Message
