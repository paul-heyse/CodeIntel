"""Timezone-aware datetime utilities shared across CodeIntel."""

from __future__ import annotations

from datetime import UTC, datetime

__all__ = ["utc_now"]


def utc_now() -> datetime:
    """Return current UTC timestamp with timezone info.

    Returns
    -------
    datetime
        Current datetime with UTC timezone attached.
    """
    return datetime.now(tz=UTC)
