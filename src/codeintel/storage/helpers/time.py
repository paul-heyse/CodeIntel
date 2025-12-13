"""Timezone-aware datetime utilities for storage operations.

This module provides shared time utilities used across tracking modules.
All timestamps are UTC-aware following the project's datetime hygiene rules.

Example
-------
>>> from codeintel.storage.helpers.time import utc_now
>>> ts = utc_now()
>>> ts.tzinfo is not None
True
"""

from __future__ import annotations

from datetime import UTC, datetime

__all__ = ["utc_now"]


def utc_now() -> datetime:
    """Return current UTC timestamp with timezone info.

    Returns
    -------
    datetime
        Current datetime with UTC timezone attached.

    Examples
    --------
    >>> ts = utc_now()
    >>> ts.tzinfo is not None
    True
    >>> from datetime import UTC
    >>> ts.tzinfo == UTC
    True
    """
    return datetime.now(tz=UTC)
