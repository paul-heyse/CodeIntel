"""Common utilities for builder row dataclasses."""

from __future__ import annotations

from datetime import UTC, datetime


def _iso(dt: datetime | None = None) -> str:
    """Return an ISO-8601 timestamp with timezone.

    Parameters
    ----------
    dt
        Optional datetime to format. Defaults to current time.

    Returns
    -------
    str
        Timestamp in ISO format with timezone.
    """
    return (dt or datetime.now(UTC)).isoformat()
