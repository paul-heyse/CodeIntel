"""Run ID generation utilities.

This module provides functions for generating unique run identifiers
with configurable prefixes for different execution contexts.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID, uuid4

try:
    from uuid6 import uuid7 as _uuid7

    _UUID7_AVAILABLE = True
except ImportError:
    _UUID7_AVAILABLE = False
    _uuid7 = None

RUN_PREFIX_PIPELINE = "ci"
RUN_PREFIX_INGEST = "ingest"
RUN_PREFIX_GRAPHS = "graphs"
RUN_PREFIX_ANALYTICS = "analytics"
RUN_PREFIX_PLAN = "plan"


def new_run_id(prefix: str = RUN_PREFIX_PIPELINE) -> str:
    """Generate a new opaque run identifier.

    Parameters
    ----------
    prefix
        Short prefix to identify the run type (e.g., "ci", "ingest", "graphs").

    Returns
    -------
    str
        Unique run identifier in format "{prefix}-{uuid_hex}".

    Examples
    --------
    >>> rid = new_run_id("test")
    >>> rid.startswith("test-")
    True
    >>> len(rid.split("-", 1)[1]) == 32
    True
    """
    return f"{prefix}-{new_uuid_hex()}"


def new_uuid() -> UUID:
    """Return a new UUID, preferring UUIDv7 when available.

    Returns
    -------
    UUID
        Generated UUID value.
    """
    if _UUID7_AVAILABLE and isinstance(_uuid7, Callable):
        return _uuid7()
    return uuid4()


def new_uuid_hex() -> str:
    """Return a hex-encoded UUID string.

    Returns
    -------
    str
        UUID value rendered as a hex string.
    """
    return new_uuid().hex


def new_uuid_str() -> str:
    """Return a standard UUID string representation.

    Returns
    -------
    str
        UUID value rendered in canonical string form.
    """
    return str(new_uuid())


__all__ = [
    "RUN_PREFIX_ANALYTICS",
    "RUN_PREFIX_GRAPHS",
    "RUN_PREFIX_INGEST",
    "RUN_PREFIX_PIPELINE",
    "RUN_PREFIX_PLAN",
    "new_run_id",
    "new_uuid",
    "new_uuid_hex",
    "new_uuid_str",
]
