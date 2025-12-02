"""Run ID generation utilities.

This module provides functions for generating unique run identifiers
with configurable prefixes for different execution contexts.
"""

from __future__ import annotations

from uuid import uuid4


def new_run_id(prefix: str = "ci") -> str:
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
    return f"{prefix}-{uuid4().hex}"


__all__ = ["new_run_id"]
