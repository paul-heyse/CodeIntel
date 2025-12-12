"""Run ID generation utilities.

This module provides functions for generating unique run identifiers
with configurable prefixes for different execution contexts.
"""

from __future__ import annotations

from uuid import uuid4

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
    return f"{prefix}-{uuid4().hex}"


__all__ = [
    "RUN_PREFIX_ANALYTICS",
    "RUN_PREFIX_GRAPHS",
    "RUN_PREFIX_INGEST",
    "RUN_PREFIX_PIPELINE",
    "RUN_PREFIX_PLAN",
    "new_run_id",
]
