"""Polars utility helpers for columnar execution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from polars import QueryOptFlags

    type PolarsQueryOptFlags = QueryOptFlags
else:
    type PolarsQueryOptFlags = object

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

LOG = logging.getLogger(__name__)


def resolve_query_opt_flags(flags: tuple[str, ...]) -> PolarsQueryOptFlags | None:
    """Resolve Polars QueryOptFlags from string names.

    Parameters
    ----------
    flags
        Tuple of flag names to resolve.

    Returns
    -------
    PolarsQueryOptFlags | None
        Combined QueryOptFlags object when available.
    """
    if pl is None or not flags:
        return None
    opt_flags = getattr(pl, "QueryOptFlags", None)
    if opt_flags is None:
        return None
    resolved: PolarsQueryOptFlags | None = None
    for raw_flag in flags:
        name = raw_flag.upper()
        candidate = getattr(opt_flags, name, None)
        if candidate is None:
            candidate = getattr(opt_flags, raw_flag, None)
        if candidate is None:
            LOG.debug("Unknown Polars QueryOptFlags value: %s", raw_flag)
            continue
        resolved_flag = cast("PolarsQueryOptFlags", candidate)
        if resolved is None:
            resolved = resolved_flag
            continue
        or_fn = getattr(resolved, "__or__", None)
        if callable(or_fn):
            resolved = cast("PolarsQueryOptFlags", or_fn(resolved_flag))
        else:
            resolved = resolved_flag
    return resolved


__all__ = ["PolarsQueryOptFlags", "resolve_query_opt_flags"]
