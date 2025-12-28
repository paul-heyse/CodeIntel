"""DataFrame utility functions for analytics pipelines.

This module provides common DataFrame operations used across analytics modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import polars as pl
import pyarrow as pa

if TYPE_CHECKING:
    from polars import DataFrame, LazyFrame


def to_records(frame: DataFrame | LazyFrame | pa.Table) -> list[dict[str, Any]]:
    """Convert a columnar frame into a list of dictionaries.

    Parameters
    ----------
    frame
        Polars DataFrame/LazyFrame or Arrow table to convert.

    Returns
    -------
    list[dict[str, Any]]
        Records returned by ``DataFrame.to_dicts()``.
    """
    if isinstance(frame, pa.Table):
        resolved = pl.from_arrow(frame)
        if isinstance(resolved, pl.Series):
            resolved = resolved.to_frame()
    elif isinstance(frame, pl.LazyFrame):
        resolved = frame.collect()
    else:
        resolved = frame
    return cast("list[dict[str, Any]]", resolved.to_dicts())


__all__ = ["to_records"]
