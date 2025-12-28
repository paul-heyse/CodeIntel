"""Backend-specific tabular step utilities for Hamilton pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl
import polars.datatypes as pl_datatypes

Frame = pl.LazyFrame
PolarsDataType = pl_datatypes.DataType | pl_datatypes.DataTypeClass


def drop_bad_rows(df: pl.LazyFrame, required_cols: tuple[str, ...]) -> pl.LazyFrame:
    """Drop rows with nulls in required columns (polars lazy).

    Returns
    -------
    pl.LazyFrame
        LazyFrame with invalid rows removed.
    """
    if not required_cols:
        return df
    return df.drop_nulls(list(required_cols))


def clip_numeric(df: pl.LazyFrame, col: str, max_value: float) -> pl.LazyFrame:
    """Clip numeric column values to a maximum bound (polars lazy).

    Returns
    -------
    pl.LazyFrame
        LazyFrame with the numeric column clipped to the max value.
    """
    return df.with_columns(pl.col(col).clip(upper_bound=max_value))


def _polars_dtype(dtype: str) -> PolarsDataType:
    return pl_datatypes.parse_into_dtype(dtype)


def cast_schema(df: pl.LazyFrame, schema: Mapping[str, str]) -> pl.LazyFrame:
    """Cast columns to the specified schema mapping.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with columns cast to the requested schema.
    """
    if not schema:
        return df
    return df.with_columns(
        [pl.col(name).cast(_polars_dtype(dtype)) for name, dtype in schema.items()]
    )


def normalize_nulls(df: pl.LazyFrame, policy: str) -> pl.LazyFrame:
    """Normalize null behavior based on the configured policy.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with null policy applied.

    Raises
    ------
    ValueError
        If the policy is not supported.
    """
    if policy == "preserve":
        return df
    if policy == "drop_bad_rows":
        return df.drop_nulls()
    msg = f"Unsupported null policy: {policy}"
    raise ValueError(msg)


def sort_columns(df: pl.LazyFrame, column_order: Sequence[str]) -> pl.LazyFrame:
    """Reorder columns to the provided stable order.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with columns reordered.
    """
    if not column_order:
        return df
    return df.select(_selector_by_name(column_order))


def _selector_by_name(names: Sequence[str]) -> pl.Expr | list[str]:
    selectors = getattr(pl, "selectors", None)
    if selectors is None:
        return list(names)
    by_name = getattr(selectors, "by_name", None)
    if callable(by_name):
        return by_name(list(names))
    return list(names)


__all__ = [
    "Frame",
    "cast_schema",
    "clip_numeric",
    "drop_bad_rows",
    "normalize_nulls",
    "sort_columns",
]
