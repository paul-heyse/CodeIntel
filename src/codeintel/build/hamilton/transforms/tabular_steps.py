"""Backend-specific tabular step utilities for Hamilton pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import polars as pl
import polars.datatypes as pl_datatypes

from codeintel.build.tabular.types import TabularInput

Frame = TabularInput
PolarsDataType = pl_datatypes.DataType | pl_datatypes.DataTypeClass


def drop_bad_rows(df: TabularInput, required_cols: tuple[str, ...]) -> TabularInput:
    """Drop rows with nulls in required columns.

    Returns
    -------
    TabularInput
        Filtered LazyFrame or the original input for non-Polars data.
    """
    lazyframe = _as_lazyframe(df)
    if lazyframe is None:
        return df
    if not required_cols:
        return lazyframe
    return lazyframe.drop_nulls(list(required_cols))


def clip_numeric(df: TabularInput, col: str, max_value: float) -> TabularInput:
    """Clip numeric column values to a maximum bound.

    Returns
    -------
    TabularInput
        LazyFrame with the numeric column clipped or the original input.
    """
    lazyframe = _as_lazyframe(df)
    if lazyframe is None:
        return df
    return lazyframe.with_columns(pl.col(col).clip(upper_bound=max_value))


def _polars_dtype(dtype: str) -> PolarsDataType:
    return pl_datatypes.parse_into_dtype(dtype)


def cast_schema(df: TabularInput, schema: Mapping[str, str]) -> TabularInput:
    """Cast columns to the specified schema mapping when available.

    Returns
    -------
    TabularInput
        LazyFrame with columns cast or the original input.
    """
    lazyframe = _as_lazyframe(df)
    if lazyframe is None:
        return df
    if not schema:
        return lazyframe
    return lazyframe.with_columns(
        [pl.col(name).cast(_polars_dtype(dtype)) for name, dtype in schema.items()]
    )


def normalize_nulls(df: TabularInput, policy: str) -> TabularInput:
    """Normalize null behavior based on the configured policy.

    Returns
    -------
    TabularInput
        LazyFrame with null policy applied or the original input.

    Raises
    ------
    ValueError
        If the policy is not supported.
    """
    lazyframe = _as_lazyframe(df)
    if lazyframe is None:
        return df
    if policy == "preserve":
        return lazyframe
    if policy == "drop_bad_rows":
        return lazyframe.drop_nulls()
    msg = f"Unsupported null policy: {policy}"
    raise ValueError(msg)


def sort_columns(df: TabularInput, column_order: Sequence[str]) -> TabularInput:
    """Reorder columns to the provided stable order for Polars data.

    Returns
    -------
    TabularInput
        LazyFrame with columns reordered or the original input.
    """
    lazyframe = _as_lazyframe(df)
    if lazyframe is None:
        return df
    if not column_order:
        return lazyframe
    return lazyframe.select(_selector_by_name(column_order))


def _selector_by_name(names: Sequence[str]) -> pl.Expr | list[str]:
    selectors = getattr(pl, "selectors", None)
    if selectors is None:
        return list(names)
    by_name = getattr(selectors, "by_name", None)
    if callable(by_name):
        return cast("pl.Expr", by_name(list(names)))
    return list(names)


def _as_lazyframe(df: TabularInput) -> pl.LazyFrame | None:
    if isinstance(df, pl.LazyFrame):
        return df
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    return None


__all__ = [
    "Frame",
    "cast_schema",
    "clip_numeric",
    "drop_bad_rows",
    "normalize_nulls",
    "sort_columns",
]
