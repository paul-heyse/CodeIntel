"""Backend-specific tabular step utilities for Hamilton pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import polars as pl
import polars.datatypes as pl_datatypes

Frame = pd.DataFrame | pl.DataFrame | pl.LazyFrame
PolarsDataType = pl_datatypes.DataType | pl_datatypes.DataTypeClass


def drop_bad_rows_pandas(df: pd.DataFrame, required_cols: tuple[str, ...]) -> pd.DataFrame:
    """Drop rows with nulls in required columns (pandas).

    Returns
    -------
    pd.DataFrame
        DataFrame with invalid rows removed.
    """
    if not required_cols:
        return df
    return df.dropna(subset=list(required_cols))


def drop_bad_rows_polars(df: pl.DataFrame, required_cols: tuple[str, ...]) -> pl.DataFrame:
    """Drop rows with nulls in required columns (polars).

    Returns
    -------
    pl.DataFrame
        DataFrame with invalid rows removed.
    """
    if not required_cols:
        return df
    return df.drop_nulls(list(required_cols))


def drop_bad_rows_polars_lazy(df: pl.LazyFrame, required_cols: tuple[str, ...]) -> pl.LazyFrame:
    """Drop rows with nulls in required columns (polars lazy).

    Returns
    -------
    pl.LazyFrame
        LazyFrame with invalid rows removed.
    """
    if not required_cols:
        return df
    return df.drop_nulls(list(required_cols))


def clip_numeric(df: Frame, col: str, max_value: float) -> Frame:
    """Clip numeric column values to a maximum bound.

    Returns
    -------
    Frame
        Frame with the column clipped to the max value.
    """
    if isinstance(df, pd.DataFrame):
        df = df.copy()
        df[col] = df[col].clip(upper=max_value)
        return df
    if isinstance(df, pl.LazyFrame):
        return df.with_columns(pl.col(col).clip(upper_bound=max_value))
    return df.with_columns(pl.col(col).clip(upper_bound=max_value))


def clip_numeric_pandas(df: pd.DataFrame, col: str, max_value: float) -> pd.DataFrame:
    """Clip numeric column values to a maximum bound (pandas).

    Returns
    -------
    pd.DataFrame
        DataFrame with the column clipped to the max value.
    """
    updated = df.copy()
    updated[col] = updated[col].clip(upper=max_value)
    return updated


def clip_numeric_polars(df: pl.DataFrame, col: str, max_value: float) -> pl.DataFrame:
    """Clip numeric column values to a maximum bound (polars).

    Returns
    -------
    pl.DataFrame
        DataFrame with the column clipped to the max value.
    """
    return df.with_columns(pl.col(col).clip(upper_bound=max_value))


def clip_numeric_polars_lazy(df: pl.LazyFrame, col: str, max_value: float) -> pl.LazyFrame:
    """Clip numeric column values to a maximum bound (polars lazy).

    Returns
    -------
    pl.LazyFrame
        LazyFrame with the column clipped to the max value.
    """
    return df.with_columns(pl.col(col).clip(upper_bound=max_value))


def _polars_dtype(dtype: str) -> PolarsDataType:
    return pl_datatypes.parse_into_dtype(dtype)


def cast_schema(df: Frame, schema: Mapping[str, str]) -> Frame:
    """Cast columns to the specified schema mapping.

    Returns
    -------
    Frame
        Frame with columns cast to the requested schema.
    """
    if not schema:
        return df
    if isinstance(df, pd.DataFrame):
        return df.astype(dict(schema))
    if isinstance(df, pl.LazyFrame):
        return df.with_columns(
            [pl.col(name).cast(_polars_dtype(dtype)) for name, dtype in schema.items()]
        )
    return df.with_columns([pl.col(name).cast(_polars_dtype(dtype)) for name, dtype in schema.items()])


def normalize_nulls(df: Frame, policy: str) -> Frame:
    """Normalize null behavior based on the configured policy.

    Returns
    -------
    Frame
        Frame with null policy applied.

    Raises
    ------
    ValueError
        If the policy is not supported.
    """
    if policy == "preserve":
        return df
    if policy == "drop_bad_rows":
        if isinstance(df, pd.DataFrame):
            return df.dropna()
        return df.drop_nulls()
    msg = f"Unsupported null policy: {policy}"
    raise ValueError(msg)


def normalize_nulls_pandas(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Normalize null behavior based on the configured policy (pandas).

    Returns
    -------
    pd.DataFrame
        DataFrame with null policy applied.

    Raises
    ------
    ValueError
        If the policy is not supported.
    """
    if policy == "preserve":
        return df
    if policy == "drop_bad_rows":
        return df.dropna()
    msg = f"Unsupported null policy: {policy}"
    raise ValueError(msg)


def normalize_nulls_polars(df: pl.DataFrame, policy: str) -> pl.DataFrame:
    """Normalize null behavior based on the configured policy (polars).

    Returns
    -------
    pl.DataFrame
        DataFrame with null policy applied.

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


def normalize_nulls_polars_lazy(df: pl.LazyFrame, policy: str) -> pl.LazyFrame:
    """Normalize null behavior based on the configured policy (polars lazy).

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


def sort_columns(df: Frame, column_order: Sequence[str]) -> Frame:
    """Reorder columns to the provided stable order.

    Returns
    -------
    Frame
        Frame with columns reordered.
    """
    if not column_order:
        return df
    if isinstance(df, pd.DataFrame):
        return df.loc[:, list(column_order)]
    if isinstance(df, pl.LazyFrame):
        return df.select(list(column_order))
    return df.select(list(column_order))


__all__ = [
    "Frame",
    "cast_schema",
    "clip_numeric",
    "clip_numeric_pandas",
    "clip_numeric_polars",
    "clip_numeric_polars_lazy",
    "drop_bad_rows_pandas",
    "drop_bad_rows_polars",
    "drop_bad_rows_polars_lazy",
    "normalize_nulls",
    "normalize_nulls_pandas",
    "normalize_nulls_polars",
    "normalize_nulls_polars_lazy",
    "sort_columns",
]
