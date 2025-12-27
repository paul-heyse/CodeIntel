"""Backend-specific tabular step utilities for Hamilton pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import polars as pl

Frame = pd.DataFrame | pl.DataFrame | pl.LazyFrame


def _drop_bad_rows_pandas(df: pd.DataFrame, required_cols: tuple[str, ...]) -> pd.DataFrame:
    """Drop rows with nulls in required columns (pandas)."""
    if not required_cols:
        return df
    return df.dropna(subset=list(required_cols))


def _drop_bad_rows_polars(df: pl.DataFrame, required_cols: tuple[str, ...]) -> pl.DataFrame:
    """Drop rows with nulls in required columns (polars)."""
    if not required_cols:
        return df
    return df.drop_nulls(list(required_cols))


def _drop_bad_rows_polars_lazy(
    df: pl.LazyFrame, required_cols: tuple[str, ...]
) -> pl.LazyFrame:
    """Drop rows with nulls in required columns (polars lazy)."""
    if not required_cols:
        return df
    return df.drop_nulls(list(required_cols))


def _clip_numeric(df: Frame, col: str, max_value: float) -> Frame:
    """Clip numeric column values to a maximum bound."""
    if isinstance(df, pd.DataFrame):
        df = df.copy()
        df[col] = df[col].clip(upper=max_value)
        return df
    if isinstance(df, pl.LazyFrame):
        return df.with_columns(pl.col(col).clip_max(max_value))
    return df.with_columns(pl.col(col).clip_max(max_value))


def _cast_schema(df: Frame, schema: Mapping[str, str]) -> Frame:
    """Cast columns to the specified schema mapping."""
    if not schema:
        return df
    if isinstance(df, pd.DataFrame):
        return df.astype(dict(schema))
    if isinstance(df, pl.LazyFrame):
        return df.with_columns([pl.col(name).cast(dtype) for name, dtype in schema.items()])
    return df.with_columns([pl.col(name).cast(dtype) for name, dtype in schema.items()])


def _normalize_nulls(df: Frame, policy: str) -> Frame:
    """Normalize null behavior based on the configured policy."""
    if policy == "preserve":
        return df
    if policy == "drop_bad_rows":
        if isinstance(df, pd.DataFrame):
            return df.dropna()
        return df.drop_nulls()
    msg = f"Unsupported null policy: {policy}"
    raise ValueError(msg)


def _sort_columns(df: Frame, column_order: Sequence[str]) -> Frame:
    """Reorder columns to the provided stable order."""
    if not column_order:
        return df
    if isinstance(df, pd.DataFrame):
        return df.loc[:, list(column_order)]
    if isinstance(df, pl.LazyFrame):
        return df.select(list(column_order))
    return df.select(list(column_order))


__all__ = [
    "Frame",
    "_cast_schema",
    "_clip_numeric",
    "_drop_bad_rows_pandas",
    "_drop_bad_rows_polars",
    "_drop_bad_rows_polars_lazy",
    "_normalize_nulls",
    "_sort_columns",
]
