"""Normalization utilities for ingestion LazyFrame outputs."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
from polars.exceptions import PolarsError

from codeintel.build.hamilton.transforms.tabular_steps import sort_columns
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.frames import dedupe_frame_for_table
from codeintel.core.columnar.schema_alignment import extras_policy_from_schema


def normalize_ingest_frame(
    frame: pl.LazyFrame | None,
    *,
    table_key: str,
    add_missing: bool = True,
    keep_extras: bool | None = None,
) -> pl.LazyFrame | None:
    """Normalize ingestion frames for schema alignment and deduping.

    Parameters
    ----------
    frame
        LazyFrame to normalize (None means skip).
    table_key
        Target table key for schema alignment.
    add_missing
        Whether to add missing schema columns as nulls.
    keep_extras
        Whether to keep extra columns not in the schema. When None, respect
        the extras policy encoded in the Arrow schema metadata.

    Returns
    -------
    pl.LazyFrame | None
        Normalized LazyFrame or None if input is None.
    """
    if frame is None:
        return None
    normalized = dedupe_frame_for_table(frame, table_key=table_key)
    schema_service = get_schema_service()
    arrow_schema = schema_service.get_arrow_schema(table_key)
    if arrow_schema is None:
        schema = schema_service.get_table_schema(table_key)
        if schema is None:
            return normalized
        ordered_columns = schema.column_names()
        resolved_keep_extras = True if keep_extras is None else keep_extras
    else:
        ordered_columns = [str(name) for name in arrow_schema.names]
        if keep_extras is None:
            resolved_keep_extras = extras_policy_from_schema(arrow_schema) == "retain"
        else:
            resolved_keep_extras = keep_extras
    if add_missing:
        normalized = _add_missing_columns(normalized, ordered_columns)
    return _reorder_columns(
        normalized,
        ordered_columns,
        keep_extras=resolved_keep_extras,
    )


def _add_missing_columns(frame: pl.LazyFrame, columns: Sequence[str]) -> pl.LazyFrame:
    current = set(_column_names(frame))
    missing = [name for name in columns if name not in current]
    if not missing:
        return frame
    additions = [pl.lit(None).alias(name) for name in missing]
    return frame.with_columns(additions)


def _reorder_columns(
    frame: pl.LazyFrame,
    columns: Sequence[str],
    *,
    keep_extras: bool,
) -> pl.LazyFrame:
    current = _column_names(frame)
    extras = [name for name in current if name not in columns]
    ordered = list(columns)
    if keep_extras and extras:
        ordered.extend(extras)
    return sort_columns(frame, ordered)


def _column_names(frame: pl.LazyFrame) -> list[str]:
    try:
        schema = frame.collect_schema()
    except PolarsError:
        return []
    if callable(getattr(schema, "names", None)):
        return [str(name) for name in schema.names()]
    return [str(name) for name in list(schema)]


__all__ = ["normalize_ingest_frame"]
