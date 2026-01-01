"""Shared helpers for analytics table construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from codeintel.core.schemas.generated_rows import columns_for_table_key


def _resolved_columns(
    *,
    table_key: str,
    columns: Sequence[str] | None,
) -> list[str]:
    if columns is not None:
        return list(columns)
    inferred = columns_for_table_key(table_key)
    if inferred is None:
        return []
    return list(inferred)


def _empty_frame(columns: Sequence[str]) -> pl.LazyFrame:
    if not columns:
        msg = "Empty frame requires column names for inference-first outputs"
        raise ValueError(msg)
    schema = [(name, pl.Null) for name in columns]
    return pl.DataFrame(schema=schema).lazy()


def empty_frame_for_table(table_key: str, *, columns: Sequence[str] | None = None) -> pl.LazyFrame:
    """Return an empty LazyFrame with ordered columns.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    columns
        Optional explicit column order when no row models exist.

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame with ordered columns.
    """
    resolved = _resolved_columns(table_key=table_key, columns=columns)
    return _empty_frame(resolved)


def rows_to_frame(
    table_key: str,
    rows: Sequence[Mapping[str, object]] | Sequence[Sequence[object]],
    *,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Convert row sequences into a LazyFrame with schema-ordered columns.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    rows
        Row mappings or row tuples in the expected column order.
    columns
        Optional explicit column order for tuple rows.

    Returns
    -------
    polars.LazyFrame
        LazyFrame with schema-ordered columns.

    Raises
    ------
    ValueError
        If tuple rows are provided without an explicit column order.
    """
    ordered_columns = _resolved_columns(table_key=table_key, columns=columns)
    if not rows:
        return _empty_frame(ordered_columns)
    first = rows[0]
    if isinstance(first, Mapping):
        frame = pl.DataFrame(rows)
        if not ordered_columns:
            ordered_columns = list(frame.columns)
        missing = [col for col in ordered_columns if col not in frame.columns]
        if missing:
            frame = frame.with_columns([pl.lit(None).alias(col) for col in missing])
        return frame.lazy().select(ordered_columns)
    if not ordered_columns:
        msg = f"Column order required for tuple rows in {table_key}"
        raise ValueError(msg)
    frame = pl.DataFrame(rows, schema=ordered_columns, orient="row")
    return frame.lazy().select(ordered_columns)


__all__ = ["empty_frame_for_table", "rows_to_frame"]
