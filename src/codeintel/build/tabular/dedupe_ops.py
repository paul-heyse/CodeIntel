"""Deduplication helpers for Arrow and Polars tabular data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.compute_helpers import array_from_compute, call_compute
from codeintel.build.tabular.conversion import reader_to_table, tabular_to_arrow_reader
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.iter import iter_rows

if TYPE_CHECKING:
    from codeintel.core.schemas.service import SchemaService


def _schema_service() -> SchemaService:
    return get_schema_service()


def _schema_service_optional() -> SchemaService | None:
    try:
        return get_schema_service()
    except (RuntimeError, TypeError):
        return None


def _row_index_array(length: int) -> pa.Array | None:
    try:
        return pa.array(range(length), type=pa.int64())
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return None


def _row_index_name(table: pa.Table, *, base: str) -> str:
    existing = set(table.column_names)
    name = base
    suffix = 1
    while name in existing:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def _sort_table_for_preference(table: pa.Table, prefer_columns: Sequence[str]) -> pa.Table:
    sort_keys = [(name, "descending") for name in prefer_columns]
    options = pc.SortOptions(sort_keys=sort_keys)
    try:
        options = pc.SortOptions(sort_keys=sort_keys, null_placement="at_end")
        indices = call_compute("sort_indices", [table], options=options)
    except (TypeError, pa.ArrowNotImplementedError):
        indices = None
    if indices is None:
        indices = call_compute("sort_indices", [table], options=options)
    if indices is None:
        return table
    return table.take(indices)


def _dedupe_table_via_compute(
    table: pa.Table,
    *,
    key_columns: Sequence[str],
) -> pa.Table | None:
    if table.num_rows == 0:
        return table
    row_index_name = _row_index_name(table, base="_row_index")
    row_index = _row_index_array(table.num_rows)
    if row_index is None:
        return None
    try:
        indexed = table.append_column(row_index_name, row_index)
        grouped = indexed.group_by(list(key_columns)).aggregate([(row_index_name, "min")])
        index_column = f"{row_index_name}_min"
        if index_column not in grouped.column_names:
            return None
        indices = grouped.column(index_column)
        mask = array_from_compute("is_in", [row_index, indices])
        if mask is None:
            return None
        return table.filter(mask)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return None


def dedupe_table_for_table(
    table_key: str,
    table: pa.Table,
    *,
    prefer_columns: Sequence[str] | None = None,
) -> pa.Table:
    """Return a table with duplicate primary-key rows removed.

    Returns
    -------
    pa.Table
        Table with duplicate primary-key rows removed.
    """
    schema_service = _schema_service()
    schema = schema_service.get_table_schema(table_key)
    if schema is None or not schema.primary_key:
        return table
    key_columns = list(schema.primary_key)
    if prefer_columns:
        prefer = [name for name in prefer_columns if name in set(table.column_names)]
        if prefer:
            table = _sort_table_for_preference(table, prefer)
    try:
        return table.drop_duplicates(key_columns)
    except (AttributeError, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        deduped = _dedupe_table_via_compute(table, key_columns=key_columns)
        if deduped is not None:
            return deduped
        seen: set[tuple[object, ...]] = set()
        rows: list[dict[str, object]] = []
        for row in iter_rows(table):
            key = tuple(row.get(col) for col in key_columns)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
        if not rows:
            return pa.Table.from_batches([], schema=table.schema)
        return pa.Table.from_pylist(rows, schema=table.schema)


def _dedupe_lazyframe_for_table(
    frame: pl.LazyFrame,
    *,
    table_key: str,
    prefer_columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    schema_service = _schema_service_optional()
    schema = schema_service.get_table_schema(table_key) if schema_service is not None else None
    if schema is None or not schema.primary_key:
        return frame
    key_columns = list(schema.primary_key)
    if prefer_columns:
        prefer = [column for column in prefer_columns if column in set(schema.column_names())]
        if prefer:
            frame = frame.sort(by=prefer, descending=[True] * len(prefer), nulls_last=True)
    return frame.unique(subset=key_columns, keep="first")


def dedupe_tabular(
    table_key: str,
    value: InferableTabularInput,
    *,
    prefer_columns: Sequence[str] | None = None,
) -> pa.Table | pl.DataFrame | pl.LazyFrame:
    """Return a deduplicated tabular object based on table primary keys.

    Returns
    -------
    deduped : pa.Table | pl.DataFrame | pl.LazyFrame
        Deduplicated table or frame.
    """
    if isinstance(value, pl.LazyFrame):
        return _dedupe_lazyframe_for_table(
            value,
            table_key=table_key,
            prefer_columns=prefer_columns,
        )
    if isinstance(value, pl.DataFrame):
        deduped = _dedupe_lazyframe_for_table(
            value.lazy(),
            table_key=table_key,
            prefer_columns=prefer_columns,
        )
        return deduped.collect()
    if isinstance(value, pa.Table):
        return dedupe_table_for_table(table_key, value, prefer_columns=prefer_columns)
    if isinstance(value, pa.RecordBatchReader):
        table = reader_to_table(value)
        return dedupe_table_for_table(table_key, table, prefer_columns=prefer_columns)
    table = reader_to_table(tabular_to_arrow_reader(value))
    return dedupe_table_for_table(table_key, table, prefer_columns=prefer_columns)


__all__ = [
    "dedupe_table_for_table",
    "dedupe_tabular",
]
