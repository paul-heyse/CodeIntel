"""Deduplication helpers for Arrow tabular data."""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa

from codeintel.core.columnar.compute_helpers import array_from_compute, call_compute, sort_options
from codeintel.core.columnar.iter import iter_rows
from codeintel.core.schemas.service import get_schema_service


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
    options = sort_options(sort_keys, null_placement="at_end")
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
    schema_service = get_schema_service()
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


__all__ = ["dedupe_table_for_table"]
