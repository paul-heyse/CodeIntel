"""Columnar test helpers for building Arrow tables.

Tuple-based helpers are deprecated; prefer the stream helpers in
``tests._helpers.columnar_streams`` for contract-aware readers/LazyFrames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.columnar.rows import ColumnarRows
from codeintel.storage.warehouse import Warehouse
from tests._helpers.columnar_streams import reader_for_rows, table_for_rows

if TYPE_CHECKING:
    from codeintel.storage.warehouse import MaterializationResult, MaterializeOptions

RowsInput = Sequence[tuple[object, ...]] | Sequence[Mapping[str, object]] | ColumnarRows


def arrow_table_for_rows(
    table_key: str,
    rows: RowsInput,
    *,
    columns: Sequence[str] | None = None,
) -> pa.Table:
    """Create an Arrow table from row tuples or columnar rows.

    Parameters
    ----------
    table_key
        Table key used to resolve schema order.
    rows
        Row tuples or columnar rows.
    columns
        Optional column order for tuple rows. Defaults to schema order.

    Returns
    -------
    pyarrow.Table
        Arrow table with schema-aligned columns.
    """
    return table_for_rows(table_key, rows, columns=columns)


def materialize_table_from_rows(
    warehouse: Warehouse,
    table_key: str,
    rows: RowsInput,
    *,
    columns: Sequence[str] | None = None,
    options: MaterializeOptions | None = None,
) -> MaterializationResult:
    """Materialize row data via the columnar table path.

    Parameters
    ----------
    warehouse
        Warehouse to use for materialization.
    table_key
        Destination table key.
    rows
        Row tuples or columnar rows to materialize.
    columns
        Optional column order for tuple rows.
    options
        Optional materialization options.

    Returns
    -------
    MaterializationResult
        Result metadata for the write.
    """
    reader = reader_for_rows(table_key, rows, columns=columns)
    return warehouse.materialize_table(table_key, reader, options=options)


__all__ = ["arrow_table_for_rows", "materialize_table_from_rows"]
