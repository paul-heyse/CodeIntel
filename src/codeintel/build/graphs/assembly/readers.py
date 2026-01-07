"""Arrow reader/table helpers for graph assembly."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pyarrow as pa

from codeintel.build.tabular.conversion import reader_to_table, table_to_reader
from codeintel.build.tabular.conversion import tabular_to_arrow_reader as tabular_to_reader
from codeintel.build.tabular.conversion import tabular_to_arrow_table as tabular_to_table
from codeintel.build.tabular.table_ops import (
    drop_table_columns,
    ensure_table_columns,
    rename_table_columns,
    select_table_columns,
    table_rows,
)
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.columnar.type_normalization import normalize_reader


def iter_normalized_tuples(
    reader: pa.RecordBatchReader,
    *,
    columns: Sequence[str] | None = None,
) -> Iterable[tuple[object, ...]]:
    """Yield normalized row tuples from a record batch reader.

    Yields
    ------
    tuple[object, ...]
        Row tuples in column order after normalization.
    """
    for batch in normalize_reader(reader):
        yield from iter_tuples(batch, columns=columns)

__all__ = [
    "drop_table_columns",
    "ensure_table_columns",
    "iter_normalized_tuples",
    "reader_to_table",
    "rename_table_columns",
    "select_table_columns",
    "table_rows",
    "table_to_reader",
    "tabular_to_reader",
    "tabular_to_table",
]
