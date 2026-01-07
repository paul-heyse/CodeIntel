"""Shared Arrow-first graph assembly helpers."""

from __future__ import annotations

from codeintel.build.graphs.assembly.collectors import (
    ColumnarBatchCollector,
    collector_for_table,
    empty_reader,
    reader_for_columnar_rows,
    reader_for_rows,
)
from codeintel.build.graphs.assembly.contracts import (
    align_reader_to_contract,
    align_table_to_contract,
    empty_contract_reader,
)
from codeintel.build.graphs.assembly.ids import payload_bytes, stable_decimal_id, stable_int_hash
from codeintel.build.graphs.assembly.readers import (
    drop_table_columns,
    ensure_table_columns,
    iter_normalized_tuples,
    reader_to_table,
    rename_table_columns,
    select_table_columns,
    table_rows,
    table_to_reader,
    tabular_to_reader,
    tabular_to_table,
)

__all__ = [
    "ColumnarBatchCollector",
    "align_reader_to_contract",
    "align_table_to_contract",
    "collector_for_table",
    "drop_table_columns",
    "empty_contract_reader",
    "empty_reader",
    "ensure_table_columns",
    "iter_normalized_tuples",
    "payload_bytes",
    "reader_for_columnar_rows",
    "reader_for_rows",
    "reader_to_table",
    "rename_table_columns",
    "select_table_columns",
    "stable_decimal_id",
    "stable_int_hash",
    "table_rows",
    "table_to_reader",
    "tabular_to_reader",
    "tabular_to_table",
]
