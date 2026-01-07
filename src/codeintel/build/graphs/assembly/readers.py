"""Arrow reader/table helpers for graph assembly."""

from __future__ import annotations

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

__all__ = [
    "drop_table_columns",
    "ensure_table_columns",
    "reader_to_table",
    "rename_table_columns",
    "select_table_columns",
    "table_rows",
    "table_to_reader",
    "tabular_to_reader",
    "tabular_to_table",
]
