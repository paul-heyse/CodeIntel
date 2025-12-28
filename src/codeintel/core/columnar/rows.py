"""Columnar row buffering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.schemas.row_models import normalize_row_value_for_type
from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.core.schemas.primitives import ColumnType

ColumnarRows = dict[str, list[object]]


@dataclass(slots=True)
class ColumnarRowBuffer:
    """Mutable buffer for building columnar row payloads."""

    table_key: str
    columns: tuple[str, ...]
    column_types: tuple[ColumnType, ...]
    data: ColumnarRows
    row_count: int = 0

    def append(self, row: Mapping[str, object]) -> None:
        """Append a row mapping to the buffer."""
        for name, col_type in zip(self.columns, self.column_types, strict=True):
            self.data[name].append(normalize_row_value_for_type(row[name], col_type))
        self.row_count += 1

    def extend(self, rows: Sequence[Mapping[str, object]]) -> None:
        """Append multiple rows to the buffer."""
        for row in rows:
            self.append(row)


def columnar_buffer_for_table_key(table_key: str) -> ColumnarRowBuffer:
    """Create a ColumnarRowBuffer using the table schema registry."""
    schema = get_schema_service().require_table_schema(table_key)
    columns = tuple(schema.column_names())
    column_types = tuple(column.type for column in schema.columns)
    return ColumnarRowBuffer(
        table_key=table_key,
        columns=columns,
        column_types=column_types,
        data={name: [] for name in columns},
    )


def columnar_row_count(columns: Mapping[str, Sequence[object]]) -> int:
    """Return row count for a columnar mapping, validating lengths."""
    lengths = {len(values) for values in columns.values()}
    if not lengths:
        return 0
    if len(lengths) > 1:
        msg = f"Column lengths mismatch: {sorted(lengths)}"
        raise ValueError(msg)
    return lengths.pop()


__all__ = [
    "ColumnarRowBuffer",
    "ColumnarRows",
    "columnar_buffer_for_table_key",
    "columnar_row_count",
]
