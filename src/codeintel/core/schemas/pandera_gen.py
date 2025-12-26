"""Pandera schema generation from core TableSchema primitives.

This module provides the default bridge from the project-wide schema language
(``TableSchema``) to Pandera schemas used for validation boundaries.

Constraints (uniqueness, ranges, etc.) are intentionally out of scope here and
should be layered separately by higher-level build contracts.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from pandera import Check, DataFrameSchema
from pandera import Column as PanderaColumn

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import ColumnType, TableSchema

from codeintel.core.schemas.pandera_types import dtype_for_column_type


def _schema_signature(schema: TableSchema) -> tuple[tuple[str, ColumnType, bool], ...]:
    return tuple((col.name, col.type, col.nullable) for col in schema.columns)


@lru_cache(maxsize=2048)
def _build_pandera_schema(
    table_key: str,
    signature: tuple[tuple[str, ColumnType, bool], ...],
    primary_key: tuple[str, ...],
) -> DataFrameSchema:
    columns = {
        name: PanderaColumn(dtype_for_column_type(col_type), nullable=nullable)
        for name, col_type, nullable in signature
    }

    checks: list[Check] = []
    if primary_key:
        checks.append(
            Check(
                lambda df, subset=primary_key: ~df.duplicated(subset=subset).any(),
                error=f"Duplicate primary key rows in {table_key}",
            )
        )

    return DataFrameSchema(
        columns,
        strict=True,
        coerce=True,
        checks=checks,
        name=table_key,
    )


def pandera_schema_from_table_schema(
    *,
    table_key: str,
    table_schema: TableSchema,
) -> DataFrameSchema:
    """Return a Pandera DataFrameSchema generated from a TableSchema.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    table_schema
        Source TableSchema.

    Returns
    -------
    DataFrameSchema
        Pandera schema derived from the TableSchema column definitions.
    """
    return _build_pandera_schema(
        table_key,
        _schema_signature(table_schema),
        table_schema.primary_key,
    )


__all__ = [
    "pandera_schema_from_table_schema",
]
