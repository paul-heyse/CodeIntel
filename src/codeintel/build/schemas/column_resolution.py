"""Deferred column resolution helpers for schema-derived materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.core.schemas.service import SchemaService


@dataclass(frozen=True, slots=True)
class DeferredColumns:
    """Placeholder for columns resolved at execution time."""

    table_key: str


def deferred_columns_for_table_key(table_key: str) -> DeferredColumns:
    """Return a deferred columns placeholder for a table key.

    Returns
    -------
    DeferredColumns
        Deferred columns placeholder for the table key.
    """
    return DeferredColumns(table_key=table_key)


def resolve_columns(
    columns: tuple[str, ...] | DeferredColumns,
    *,
    schema_service: SchemaService,
) -> tuple[str, ...]:
    """Resolve columns from a SchemaService when deferred.

    Returns
    -------
    tuple[str, ...]
        Resolved column names, or an empty tuple if the schema is missing.
    """
    if isinstance(columns, DeferredColumns):
        schema = schema_service.get_table_schema(columns.table_key)
        if schema is None:
            return ()
        return tuple(column.name for column in schema.columns)
    return columns


__all__ = [
    "DeferredColumns",
    "deferred_columns_for_table_key",
    "resolve_columns",
]
