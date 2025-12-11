"""Generic insert helpers backed by the generated registry.

These helpers convert TypedDict row models into schema-ordered tuples and
delegate to the existing macro_insert_rows bulk insertion utility. They
provide a data-driven alternative to the generated insert mixins while
keeping runtime behavior identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, TypedDict, cast

from codeintel.storage.gateway.registry_generated import TABLE_REGISTRY
from codeintel.storage.helpers.db import macro_insert_rows

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from codeintel.storage.gateway.protocol import DuckDBConnection

__all__ = ["insert_one", "insert_rows"]


class TableMetadata(TypedDict):
    """Generated table metadata entry."""

    table: str
    columns: list[str]


TABLE_REGISTRY_TYPED: Final[Mapping[str, TableMetadata]] = cast(
    "Mapping[str, TableMetadata]", TABLE_REGISTRY
)


def _lookup_metadata(table_key: str) -> TableMetadata:
    """Return registry metadata for a table or raise if missing.

    Raises
    ------
    ValueError
        If the table key is not present in the registry.

    Returns
    -------
    TableMetadata
        Registry entry for the provided table.
    """
    metadata = TABLE_REGISTRY_TYPED.get(table_key)
    if metadata is None:
        message = f"Unknown table key: {table_key}"
        raise ValueError(message)
    return metadata


def _normalize_row(
    row: Mapping[str, object], columns: Sequence[str], table_key: str
) -> tuple[object, ...]:
    """Transform a mapping row into an ordered tuple according to columns.

    Raises
    ------
    ValueError
        If the mapping is missing any required column.

    Returns
    -------
    tuple[object, ...]
        Values ordered per the table schema.
    """
    try:
        return tuple(row[column] for column in columns)
    except KeyError as exc:
        message = f"Missing column {exc.args[0]} for {table_key}"
        raise ValueError(message) from exc


def insert_rows(
    con: DuckDBConnection,
    table_key: str,
    rows: Iterable[Mapping[str, object]],
    executor: Callable[[DuckDBConnection, str, Iterable[tuple[object, ...]]], None]
    | None = macro_insert_rows,
) -> None:
    """Insert mapping rows into the given table using registry-driven ordering.

    Parameters
    ----------
    con
        DuckDB connection.
    table_key
        Fully qualified table name (schema.table).
    rows
        Iterable of mapping-based rows (e.g., TypedDict models) whose keys
        align with the table's schema columns.
    executor
        Optional callable to execute the inserts (defaults to macro_insert_rows).
    """
    metadata = _lookup_metadata(table_key)
    columns = metadata["columns"]
    normalized_rows = [_normalize_row(row, columns, table_key) for row in rows]
    if not normalized_rows:
        return
    insert_executor = executor or macro_insert_rows
    insert_executor(con, metadata["table"], normalized_rows)


def insert_one(
    con: DuckDBConnection,
    table_key: str,
    row: Mapping[str, object],
    executor: Callable[[DuckDBConnection, str, Iterable[tuple[object, ...]]], None]
    | None = macro_insert_rows,
) -> None:
    """Insert a single mapping row into the given table."""
    insert_rows(con, table_key, (row,), executor=executor)
