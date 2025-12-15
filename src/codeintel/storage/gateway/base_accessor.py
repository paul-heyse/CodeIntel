"""Base class for DuckDB table accessor classes.

This module provides a standardized base class that all table accessor classes
(CoreTables, GraphTables, AnalyticsTables, DocsViews) inherit from. The base
class provides common table access and row insertion operations, ensuring
consistent backend wiring across all accessor types.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.config.datasets.columns import load_columns_by_table
from codeintel.storage.gateway.insert_helpers import insert_rows

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from codeintel.storage.gateway.protocol import DuckDBConnection, DuckDBRelation, StorageGateway

__all__ = ["BaseTableAccessor"]


def get_table_columns(table_key: str) -> list[str]:
    """Return ordered column names for a specific table.

    Parameters
    ----------
    table_key
        Fully qualified table key (e.g., "core.ast_nodes").

    Returns
    -------
    list[str]
        Column names in storage order.
    """
    return list(load_columns_by_table().get(table_key, []))


def _normalize_to_mapping(
    row: Sequence[object],
    columns: Sequence[str],
    table_key: str,
) -> dict[str, object]:
    """Convert a positional row sequence into a mapping keyed by columns.

    Parameters
    ----------
    row
        Positional sequence of values.
    columns
        Column names in order.
    table_key
        Table key for error messages.

    Returns
    -------
    dict[str, object]
        Mapping of column names to values from the sequence.

    Raises
    ------
    ValueError
        If the row length does not match the expected columns.
    """
    if len(row) != len(columns):
        message = f"Row for {table_key} has {len(row)} values, expected {len(columns)}"
        raise ValueError(message)
    return {column: row[index] for index, column in enumerate(columns)}


def _get_columns_for_table(table_key: str) -> tuple[str, ...]:
    """Get column names from DatasetContract for a table.

    Parameters
    ----------
    table_key
        Fully qualified table key (e.g., "core.goids").

    Returns
    -------
    tuple[str, ...]
        Column names in storage order, or empty tuple if not found.
    """
    return tuple(get_table_columns(table_key))


@dataclass(frozen=True)
class BaseTableAccessor:
    """Base class providing common table access operations.

    Subclasses should define typed accessor methods that delegate to
    the base methods for consistent behavior.

    Parameters
    ----------
    gateway
        Storage gateway providing database access.
    """

    gateway: StorageGateway

    @property
    def con(self) -> DuckDBConnection:
        """Return the underlying DuckDB connection."""
        return self.gateway.con

    def _table(self, table_key: str) -> DuckDBRelation:
        """Return a relation for the given table key.

        Parameters
        ----------
        table_key
            Fully qualified table name (schema.table).

        Returns
        -------
        DuckDBRelation
            Relation bound to the table.
        """
        return self.con.table(table_key)

    def _insert_rows(
        self,
        table_key: str,
        rows: Iterable[Sequence[object]],
    ) -> None:
        """Insert rows into a table via the policy backend.

        Parameters
        ----------
        table_key
            Fully qualified table name (schema.table).
        rows
            Iterable of row tuples matching the table schema.
        """
        row_list = [tuple(row) for row in rows]
        if not row_list:
            return
        self.gateway.policy.bulk_insert(table_key, row_list)

    def _insert_normalized(
        self,
        table_key: str,
        rows: Iterable[Mapping[str, object] | Sequence[object]],
        *,
        columns: Sequence[str] | None = None,
    ) -> None:
        """Insert rows with automatic normalization.

        Handles both mapping and sequence row formats, normalizing
        sequences to mappings using columns from DatasetContract.

        Parameters
        ----------
        table_key
            Fully qualified table name (schema.table).
        rows
            Iterable of rows that can be either mappings (TypedDict)
            or sequences (tuples) matching the table schema.
        columns
            Optional explicit column list. When omitted, columns are
            derived from the DatasetContract schema.
        """
        resolved_columns = columns if columns is not None else _get_columns_for_table(table_key)

        def normalize(row: Mapping[str, object] | Sequence[object]) -> Mapping[str, object]:
            if isinstance(row, Mapping):
                return row
            return _normalize_to_mapping(row, resolved_columns, table_key)

        insert_rows(self.gateway, table_key, (normalize(r) for r in rows))
