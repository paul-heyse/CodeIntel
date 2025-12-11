"""Base class for DuckDB table accessor classes.

This module provides a standardized base class that all table accessor classes
(CoreTables, GraphTables, AnalyticsTables, DocsViews) inherit from. The base
class provides common table access and row insertion operations, ensuring
consistent backend wiring across all accessor types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.helpers.db import macro_insert_rows

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from codeintel.storage.gateway.protocol import DuckDBConnection, DuckDBRelation

__all__ = ["BaseTableAccessor"]


@dataclass(frozen=True)
class BaseTableAccessor:
    """Base class providing common table access operations.

    Subclasses should define typed accessor methods that delegate to
    the base methods for consistent behavior.

    Parameters
    ----------
    con
        DuckDB connection instance.
    """

    con: DuckDBConnection

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
        """Insert rows into a table using the schema-aware macro.

        This is the canonical method for bulk row insertion. It uses
        macro_insert_rows which validates rows against the table schema
        and pads missing columns with NULL values.

        Parameters
        ----------
        table_key
            Fully qualified table name (schema.table).
        rows
            Iterable of row tuples matching the table schema.
        """
        macro_insert_rows(self.con, table_key, rows)
