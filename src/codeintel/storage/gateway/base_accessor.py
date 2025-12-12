"""Base class for DuckDB table accessor classes.

This module provides a standardized base class that all table accessor classes
(CoreTables, GraphTables, AnalyticsTables, DocsViews) inherit from. The base
class provides common table access and row insertion operations, ensuring
consistent backend wiring across all accessor types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from codeintel.storage.gateway.protocol import DuckDBConnection, DuckDBRelation, StorageGateway

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
