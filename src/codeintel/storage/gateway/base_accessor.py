"""Base class for DuckDB table accessor classes.

The gateway accessor classes are intentionally read-focused; all mutation/write
operations are routed through `codeintel.storage.warehouse.Warehouse`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import DuckDBConnection, DuckDBRelation, StorageGateway

__all__ = ["BaseTableAccessor"]


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
        return self.gateway.relation_from_table_key(table_key)
