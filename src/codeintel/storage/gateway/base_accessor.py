"""Base class for DuckDB table accessor classes.

The gateway accessor classes are intentionally read-focused; all mutation/write
operations are routed through `codeintel.storage.warehouse.Warehouse`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sqlglot.expressions as exp

from codeintel.storage.constants import DUCKDB_DIALECT
from codeintel.storage.helpers.table_key import split_table_key

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
        schema, name = split_table_key(table_key)
        select_expr = exp.select("*").from_(
            exp.Table(this=exp.to_identifier(name), db=exp.to_identifier(schema))
        )
        return self.con.sql(select_expr.sql(dialect=DUCKDB_DIALECT))
