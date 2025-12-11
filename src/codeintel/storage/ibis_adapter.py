"""Ibis adapter for DuckDB-backed storage gateways."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import ibis
import ibis.expr.types as it

from codeintel.storage.gateway.protocol import StorageGateway

if TYPE_CHECKING:
    from ibis.backends.duckdb import Backend as DuckDBBackend

__all__ = ["IbisGateway"]


class IbisGateway:
    """Expose an Ibis backend bound to a `StorageGateway`."""

    def __init__(self, gateway: StorageGateway) -> None:
        self._gateway = gateway

    @cached_property
    def con(self) -> DuckDBBackend:
        """Return an Ibis backend that reuses the gateway DuckDB connection."""
        return ibis.duckdb.connect(con=self._gateway.con)

    def table(self, table_name: str) -> it.Table:
        """Return an Ibis table expression for a fully qualified table."""
        return self.con.table(table_name)

    def view(self, view_name: str) -> it.Table:
        """Alias for `table` for semantic clarity when accessing views."""
        return self.table(view_name)

    def sql(self, raw_sql: str) -> it.Table:
        """Execute raw SQL through Ibis and return the resulting table expression."""
        return self.con.sql(raw_sql)
