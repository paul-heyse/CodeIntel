"""Ibis adapter for DuckDB-backed storage gateways."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import ibis
import ibis.expr.types as it

if TYPE_CHECKING:
    from ibis.backends.duckdb import Backend as DuckDBBackend

    from codeintel.storage.gateway.protocol import StorageGateway

__all__ = ["IbisGateway"]


class IbisGateway:
    """Expose an Ibis backend bound to a `StorageGateway`."""

    def __init__(self, gateway: StorageGateway) -> None:
        self._gateway = gateway

    @cached_property
    def con(self) -> DuckDBBackend:
        """
        Return an Ibis backend that reuses the gateway DuckDB connection.

        Returns
        -------
        DuckDBBackend
            Ibis backend bound to the DuckDB connection.
        """
        return ibis.duckdb.from_connection(self._gateway.con)

    def table(self, table_name: str) -> it.Table:
        """
        Return an Ibis table expression for a fully qualified table.

        Parameters
        ----------
        table_name
            Fully qualified table or view name (e.g., "analytics.function_metrics").

        Returns
        -------
        it.Table
            Ibis table expression for the requested object.

        Note
        ----
        Ibis 11+ requires the `database` parameter for schema-qualified names.
        This method automatically splits "schema.table" into the correct form.
        """
        if "." in table_name:
            database, name = table_name.split(".", 1)
            return self.con.table(name, database=database)
        return self.con.table(table_name)

    def view(self, view_name: str) -> it.Table:
        """
        Alias for `table` for semantic clarity when accessing views.

        Parameters
        ----------
        view_name
            Fully qualified view name.

        Returns
        -------
        it.Table
            Ibis table expression for the view.
        """
        return self.table(view_name)

    def sql(self, raw_sql: str) -> it.Table:
        """
        Execute raw SQL through Ibis and return the resulting table expression.

        Parameters
        ----------
        raw_sql
            SQL string to execute via Ibis.

        Returns
        -------
        it.Table
            Table expression backed by the SQL statement.
        """
        return self.con.sql(raw_sql)
