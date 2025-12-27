"""Typed facade for Ibis access via DuckDB connections.

This module centralizes "Ibis backend from gateway connection" logic so call
sites do not depend on gateway-level Ibis adapters.
"""

from __future__ import annotations

import ibis
import ibis.backends.duckdb
import ibis.expr.types as ir

from codeintel.storage.gateway.protocol import MinimalGateway
from codeintel.storage.helpers.table_key import TableKey, split_table_key

_BACKEND_CACHE: dict[int, ibis.backends.duckdb.Backend] = {}

__all__ = ["backend", "table"]


def backend(gateway: MinimalGateway) -> ibis.backends.duckdb.Backend:
    """Return an Ibis DuckDB backend bound to the gateway connection.

    Parameters
    ----------
    gateway
        Storage gateway providing the DuckDB connection.

    Returns
    -------
    ibis.backends.duckdb.Backend
        Ibis backend bound to the gateway connection.
    """
    key = id(gateway.con)
    cached = _BACKEND_CACHE.get(key)
    if cached is not None:
        return cached
    backend_conn = ibis.duckdb.from_connection(gateway.con)
    _BACKEND_CACHE[key] = backend_conn
    return backend_conn


def table(gateway: MinimalGateway, table_key: TableKey) -> ir.Table:
    """Return an Ibis table expression for a fully qualified table key.

    Parameters
    ----------
    gateway
        Storage gateway providing the DuckDB connection.
    table_key
        Fully qualified table/view key (e.g., ``analytics.function_metrics``).

    Returns
    -------
    ir.Table
        Ibis table expression for the requested object.
    """
    ibis_backend = backend(gateway)
    if "." in table_key:
        database, name = split_table_key(table_key)
        return ibis_backend.table(name, database=database)
    return ibis_backend.table(table_key)
