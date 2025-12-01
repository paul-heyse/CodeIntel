"""Shared DuckDB helpers for tests to ensure consistent macro availability."""

from __future__ import annotations

import duckdb

from codeintel.storage.gateway import StorageGateway, open_memory_gateway
from codeintel.storage.ingest_macros import ensure_ingest_macros, list_ingest_macros
from codeintel.storage.metadata_bootstrap import INGEST_MACROS

DuckDBConnection = duckdb.DuckDBPyConnection
MACROS_EXPECTED = {m.lower() for m in INGEST_MACROS.values()}


def memory_con_with_macros() -> DuckDBConnection:
    """
    Create an in-memory DuckDB connection with ingest macros registered.

    Returns
    -------
    DuckDBConnection
        Connection to an in-memory DuckDB instance with macros ensured.
    """
    con = duckdb.connect(database=":memory:")
    ensure_ingest_macros(con)
    return con


def gateway_with_macros(
    *,
    apply_schema: bool = True,
    ensure_views: bool = True,
    validate_schema: bool = True,
) -> StorageGateway:
    """
    Create an in-memory StorageGateway with schemas/views/macros ensured.

    Returns
    -------
    StorageGateway
        Gateway backed by an in-memory DuckDB connection with ingest macros present.

    Raises
    ------
    RuntimeError
        If ingest macros could not be registered.
    """
    gateway = open_memory_gateway(
        apply_schema=apply_schema,
        ensure_views=ensure_views,
        validate_schema=validate_schema,
    )
    ensure_ingest_macros(gateway.con)
    registered = list_ingest_macros(gateway.con)
    missing = MACROS_EXPECTED - registered
    if missing:
        ensure_ingest_macros(gateway.con)
        registered = list_ingest_macros(gateway.con)
        missing = MACROS_EXPECTED - registered
    if missing:
        gateway.close()
        message = f"Missing ingest macros on gateway: {sorted(missing)}"
        raise RuntimeError(message)
    return gateway
