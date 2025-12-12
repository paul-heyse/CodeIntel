"""Connection management for DuckDB (macros removed).

All ingestion/DDL is now policy-backend + ibis driven; ingest macros are retired and
no longer applied here. This module remains for gateway wiring and schema/view setup.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

import duckdb

from codeintel.storage.schema import apply_all_schemas, assert_schema_alignment
from codeintel.storage.views import create_all_views

if TYPE_CHECKING:
    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.gateway.protocol import DuckDBConnection

__all__ = [
    "connect",
]


def connect(config: StorageConfig) -> DuckDBConnection:
    """
    Open a DuckDB connection using the provided configuration.

    Parameters
    ----------
    config
        Storage configuration controlling path, schema application, and validation.

    Returns
    -------
    DuckDBConnection
        Live DuckDB connection with optional schema/views applied.

    """
    if not config.read_only and config.db_path != Path(":memory:"):
        config.db_path.parent.mkdir(parents=True, exist_ok=True)
    con: DuckDBConnection = _open_primary_connection(config)
    _attach_history_if_needed(con, config)
    _apply_schema_and_views(con, config)
    _ensure_macros_and_schema(con, config)
    return con


def _open_primary_connection(config: StorageConfig) -> DuckDBConnection:
    """
    Open or attach the primary DuckDB connection.

    Returns
    -------
    DuckDBConnection
        Live connection to the requested database (file-backed or memory).
    """
    if not config.read_only and config.db_path != Path(":memory:") and not config.db_path.exists():
        # Bootstrap the database file with the latest storage version.
        #
        # We do this in a short-lived in-memory connection so that the gateway's
        # primary connection can open the file directly (avoiding cross-connection
        # file handle conflicts when the file remains attached).
        con = duckdb.connect(str(Path(":memory:")))
        db_path_str = str(config.db_path).replace("'", "''")
        con.execute(f"ATTACH DATABASE '{db_path_str}' AS main_db (STORAGE_VERSION 'latest')")
        con.execute("USE main_db")
        con.close()
        return duckdb.connect(str(config.db_path), read_only=False)
    return duckdb.connect(str(config.db_path), read_only=config.read_only)


def _attach_history_if_needed(con: DuckDBConnection, config: StorageConfig) -> None:
    """
    Attach history database when configured.

    Raises
    ------
    ValueError
        If attach_history is enabled without history_db_path.
    FileNotFoundError
        If the history database path does not exist.
    """
    if not config.attach_history:
        return
    if config.history_db_path is None:
        message = "attach_history requires history_db_path"
        raise ValueError(message)
    if not config.history_db_path.exists():
        message = f"History database not found: {config.history_db_path}"
        raise FileNotFoundError(message)
    history_path_str = str(config.history_db_path).replace("'", "''")
    con.execute(f"ATTACH DATABASE '{history_path_str}' AS history")


def _apply_schema_and_views(con: DuckDBConnection, config: StorageConfig) -> None:
    """Apply schemas and views when configured."""
    if config.apply_schema and not config.read_only:
        apply_all_schemas(con)
    if config.ensure_views and not config.read_only:
        create_all_views(con)


def _ensure_macros_and_schema(con: DuckDBConnection, config: StorageConfig) -> None:
    """Validate schema when configured (macros deprecated)."""
    if not config.read_only:
        warn(
            "Ingest macros are deprecated and no longer applied during gateway setup.",
            DeprecationWarning,
            stacklevel=2,
        )
    if config.validate_schema:
        assert_schema_alignment(
            con,
            include_views=config.ensure_views and not config.read_only,
            strict=True,
        )
