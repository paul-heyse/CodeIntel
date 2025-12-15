"""Connection management for DuckDB.

This module owns low-level connection wiring (open/attach history) and optional
schema application. View materialization and other policy operations are handled
at the gateway layer via DuckDBPolicyBackend.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from codeintel.storage.schema import apply_all_schemas

if TYPE_CHECKING:
    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.gateway.protocol import DuckDBConnection

DuckDBConnectConfigValue = bool | float | int | list[str] | str
DuckDBConnectConfig = dict[str, DuckDBConnectConfigValue]

__all__ = [
    "connect",
]


def connect(
    config: StorageConfig, *, duckdb_config: DuckDBConnectConfig | None = None
) -> DuckDBConnection:
    """
    Open a DuckDB connection using the provided configuration.

    Parameters
    ----------
    config
        Storage configuration controlling path, schema application, and validation.
    duckdb_config
        Optional DuckDB connection configuration (e.g., threads, memory_limit).

    Returns
    -------
    DuckDBConnection
        Live DuckDB connection with optional schema/views applied.

    """
    if not config.read_only and config.db_path != Path(":memory:"):
        config.db_path.parent.mkdir(parents=True, exist_ok=True)
    con: DuckDBConnection = _open_primary_connection(config, duckdb_config=duckdb_config)
    _attach_history_if_needed(con, config)
    _apply_schema(con, config)
    return con


def _open_primary_connection(
    config: StorageConfig, *, duckdb_config: DuckDBConnectConfig | None = None
) -> DuckDBConnection:
    """
    Open or attach the primary DuckDB connection.

    Returns
    -------
    DuckDBConnection
        Live connection to the requested database (file-backed or memory).
    """
    cfg = duckdb_config
    if not config.read_only and config.db_path != Path(":memory:") and not config.db_path.exists():
        con = duckdb.connect(str(Path(":memory:")))
        db_path_str = str(config.db_path).replace("'", "''")
        con.execute(f"ATTACH DATABASE '{db_path_str}' AS main_db (STORAGE_VERSION 'latest')")
        con.execute("USE main_db")
        con.close()
        if cfg is None:
            return duckdb.connect(str(config.db_path), read_only=False)
        return duckdb.connect(str(config.db_path), read_only=False, config=cfg)

    if cfg is None:
        return duckdb.connect(str(config.db_path), read_only=config.read_only)
    return duckdb.connect(str(config.db_path), read_only=config.read_only, config=cfg)


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


def _apply_schema(con: DuckDBConnection, config: StorageConfig) -> None:
    """Apply schemas when configured."""
    if config.apply_schema and not config.read_only:
        apply_all_schemas(con)
