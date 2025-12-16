"""DuckDB session lifecycle wrapper.

This module provides a minimal session abstraction that can evolve into the
single place where we manage:
- connection open/read-only connections
- extension + secret management
- attach/export/import helpers
- concurrency guardrails (single-writer discipline)

Today it is intentionally thin and delegates to `codeintel.storage.gateway.connection.connect`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from codeintel.storage.gateway.connection import connect

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from pathlib import Path

    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.gateway.connection import DuckDBConnectConfig
    from codeintel.storage.gateway.protocol import DuckDBConnection

_INIT_SQL_ENV = "CODEINTEL_DUCKDB_INIT_SQL"


@dataclass(frozen=True, slots=True)
class DuckDBSession:
    """Create and manage DuckDB connections for storage operations.

    Parameters
    ----------
    config
        Storage configuration controlling the database path and bootstrap behaviors.
    duckdb_config
        Optional DuckDB client configuration (threads, memory limit, etc.).
    """

    config: StorageConfig
    duckdb_config: DuckDBConnectConfig | None = None

    def open(self) -> DuckDBConnection:
        """Open a new DuckDB connection for this session.

        Returns
        -------
        DuckDBConnection
            Open DuckDB connection.
        """
        con = connect(self.config, duckdb_config=self.duckdb_config)
        _run_init_sql_from_env(con)
        return con

    def open_reader(self) -> DuckDBConnection:
        """Open a new read-only connection to the same database.

        Returns
        -------
        DuckDBConnection
            Read-only DuckDB connection.
        """
        cfg = replace(
            self.config,
            read_only=True,
            apply_schema=False,
            ensure_views=False,
            validate_schema=False,
        )
        con = connect(cfg, duckdb_config=self.duckdb_config)
        _run_init_sql_from_env(con)
        return con

    def connect(self) -> AbstractContextManager[DuckDBConnection]:
        """Return a context manager that yields an open connection.

        Returns
        -------
        AbstractContextManager[DuckDBConnection]
            Context manager that opens and closes the connection.
        """

        class _ConnCtx:
            def __init__(self, session: DuckDBSession) -> None:
                self._session = session
                self._con: DuckDBConnection | None = None

            def __enter__(self) -> DuckDBConnection:
                self._con = self._session.open()
                return self._con

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                if self._con is not None:
                    self._con.close()

        return _ConnCtx(self)

    @staticmethod
    def attach_database(con: DuckDBConnection, *, db_path: Path, alias: str) -> None:
        """Attach a DuckDB database file to an existing connection.

        Parameters
        ----------
        con
            Connection to attach onto.
        db_path
            Path to a DuckDB database file.
        alias
            Alias name used to refer to the attached database.
        """
        escaped_path = str(db_path).replace("'", "''")
        escaped_alias = alias.replace('"', '""')
        con.execute(f"ATTACH DATABASE '{escaped_path}' AS \"{escaped_alias}\"")

    @staticmethod
    def export_database(con: DuckDBConnection, *, directory: Path) -> None:
        """Export the current database to a directory via DuckDB EXPORT DATABASE.

        Parameters
        ----------
        con
            Connection to export from.
        directory
            Directory to write the export into.
        """
        directory.mkdir(parents=True, exist_ok=True)
        escaped_dir = str(directory).replace("'", "''")
        con.execute(f"EXPORT DATABASE '{escaped_dir}'")

    @staticmethod
    def import_database(con: DuckDBConnection, *, directory: Path) -> None:
        """Import a database directory via DuckDB IMPORT DATABASE.

        Parameters
        ----------
        con
            Connection to import into.
        directory
            Directory previously created by EXPORT DATABASE.
        """
        escaped_dir = str(directory).replace("'", "''")
        con.execute(f"IMPORT DATABASE '{escaped_dir}'")


__all__ = ["DuckDBSession"]


def _run_init_sql_from_env(con: DuckDBConnection) -> None:
    """Execute optional initialization SQL configured by environment.

    The environment variable `CODEINTEL_DUCKDB_INIT_SQL` can contain either:
    - a JSON array of SQL statements, or
    - a newline-delimited string of SQL statements.

    Raises
    ------
    TypeError
        If the JSON payload is not an array of strings.
    ValueError
        If the JSON payload cannot be decoded.
    """
    raw = os.environ.get(_INIT_SQL_ENV, "").strip()
    if not raw:
        return

    statements: list[str]
    if raw.lstrip().startswith("["):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"Invalid {_INIT_SQL_ENV} JSON: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            msg = f"{_INIT_SQL_ENV} must be a JSON array of strings"
            raise TypeError(msg)
        statements = [item.strip() for item in payload if item.strip()]
    else:
        statements = [line.strip() for line in raw.splitlines() if line.strip()]

    for stmt in statements:
        con.execute(stmt)
