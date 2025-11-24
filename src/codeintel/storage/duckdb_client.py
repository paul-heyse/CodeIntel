"""DuckDB client utilities for the CodeIntel metadata warehouse."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import duckdb

from codeintel.storage.schemas import apply_all_schemas, assert_schema_alignment
from codeintel.storage.views import create_all_views

log = logging.getLogger(__name__)


@dataclass
class DuckDBConfig:
    """Configuration for connecting to the CodeIntel DuckDB database."""

    db_path: Path
    read_only: bool = False
    threads: int | None = None  # PRAGMA threads


class DuckDBClient:
    """Thin wrapper around a DuckDB connection."""

    def __init__(self, cfg: DuckDBConfig) -> None:
        self.cfg = cfg
        self._con: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """
        Establish a DuckDB connection, applying schemas on first connect.

        Returns
        -------
        duckdb.DuckDBPyConnection
            Live connection configured per `self.cfg`.
        """
        if self._con is not None:
            return self._con

        db_path = self.cfg.db_path
        if not self.cfg.read_only:
            db_path.parent.mkdir(parents=True, exist_ok=True)

        log.info("Connecting to DuckDB at %s (read_only=%s)", db_path, self.cfg.read_only)
        con = duckdb.connect(str(db_path), read_only=self.cfg.read_only)

        # Apply schema migrations (create schemas + tables) for read-write connections.
        if not self.cfg.read_only:
            apply_all_schemas(con)

        if self.cfg.threads is not None:
            con.execute(f"PRAGMA threads={int(self.cfg.threads)};")

        self._con = con
        return self._con

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        """
        Expose the lazily created connection.

        Returns
        -------
        duckdb.DuckDBPyConnection
            Active DuckDB connection.
        """
        return self.connect()

    def close(self) -> None:
        """Close the current connection if it exists."""
        if self._con is not None:
            log.info("Closing DuckDB connection to %s", self.cfg.db_path)
            self._con.close()
            self._con = None

    # Context manager helpers
    def __enter__(self) -> duckdb.DuckDBPyConnection:
        """
        Open the connection when entering a context manager.

        Returns
        -------
        duckdb.DuckDBPyConnection
            Active DuckDB connection.
        """
        return self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the connection when exiting a context manager."""
        self.close()


def get_connection(cfg: DuckDBConfig) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection using a provided configuration.

    Parameters
    ----------
    cfg:
        Connection configuration to use.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Live DuckDB connection.
    """
    client = DuckDBClient(cfg)
    return client.con


def connect_with_schema(
    db_path: Path,
    *,
    read_only: bool = False,
    apply_schema: bool = False,
    ensure_views: bool = False,
    validate_schema: bool = True,
) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection and apply/validate schemas in one step.

    Parameters
    ----------
    db_path:
        Path to the DuckDB database.
    read_only:
        Whether to open in read-only mode (skips DDL when True).
    ensure_views:
        Whether to (re)create docs views after schema application.
    apply_schema:
        When True, apply DDL (drop/create) using TABLE_SCHEMAS; for read-only connections
        this is ignored.
    validate_schema:
        When True, assert that the live schema matches TABLE_SCHEMAS.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Live connection configured per the provided options.
    """
    if not read_only:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path), read_only=read_only)
    if apply_schema and not read_only:
        apply_all_schemas(con)
    if validate_schema:
        assert_schema_alignment(con, strict=True)
    if ensure_views and not read_only:
        create_all_views(con)
    return con
