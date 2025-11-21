# src/codeintel/storage/duckdb_client.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb

from .schemas import apply_all_schemas

log = logging.getLogger(__name__)


@dataclass
class DuckDBConfig:
    """
    Configuration for connecting to the CodeIntel DuckDB database.
    """

    db_path: Path
    read_only: bool = False
    threads: Optional[int] = None  # PRAGMA threads


class DuckDBClient:
    """
    Thin wrapper around a DuckDB connection that:
      - ensures the directory exists
      - applies all schema DDL on first connect
      - can be used as a context manager
    """

    def __init__(self, cfg: DuckDBConfig) -> None:
        self.cfg = cfg
        self._con: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self) -> duckdb.DuckDBPyConnection:
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
        return self.connect()

    def close(self) -> None:
        if self._con is not None:
            log.info("Closing DuckDB connection to %s", self.cfg.db_path)
            self._con.close()
            self._con = None

    # Context manager helpers
    def __enter__(self) -> duckdb.DuckDBPyConnection:
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def get_connection(db_path: Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Convenience helper for one-off use cases.

    Example:
        con = get_connection(Path("build/db/codeintel.duckdb"))
    """
    client = DuckDBClient(DuckDBConfig(db_path=db_path, read_only=read_only))
    return client.con
