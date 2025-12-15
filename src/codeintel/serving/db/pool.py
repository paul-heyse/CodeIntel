"""Read-only DuckDB connection pool for serving.

DuckDB supports multiple connections; a single connection serializes queries.
This pool provides N read-only handles per worker for concurrent query execution.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from queue import Empty, LifoQueue
from typing import TYPE_CHECKING

from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.gateway.connection import connect

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.gateway.protocol import DuckDBConnection


@dataclass(frozen=True)
class DuckDBPoolConfig:
    """Pool configuration parameters.

    Parameters
    ----------
    size
        Number of connections in the pool.
    threads
        DuckDB threads per connection (None = default).
    memory_limit
        DuckDB memory limit per connection (e.g., "2GB").
    temp_directory
        Temporary directory for spilling.
    """

    size: int = 4
    threads: int | None = None
    memory_limit: str | None = None
    temp_directory: str | None = None


class DuckDBReadPool:
    """Thread-safe pool of read-only DuckDB connections.

    Parameters
    ----------
    db_path
        Path to DuckDB database file.
    cfg
        Pool configuration.
    """

    def __init__(self, db_path: Path, cfg: DuckDBPoolConfig) -> None:
        self._db_path = db_path
        self._cfg = cfg
        self._available: LifoQueue[DuckDBConnection] = LifoQueue()
        self._lock = threading.Lock()
        self._in_use: set[DuckDBConnection] = set()
        self._closing = False
        self._init_connections()

    def _open(self) -> DuckDBConnection:
        """Open a new read-only connection.

        Returns
        -------
        DuckDBConnection
            New read-only DuckDB connection.
        """
        duckdb_config: dict[str, bool | float | int | list[str] | str] = {}
        if self._cfg.threads is not None:
            duckdb_config["threads"] = self._cfg.threads
        if self._cfg.memory_limit is not None:
            duckdb_config["memory_limit"] = self._cfg.memory_limit
        if self._cfg.temp_directory is not None:
            duckdb_config["temp_directory"] = self._cfg.temp_directory
        return connect(StorageConfig.for_readonly(self._db_path), duckdb_config=duckdb_config)

    def _init_connections(self) -> None:
        """Pre-create pool connections."""
        for _ in range(max(1, self._cfg.size)):
            self._available.put(self._open())

    def acquire(self) -> DuckDBConnection:
        """Acquire a connection from the pool.

        Returns
        -------
        DuckDBConnection
            Read-only database connection.

        Raises
        ------
        RuntimeError
            If pool is closing.
        """
        with self._lock:
            if self._closing:
                msg = "Pool is closing"
                raise RuntimeError(msg)

        con = self._available.get()
        with self._lock:
            self._in_use.add(con)
        return con

    def release(self, con: DuckDBConnection) -> None:
        """Return a connection to the pool.

        Parameters
        ----------
        con
            Connection to release.
        """
        with self._lock:
            self._in_use.discard(con)
            closing = self._closing
        if closing:
            con.close()
            return
        self._available.put(con)

    def close_gracefully(self) -> None:
        """Mark pool as closing and drain available connections."""
        with self._lock:
            self._closing = True

        while True:
            try:
                con = self._available.get_nowait()
            except Empty:
                break
            con.close()


__all__ = ["DuckDBPoolConfig", "DuckDBReadPool"]
