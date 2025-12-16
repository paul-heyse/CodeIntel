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

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.gateway.connection import connect

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.gateway.connection import DuckDBConnectConfig
    from codeintel.storage.gateway.protocol import DuckDBConnection


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
        return connect(self.config, duckdb_config=self.duckdb_config)

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


__all__ = ["DuckDBSession"]
