"""Gateway port protocol for storage operations.

This module defines the GatewayPort protocol that abstracts storage gateway
operations. Tests code to this protocol while using real DuckDB-backed
implementations per the Testing Charter.

Design Notes
------------
- Protocol methods mirror essential StorageGateway operations
- Real adapter is the production StorageGateway class
- No in-memory fakes; tests use real DuckDB (file or memory mode)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from duckdb import DuckDBPyConnection


@runtime_checkable
class GatewayPort(Protocol):
    """Protocol for storage gateway operations.

    Defines the interface tests use for database access. The production
    StorageGateway class satisfies this protocol.

    Attributes
    ----------
    con : DuckDBPyConnection
        Underlying database connection.
    """

    @property
    def con(self) -> DuckDBPyConnection:
        """Return the underlying database connection.

        Returns
        -------
        DuckDBPyConnection
            Active DuckDB connection.
        """
        ...

    def execute(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> DuckDBPyConnection:
        """Execute a SQL statement.

        Parameters
        ----------
        sql
            SQL statement to execute.
        params
            Optional parameters for parameterized query.

        Returns
        -------
        DuckDBPyConnection
            Connection for result fetching.
        """
        ...

    def executemany(
        self,
        sql: str,
        params: Sequence[Sequence[object]],
    ) -> None:
        """Execute a SQL statement with multiple parameter sets.

        Parameters
        ----------
        sql
            SQL statement with placeholders.
        params
            Sequence of parameter tuples.
        """
        ...

    def close(self) -> None:
        """Close the gateway connection."""
        ...

    def insert_modules(
        self,
        rows: Sequence[tuple[str, str, str, str, int]],
    ) -> None:
        """Insert module records.

        Parameters
        ----------
        rows
            Tuples of (repo, commit, module_path, module_hash, loc).
        """
        ...

    def insert_goids(
        self,
        rows: Sequence[tuple[str, str, str, str, str, str, int, int, str | None, str]],
    ) -> None:
        """Insert GOID records.

        Parameters
        ----------
        rows
            Tuples of GOID attributes per schema.
        """
        ...

    def export_to_parquet(self, table: str, path: Path) -> None:
        """Export a table to Parquet format.

        Parameters
        ----------
        table
            Table name (schema.table format).
        path
            Destination file path.
        """
        ...

    def import_from_parquet(self, table: str, path: Path) -> int:
        """Import a Parquet file into a table.

        Parameters
        ----------
        table
            Target table name (schema.table format).
        path
            Source file path.

        Returns
        -------
        int
            Number of rows imported.
        """
        ...


__all__ = ["GatewayPort"]
