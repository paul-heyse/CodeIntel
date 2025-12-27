"""DuckDB context wrapper for relation-first access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.duckdb.catalog import duckdb_default_catalog

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.storage.duckdb_types import DuckDBConnection, DuckDBRelation


@dataclass(frozen=True, slots=True)
class DuckDBContext:
    """Lightweight DuckDB context for connection-bound operations."""

    con: DuckDBConnection
    default_catalog: str | None = None

    @classmethod
    def from_connection(cls, con: DuckDBConnection) -> DuckDBContext:
        """Create a context with resolved default catalog metadata.

        Parameters
        ----------
        con
            DuckDB connection to wrap.

        Returns
        -------
        DuckDBContext
            Context bound to the provided connection.
        """
        return cls(con=con, default_catalog=duckdb_default_catalog(con))

    def execute(
        self,
        sql: str,
        params: Sequence[object] | Mapping[str, object] | None = None,
    ) -> DuckDBConnection:
        """Execute SQL with optional parameter binding.

        Parameters
        ----------
        sql
            SQL statement to execute.
        params
            Optional positional or named parameters.

        Returns
        -------
        DuckDBConnection
            Connection handle after execution.
        """
        if params is None:
            return self.con.execute(sql)
        return self.con.execute(sql, params)

    def register(self, name: str, obj: object) -> None:
        """Register a Python object with DuckDB.

        Parameters
        ----------
        name
            Name to register within DuckDB.
        obj
            Object to register for replacement scans.
        """
        self.con.register(name, obj)

    def unregister(self, name: str) -> None:
        """Unregister a previously registered object.

        Parameters
        ----------
        name
            Registered name to unregister.
        """
        self.con.unregister(name)

    def table(self, name: str) -> DuckDBRelation:
        """Return a relation for a fully qualified table or view name.

        Parameters
        ----------
        name
            Fully qualified table or view name.

        Returns
        -------
        DuckDBRelation
            Relation bound to the requested table/view.
        """
        return self.con.table(name)


__all__ = ["DuckDBContext"]
