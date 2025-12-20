"""Minimal gateway wrapper for raw DuckDB connections.

This module provides a lightweight gateway adapter that wraps a raw DuckDB
connection for limited use cases like bootstrap and migrations.

Architecture Note
-----------------
MinimalStorageGateway is the composition root for DuckDBPolicyBackend and
IbisGateway. Both classes depend only on the MinimalGateway protocol and
access each other through the gateway reference. This avoids circular imports
while keeping all imports at the top level.

Warning
-------
This class intentionally does NOT implement the full StorageGateway protocol.
It provides only connection-level access. Accessor properties raise
NotImplementedError if accessed. Use the full gateway (open_gateway) when
you need accessor functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.exports import ExportService
from codeintel.storage.ibis_adapter import IbisGateway

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from duckdb import DuckDBPyConnection, DuckDBPyRelation

    from codeintel.core.schemas.provider import SchemaProvider

__all__ = ["MinimalStorageGateway"]


def _unsupported_accessor_msg(name: str) -> str:
    """Create error message for unsupported accessor access.

    Parameters
    ----------
    name
        Name of the accessor.

    Returns
    -------
    str
        Error message for NotImplementedError.
    """
    return f"MinimalStorageGateway.{name} is not available. Use open_gateway() for full accessor support."


class MinimalStorageGateway:
    """Lightweight StorageGateway adapter for raw DuckDB connections.

    Provides just enough interface to satisfy DuckDBPolicyBackend and
    view creation without full gateway initialization overhead.

    Important
    ---------
    This class does NOT implement the full StorageGateway protocol.
    Accessor properties (analytics, assets, core, etc.) will raise
    NotImplementedError if accessed. Use this only for:

    - Schema bootstrap operations
    - Database migrations
    - Direct SQL operations via policy backend

    For full gateway functionality, use open_gateway() instead.

    Parameters
    ----------
    connection
        Raw DuckDB connection to wrap.

    Examples
    --------
    >>> import duckdb
    >>> from codeintel.storage.gateway.minimal import MinimalStorageGateway
    >>> con = duckdb.connect(":memory:")
    >>> gateway = MinimalStorageGateway(con)
    >>> gateway.policy.ensure_all_schemas()

    See Also
    --------
    open_gateway : Full gateway with accessor support.
    """

    def __init__(
        self,
        connection: DuckDBPyConnection,
        *,
        schema_provider: SchemaProvider | None = None,
    ) -> None:
        """Initialize minimal gateway with a DuckDB connection.

        Parameters
        ----------
        connection
            Raw DuckDB connection to wrap.
        schema_provider
            Optional schema provider for DDL and column-order enforcement.
        """
        self._con = connection
        self._ibis: IbisGateway | None = None
        self._policy: DuckDBPolicyBackend | None = None
        self._exports: ExportService | None = None
        self._schema_provider = schema_provider

    @property
    def con(self) -> DuckDBPyConnection:
        """Return the underlying DuckDB connection.

        Returns
        -------
        DuckDBPyConnection
            The wrapped DuckDB connection.
        """
        return self._con

    @property
    def ibis(self) -> IbisGateway:
        """Return an Ibis gateway bound to this connection.

        Lazily initializes the IbisGateway on first access.

        Returns
        -------
        IbisGateway
            Ibis gateway for expression-based queries.
        """
        if self._ibis is None:
            self._ibis = IbisGateway(self)
        return self._ibis

    @property
    def policy(self) -> DuckDBPolicyBackend:
        """Return the policy backend for this connection.

        Returns
        -------
        DuckDBPolicyBackend
            Policy backend for DDL and mutation operations.
        """
        if self._policy is None:
            self._policy = DuckDBPolicyBackend(self, schema_provider=self._schema_provider)
        return self._policy

    @property
    def exports(self) -> ExportService:
        """Return the export service for this connection.

        Returns
        -------
        ExportService
            Export service wrapper for this gateway.
        """
        if self._exports is None:
            self._exports = ExportService(self)
        return self._exports

    # -------------------------------------------------------------------------
    # Unsupported accessor properties
    # -------------------------------------------------------------------------

    @property
    def analytics(self) -> NoReturn:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("analytics"))

    @property
    def assets(self) -> NoReturn:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("assets"))

    @property
    def build(self) -> NoReturn:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("build"))

    @property
    def config(self) -> NoReturn:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("config"))

    @property
    def core(self) -> NoReturn:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("core"))

    @property
    def datasets(self) -> NoReturn:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("datasets"))

    @property
    def docs(self) -> NoReturn:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("docs"))

    @property
    def graph(self) -> NoReturn:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("graph"))

    @property
    def runs(self) -> NoReturn:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("runs"))

    # -------------------------------------------------------------------------
    # Connection methods
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._con.close()

    def execute(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> DuckDBPyConnection:
        """Execute SQL against the underlying connection.

        Parameters
        ----------
        sql
            SQL statement to execute.
        params
            Optional parameters for the query.

        Returns
        -------
        DuckDBPyConnection
            Connection representing the executed query.
        """
        return self._con.execute(sql, params)

    def table(self, name: str) -> DuckDBPyRelation:
        """Return a relation object for the specified table or view.

        Parameters
        ----------
        name
            Table or view name.

        Returns
        -------
        DuckDBPyRelation
            Relation bound to the requested table/view.
        """
        return self._con.table(name)

    def export_database(self, *, directory: Path) -> None:
        """Export the database to a directory via DuckDB EXPORT DATABASE."""
        directory.mkdir(parents=True, exist_ok=True)
        escaped_dir = str(directory).replace("'", "''")
        self._con.execute(f"EXPORT DATABASE '{escaped_dir}'")

    def import_database(self, *, directory: Path) -> None:
        """Import the database from a directory via DuckDB IMPORT DATABASE."""
        escaped_dir = str(directory).replace("'", "''")
        self._con.execute(f"IMPORT DATABASE '{escaped_dir}'")
