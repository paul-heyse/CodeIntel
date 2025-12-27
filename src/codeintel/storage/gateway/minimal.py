"""Minimal gateway wrapper for raw DuckDB connections.

This module provides a lightweight gateway adapter that wraps a raw DuckDB
connection for limited use cases like bootstrap and migrations.

Architecture Note
-----------------
MinimalStorageGateway is the composition root for DuckDBPolicyBackend and
DuckDBContext. Both classes depend only on the MinimalGateway protocol and
access the underlying connection through the gateway reference. This avoids
circular imports while keeping all imports at the top level.

Warning
-------
This class intentionally does NOT implement the full StorageGateway protocol.
It provides only connection-level access. Accessor properties raise
NotImplementedError if accessed. Use the full gateway (open_gateway) when
you need accessor functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.storage.duckdb.context import DuckDBContext
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.exports import ExportService
from codeintel.storage.ibis_adapter import IbisGateway

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from duckdb import DuckDBPyConnection, DuckDBPyRelation

    from codeintel.core.schemas.provider import SchemaProvider
    from codeintel.storage.datasets import DatasetRegistry
    from codeintel.storage.gateway.accessors import (
        AnalyticsTables,
        CoreTables,
        DocsViews,
        GraphTables,
    )
    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.tracking import PipelineRunTracking
    from codeintel.storage.tracking.asset_tracking import AssetTracking
    from codeintel.storage.tracking.build_tracking import BuildTracking

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
    return (
        f"MinimalStorageGateway.{name} is not available. "
        "Use open_gateway() for full accessor support."
    )


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
        self._duckdb: DuckDBContext | None = None
        self._policy: DuckDBPolicyBackend | None = None
        self._ibis: IbisGateway | None = None
        self._schema_provider = schema_provider
        self.exports = ExportService(self)

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
    def duckdb(self) -> DuckDBContext:
        """Return a DuckDB context bound to this connection."""
        if self._duckdb is None:
            self._duckdb = DuckDBContext.from_connection(self._con)
        return self._duckdb

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
    def ibis(self) -> IbisGateway:
        """Return an Ibis gateway bound to this connection."""
        if self._ibis is None:
            self._ibis = IbisGateway(self)
        return self._ibis

    def relation_from_table_key(self, table_key: str) -> DuckDBPyRelation:
        """Return a relation for a fully qualified table key.

        Parameters
        ----------
        table_key
            Fully qualified table key to resolve.

        Returns
        -------
        DuckDBPyRelation
            Relation bound to the requested table.
        """
        return self._con.table(table_key)

    # -------------------------------------------------------------------------
    # Unsupported accessor properties
    # -------------------------------------------------------------------------

    @property
    def analytics(self) -> AnalyticsTables:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("analytics"))

    @property
    def assets(self) -> AssetTracking:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("assets"))

    @property
    def build(self) -> BuildTracking:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("build"))

    @property
    def config(self) -> StorageConfig:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("config"))

    @property
    def core(self) -> CoreTables:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("core"))

    @property
    def datasets(self) -> DatasetRegistry:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("datasets"))

    @property
    def docs(self) -> DocsViews:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("docs"))

    @property
    def graph(self) -> GraphTables:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("graph"))

    @property
    def runs(self) -> PipelineRunTracking:
        """Raise NotImplementedError - use full gateway for accessor access."""
        raise NotImplementedError(_unsupported_accessor_msg("runs"))

    # -------------------------------------------------------------------------
    # Connection methods
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        if self._ibis is not None:
            self._ibis.close()
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
