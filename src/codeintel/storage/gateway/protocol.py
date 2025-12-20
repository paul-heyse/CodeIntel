"""Protocol and type definitions for the StorageGateway."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from codeintel.storage.datasets import DatasetRegistry
    from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
    from codeintel.storage.exports import ExportService
    from codeintel.storage.gateway.accessors import (
        AnalyticsTables,
        CoreTables,
        DocsViews,
        GraphTables,
    )
    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.ibis_adapter import IbisGateway
    from codeintel.storage.tracking import PipelineRunTracking
    from codeintel.storage.tracking.asset_tracking import AssetTracking
    from codeintel.storage.tracking.build_tracking import BuildTracking

__all__ = [
    "DuckDBBinderException",
    "DuckDBCatalogException",
    "DuckDBConnection",
    "DuckDBConnectionException",
    "DuckDBDatabaseError",
    "DuckDBError",
    "DuckDBInvalidInputException",
    "DuckDBProgrammingError",
    "DuckDBRelation",
    "ExportService",
    "MinimalGateway",
    "SnapshotGatewayResolver",
    "StorageGateway",
]


if TYPE_CHECKING:
    import duckdb

    type DuckDBConnection = duckdb.DuckDBPyConnection
    type DuckDBRelation = duckdb.DuckDBPyRelation
    type DuckDBError = duckdb.Error
    type DuckDBCatalogException = duckdb.CatalogException
    type DuckDBConnectionException = duckdb.ConnectionException
    type DuckDBDatabaseError = duckdb.DatabaseError
    type DuckDBInvalidInputException = duckdb.InvalidInputException
    type DuckDBProgrammingError = duckdb.ProgrammingError
    type DuckDBBinderException = duckdb.BinderException
else:
    type DuckDBConnection = object
    type DuckDBRelation = object
    type DuckDBError = Exception
    type DuckDBCatalogException = Exception
    type DuckDBConnectionException = Exception
    type DuckDBDatabaseError = Exception
    type DuckDBInvalidInputException = Exception
    type DuckDBProgrammingError = Exception
    type DuckDBBinderException = Exception


class MinimalGateway(Protocol):
    """Minimal protocol for DuckDB access without full accessor support.

    This protocol defines only the minimal interface needed by
    DuckDBPolicyBackend and IbisGateway. Use StorageGateway for
    full accessor support.

    Both IbisGateway and DuckDBPolicyBackend depend only on this protocol,
    accessing each other through the gateway reference. MinimalStorageGateway
    is the composition root that creates both.
    """

    @property
    def con(self) -> DuckDBConnection:
        """Return an open DuckDB connection."""
        ...

    @property
    def ibis(self) -> IbisGateway:
        """Return an Ibis gateway bound to this connection."""
        ...

    @property
    def policy(self) -> DuckDBPolicyBackend:
        """Return the policy backend for DDL and mutation operations."""
        ...


class StorageGateway(MinimalGateway, Protocol):
    """Expose DuckDB access along with dataset registry metadata."""

    analytics: AnalyticsTables
    assets: AssetTracking
    build: BuildTracking
    config: StorageConfig
    core: CoreTables
    datasets: DatasetRegistry
    docs: DocsViews
    exports: ExportService
    graph: GraphTables
    ibis: IbisGateway
    policy: DuckDBPolicyBackend
    runs: PipelineRunTracking

    @property
    def con(self) -> DuckDBConnection:
        """
        Return an open DuckDB connection.

        Returns
        -------
        DuckDBConnection
            Live connection bound to the configured database.
        """
        ...

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        ...

    def execute(self, sql: str, params: Sequence[object] | None = None) -> DuckDBConnection:
        """Execute SQL against the underlying connection."""
        ...

    def table(self, name: str) -> DuckDBRelation:
        """Return a relation for a fully qualified table or view name."""
        ...

    def export_database(self, *, directory: Path) -> None:
        """Export the database to a directory via DuckDB EXPORT DATABASE."""
        ...

    def import_database(self, *, directory: Path) -> None:
        """Import the database from a directory via DuckDB IMPORT DATABASE."""
        ...


SnapshotGatewayResolver = Callable[[str], StorageGateway]
"""Callable returning a StorageGateway for a given commit."""
