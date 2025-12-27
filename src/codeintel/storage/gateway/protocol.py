"""Protocol and type definitions for the StorageGateway."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from codeintel.storage.duckdb.context import DuckDBContext

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from codeintel.storage.datasets import DatasetRegistry
    from codeintel.storage.duckdb.context import DuckDBContext
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
    from codeintel.storage.tracking.schema_catalog import SchemaCatalogTracking

__all__ = [
    "DuckDBBinderException",
    "DuckDBCatalogException",
    "DuckDBConnection",
    "DuckDBConnectionException",
    "DuckDBContext",
    "DuckDBDatabaseError",
    "DuckDBError",
    "DuckDBInvalidInputException",
    "DuckDBProgrammingError",
    "DuckDBRelation",
    "MinimalGateway",
    "SnapshotGatewayResolver",
    "StorageGateway",
]


if TYPE_CHECKING:
    from duckdb import (
        BinderException as DuckDBBinderException,
    )
    from duckdb import (
        CatalogException as DuckDBCatalogException,
    )
    from duckdb import (
        ConnectionException as DuckDBConnectionException,
    )
    from duckdb import (
        DatabaseError as DuckDBDatabaseError,
    )
    from duckdb import (
        DuckDBPyConnection as DuckDBConnection,
    )
    from duckdb import (
        DuckDBPyRelation as DuckDBRelation,
    )
    from duckdb import (
        Error as DuckDBError,
    )
    from duckdb import (
        InvalidInputException as DuckDBInvalidInputException,
    )
    from duckdb import (
        ProgrammingError as DuckDBProgrammingError,
    )
else:
    import duckdb

    DuckDBConnection = duckdb.DuckDBPyConnection
    DuckDBRelation = duckdb.DuckDBPyRelation
    DuckDBError = duckdb.Error
    DuckDBCatalogException = duckdb.CatalogException
    DuckDBConnectionException = duckdb.ConnectionException
    DuckDBDatabaseError = duckdb.DatabaseError
    DuckDBInvalidInputException = duckdb.InvalidInputException
    DuckDBProgrammingError = duckdb.ProgrammingError
    DuckDBBinderException = duckdb.BinderException


class MinimalGateway(Protocol):
    """Minimal protocol for DuckDB access without full accessor support.

    This protocol defines only the minimal interface needed by
    DuckDBPolicyBackend. Use StorageGateway for
    full accessor support.

    DuckDBPolicyBackend depends only on this protocol, accessing the
    underlying connection through the gateway reference. MinimalStorageGateway
    is the composition root that creates both.
    """

    @property
    def con(self) -> DuckDBConnection:
        """Return an open DuckDB connection."""
        ...

    @property
    def duckdb(self) -> DuckDBContext:
        """Return a DuckDB context bound to this connection."""
        ...

    @property
    def policy(self) -> DuckDBPolicyBackend:
        """Return the policy backend for DDL and mutation operations."""
        ...

    @property
    def ibis(self) -> IbisGateway:
        """Return the Ibis gateway for this connection."""
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
    schemas: SchemaCatalogTracking

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

    def execute(
        self,
        sql: str,
        params: Sequence[object] | Mapping[str, object] | None = None,
    ) -> DuckDBConnection:
        """Execute SQL against the underlying connection."""
        ...

    def register(self, name: str, obj: object) -> None:
        """Register a Python object in DuckDB."""
        ...

    def unregister(self, name: str) -> None:
        """Unregister a previously registered object."""
        ...

    def table(self, name: str) -> DuckDBRelation:
        """Return a relation for a fully qualified table or view name."""
        ...

    def relation_from_table_key(self, table_key: str) -> DuckDBRelation:
        """Return a relation for a fully qualified table key."""
        ...

    def export_database(self, *, directory: Path) -> None:
        """Export the database to a directory via DuckDB EXPORT DATABASE."""
        ...

    def import_database(self, *, directory: Path) -> None:
        """Import the database from a directory via DuckDB IMPORT DATABASE."""
        ...


SnapshotGatewayResolver = Callable[[str], StorageGateway]
"""Callable returning a StorageGateway for a given commit."""
