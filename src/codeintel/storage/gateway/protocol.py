"""Protocol and type definitions for the StorageGateway."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol

import duckdb

if TYPE_CHECKING:
    from codeintel.storage.datasets import DatasetRegistry
    from codeintel.storage.gateway.accessors import (
        AnalyticsTables,
        CoreTables,
        DocsViews,
        GraphTables,
    )
    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.tracking import PipelineRunTracking
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
    "SnapshotGatewayResolver",
    "StorageGateway",
]

# DuckDB type aliases for consistent usage across the codebase
DuckDBConnection = duckdb.DuckDBPyConnection
DuckDBRelation = duckdb.DuckDBPyRelation
DuckDBError = duckdb.Error
DuckDBCatalogException = duckdb.CatalogException
DuckDBConnectionException = duckdb.ConnectionException
DuckDBDatabaseError = duckdb.DatabaseError
DuckDBInvalidInputException = duckdb.InvalidInputException
DuckDBProgrammingError = duckdb.ProgrammingError
DuckDBBinderException = duckdb.BinderException


class StorageGateway(Protocol):
    """Expose DuckDB access along with dataset registry metadata."""

    analytics: AnalyticsTables
    build: BuildTracking
    config: StorageConfig
    core: CoreTables
    datasets: DatasetRegistry
    docs: DocsViews
    graph: GraphTables
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


SnapshotGatewayResolver = Callable[[str], StorageGateway]
"""Callable returning a StorageGateway for a given commit."""
