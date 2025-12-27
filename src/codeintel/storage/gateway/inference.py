"""Inference-only storage gateway implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.core.schemas.provider import SchemaProvider
from codeintel.storage.backend import DuckDBSession
from codeintel.storage.datasets.registry import DatasetRegistry
from codeintel.storage.duckdb.context import DuckDBContext
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.exports import ExportService
from codeintel.storage.gateway.accessors import AnalyticsTables, CoreTables, DocsViews, GraphTables
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.tracking.asset_tracking import AssetTracking
from codeintel.storage.tracking.build_tracking import BuildTracking
from codeintel.storage.tracking.run_tracking import PipelineRunTracking
from codeintel.storage.tracking.schema_catalog import SchemaCatalogTracking

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.storage.gateway.protocol import DuckDBConnection, DuckDBRelation


def _empty_registry() -> DatasetRegistry:
    return DatasetRegistry(
        by_name={},
        by_table_key={},
        jsonl_datasets={},
        parquet_datasets={},
    )


def _inference_config() -> StorageConfig:
    return StorageConfig(
        db_path=Path(":memory:"),
        read_only=False,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
    )


@dataclass
class InferenceGateway:
    """Minimal StorageGateway for schema inference workflows."""

    con: DuckDBConnection
    schema_provider: SchemaProvider
    config: StorageConfig = field(default_factory=_inference_config)
    datasets: DatasetRegistry = field(default_factory=_empty_registry)
    duckdb: DuckDBContext = field(init=False)
    policy: DuckDBPolicyBackend = field(init=False)
    exports: ExportService = field(init=False)
    analytics: AnalyticsTables = field(init=False)
    assets: AssetTracking = field(init=False)
    build: BuildTracking = field(init=False)
    core: CoreTables = field(init=False)
    docs: DocsViews = field(init=False)
    graph: GraphTables = field(init=False)
    runs: PipelineRunTracking = field(init=False)
    schemas: SchemaCatalogTracking = field(init=False)

    def __post_init__(self) -> None:
        """Initialize accessor helpers after construction."""
        self.duckdb = DuckDBContext.from_connection(self.con)
        self.policy = DuckDBPolicyBackend(self, schema_provider=self.schema_provider)
        self.exports = ExportService(self)
        self.analytics = AnalyticsTables(self)
        self.assets = AssetTracking(self)
        self.build = BuildTracking(self)
        self.core = CoreTables(self)
        self.docs = DocsViews(self)
        self.graph = GraphTables(self)
        self.runs = PipelineRunTracking(self.con)
        self.schemas = SchemaCatalogTracking(self)

    def close(self) -> None:
        """Close the underlying connection."""
        self.con.close()

    def execute(
        self,
        sql: str,
        params: Sequence[object] | Mapping[str, object] | None = None,
    ) -> DuckDBConnection:
        """Execute SQL against the underlying connection.

        Returns
        -------
        DuckDBConnection
            Connection handle after execution.
        """
        if params is None:
            return self.con.execute(sql)
        return self.con.execute(sql, params)

    def register(self, name: str, obj: object) -> None:
        """Register a Python object in DuckDB."""
        self.con.register(name, obj)

    def unregister(self, name: str) -> None:
        """Unregister a previously registered object."""
        self.con.unregister(name)

    def table(self, name: str) -> DuckDBRelation:
        """Return a relation for the requested table/view.

        Returns
        -------
        DuckDBRelation
            Relation bound to the requested table/view.
        """
        return self.con.table(name)

    def relation_from_table_key(self, table_key: str) -> DuckDBRelation:
        """Return a relation for a fully qualified table key.

        Returns
        -------
        DuckDBRelation
            Relation bound to the requested table/view.
        """
        return self.con.table(table_key)

    def export_database(self, *, directory: Path) -> None:
        """Export the database to a directory via DuckDB EXPORT DATABASE."""
        DuckDBSession.export_database(self.con, directory=directory)

    def import_database(self, *, directory: Path) -> None:
        """Import the database from a directory via DuckDB IMPORT DATABASE."""
        DuckDBSession.import_database(self.con, directory=directory)


__all__ = ["InferenceGateway"]
