"""Table accessor classes for DuckDB schema access.

The gateway accessors provide a small, typed read surface over DuckDB relations.
All mutation/write operations are routed through `codeintel.storage.warehouse.Warehouse`
to preserve a single I/O boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.core.schemas.provider import FallbackSchemaProvider, MappingSchemaProvider
from codeintel.core.schemas.service import get_schema_service
from codeintel.serving.semantic.duckdb_scan_adapter import scan_parquet
from codeintel.storage.backend import DuckDBSession
from codeintel.storage.datasets.manifests import load_dataset_manifest
from codeintel.storage.datasets.paths import dataset_snapshot_dir
from codeintel.storage.duckdb.context import DuckDBContext
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.exports import ExportService
from codeintel.storage.gateway.base_accessor import BaseTableAccessor
from codeintel.storage.gateway.relation import relation_from_table_key as _relation_from_table_key
from codeintel.storage.tracking.asset_tracking import AssetTracking
from codeintel.storage.tracking.build_tracking import BuildTracking
from codeintel.storage.tracking.run_tracking import PipelineRunTracking
from codeintel.storage.tracking.schema_catalog import SchemaCatalogTracking

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.core.schemas.provider import SchemaProvider
    from codeintel.storage.datasets import DatasetRegistry
    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.gateway.protocol import DuckDBConnection, DuckDBRelation

__all__ = [
    "AnalyticsTables",
    "AssetTracking",
    "BaseTableAccessor",
    "BuildTracking",
    "CoreTables",
    "DocsViews",
    "DuckDBGateway",
    "GraphTables",
]


@dataclass(frozen=True, slots=True)
class _ParquetManifestContext:
    dataset_root_dir: Path
    table_key: str
    snapshot_id: str
    partition_columns: tuple[str, ...]
    files: tuple[str, ...]


def _schema_provider_for_gateway(*, datasets: DatasetRegistry) -> SchemaProvider:
    schemas = {
        table_key: contract.schema
        for table_key, contract in datasets.by_table_key.items()
        if contract.schema is not None and not contract.is_view
    }
    fallback = MappingSchemaProvider(schemas)
    try:
        service = get_schema_service()
    except RuntimeError:
        return fallback
    return FallbackSchemaProvider(primary=service.table_provider, fallback=fallback)


def _parquet_relation_for_manifest(
    con: DuckDBConnection,
    *,
    context: _ParquetManifestContext,
) -> DuckDBRelation:
    dataset_dir = dataset_snapshot_dir(
        context.dataset_root_dir,
        table_key=context.table_key,
        snapshot_id=context.snapshot_id,
    )
    if not dataset_dir.is_dir():
        msg = f"Dataset snapshot directory missing for {context.table_key}: {dataset_dir}"
        raise FileNotFoundError(msg)
    scan_paths = (
        [str(dataset_dir / file) for file in context.files] if context.files else [str(dataset_dir)]
    )
    return scan_parquet(
        con,
        scan_paths=scan_paths,
        hive_partitioning=bool(context.partition_columns),
        union_by_name=True,
    )


def _relation_for_table_key(
    con: DuckDBConnection,
    *,
    table_key: str,
    datasets: DatasetRegistry,
    config: StorageConfig,
) -> DuckDBRelation:
    dataset = datasets.by_table_key.get(table_key)
    if dataset is None or dataset.is_view:
        return _relation_from_table_key(con, table_key)

    dataset_root_dir = config.dataset_root_dir
    snapshot_id = config.commit
    if dataset_root_dir is None or snapshot_id is None:
        msg = f"Parquet-backed datasets require dataset_root_dir and commit (table={table_key})"
        raise RuntimeError(msg)

    manifest = datasets.dataset_manifest_for_table(table_key)
    if manifest is None:
        manifest = load_dataset_manifest(
            dataset_root=dataset_root_dir,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
        if manifest is None:
            msg = f"Dataset manifest missing for {table_key} at snapshot {snapshot_id}"
            raise FileNotFoundError(msg)
    if manifest.snapshot_id != snapshot_id:
        msg = (
            "Dataset manifest snapshot mismatch for "
            f"{table_key}: {manifest.snapshot_id} != {snapshot_id}"
        )
        raise ValueError(msg)

    context = _ParquetManifestContext(
        dataset_root_dir=dataset_root_dir,
        table_key=table_key,
        snapshot_id=manifest.snapshot_id,
        partition_columns=manifest.partition_columns or (),
        files=manifest.files,
    )
    return _parquet_relation_for_manifest(con, context=context)


@dataclass(frozen=True)
class CoreTables(BaseTableAccessor):
    """Read accessors for core schema tables."""

    def goids(self) -> DuckDBRelation:
        """Return the ``core.goids`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``core.goids``.
        """
        return self._table("core.goids")

    def file_state(self) -> DuckDBRelation:
        """Return the ``core.file_state`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``core.file_state``.
        """
        return self._table("core.file_state")

    def scip_occurrences(self) -> DuckDBRelation:
        """Return the ``core.scip_occurrences`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``core.scip_occurrences``.
        """
        return self._table("core.scip_occurrences")

    def modules(self) -> DuckDBRelation:
        """Return the ``core.modules`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``core.modules``.
        """
        return self._table("core.modules")

    def repo_map(self) -> DuckDBRelation:
        """Return the ``core.repo_map`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``core.repo_map``.
        """
        return self._table("core.repo_map")


@dataclass(frozen=True)
class GraphTables(BaseTableAccessor):
    """Read accessors for graph schema tables."""

    def call_graph_edges(self) -> DuckDBRelation:
        """Return the ``graph.call_graph_edges`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``graph.call_graph_edges``.
        """
        return self._table("graph.call_graph_edges")

    def call_graph_nodes(self) -> DuckDBRelation:
        """Return the ``graph.call_graph_nodes`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``graph.call_graph_nodes``.
        """
        return self._table("graph.call_graph_nodes")

    def import_graph_edges(self) -> DuckDBRelation:
        """Return the ``graph.import_graph_edges`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``graph.import_graph_edges``.
        """
        return self._table("graph.import_graph_edges")

    def symbol_use_edges(self) -> DuckDBRelation:
        """Return the ``graph.symbol_use_edges`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``graph.symbol_use_edges``.
        """
        return self._table("graph.symbol_use_edges")

    def cfg_blocks(self) -> DuckDBRelation:
        """Return the ``graph.cfg_blocks`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``graph.cfg_blocks``.
        """
        return self._table("graph.cfg_blocks")

    def cfg_edges(self) -> DuckDBRelation:
        """Return the ``graph.cfg_edges`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``graph.cfg_edges``.
        """
        return self._table("graph.cfg_edges")

    def dfg_edges(self) -> DuckDBRelation:
        """Return the ``graph.dfg_edges`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``graph.dfg_edges``.
        """
        return self._table("graph.dfg_edges")


@dataclass(frozen=True)
class DocsViews(BaseTableAccessor):
    """Accessors for docs schema views."""

    def call_graph_enriched(self) -> DuckDBRelation:
        """Return the ``docs.v_call_graph_enriched`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``docs.v_call_graph_enriched``.
        """
        return self._table("docs.v_call_graph_enriched")


@dataclass(frozen=True)
class AnalyticsTables(BaseTableAccessor):
    """Read accessors for analytics schema tables."""

    def function_types(self) -> DuckDBRelation:
        """Return the ``analytics.function_types`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.function_types``.
        """
        return self._table("analytics.function_types")

    def function_validation(self) -> DuckDBRelation:
        """Return the ``analytics.function_validation`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.function_validation``.
        """
        return self._table("analytics.function_validation")

    def test_catalog(self) -> DuckDBRelation:
        """Return the ``analytics.test_catalog`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.test_catalog``.
        """
        return self._table("analytics.test_catalog")

    def config_values(self) -> DuckDBRelation:
        """Return the ``analytics.config_values`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.config_values``.
        """
        return self._table("analytics.config_values")

    def static_diagnostics(self) -> DuckDBRelation:
        """Return the ``analytics.static_diagnostics`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.static_diagnostics``.
        """
        return self._table("analytics.static_diagnostics")

    def subsystems(self) -> DuckDBRelation:
        """Return the ``analytics.subsystems`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.subsystems``.
        """
        return self._table("analytics.subsystems")

    def subsystem_modules(self) -> DuckDBRelation:
        """Return the ``analytics.subsystem_modules`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.subsystem_modules``.
        """
        return self._table("analytics.subsystem_modules")


@dataclass
class DuckDBGateway:
    """Concrete StorageGateway implementation."""

    config: StorageConfig
    datasets: DatasetRegistry
    con: DuckDBConnection
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
        """Initialize table accessor instances after dataclass init."""
        self.duckdb = DuckDBContext.from_connection(self.con)
        schema_provider = _schema_provider_for_gateway(datasets=self.datasets)
        self.policy = DuckDBPolicyBackend(self, schema_provider=schema_provider)
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

        Parameters
        ----------
        sql
            DuckDB SQL statement to execute.
        params
            Optional positional parameters for ``sql``.

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

        Parameters
        ----------
        name
            Table name or schema-qualified identifier.

        Returns
        -------
        DuckDBRelation
            DuckDB relation for the requested table/view.
        """
        return self.con.table(name)

    def relation_from_table_key(self, table_key: str) -> DuckDBRelation:
        """Return a relation for a fully qualified table key.

        Returns
        -------
        DuckDBRelation
            Relation bound to the requested table/view.
        """
        return _relation_for_table_key(
            self.con,
            table_key=table_key,
            datasets=self.datasets,
            config=self.config,
        )

    def export_database(self, *, directory: Path) -> None:
        """Export the database to a directory via DuckDB EXPORT DATABASE."""
        DuckDBSession.export_database(self.con, directory=directory)

    def import_database(self, *, directory: Path) -> None:
        """Import the database from a directory via DuckDB IMPORT DATABASE.

        Raises
        ------
        RuntimeError
            If the gateway is read-only.
        """
        if self.config.read_only:
            msg = "Cannot import into a read-only storage gateway"
            raise RuntimeError(msg)
        DuckDBSession.import_database(self.con, directory=directory)
