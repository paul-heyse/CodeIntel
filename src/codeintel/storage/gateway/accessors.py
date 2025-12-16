"""Table accessor classes for DuckDB schema access.

The gateway accessors provide a small, typed read surface over DuckDB relations.
All mutation/write operations are routed through `codeintel.storage.warehouse.Warehouse`
to preserve a single I/O boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.gateway.base_accessor import BaseTableAccessor
from codeintel.storage.ibis_adapter import IbisGateway
from codeintel.storage.tracking.asset_tracking import AssetTracking
from codeintel.storage.tracking.build_tracking import BuildTracking
from codeintel.storage.tracking.run_tracking import PipelineRunTracking

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    def function_summary(self) -> DuckDBRelation:
        """Return the ``docs.v_function_summary`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``docs.v_function_summary``.
        """
        return self._table("docs.v_function_summary")

    def call_graph_enriched(self) -> DuckDBRelation:
        """Return the ``docs.v_call_graph_enriched`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``docs.v_call_graph_enriched``.
        """
        return self._table("docs.v_call_graph_enriched")

    def function_profile(self) -> DuckDBRelation:
        """Return the ``analytics.function_profile`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.function_profile``.
        """
        return self._table("analytics.function_profile")


@dataclass(frozen=True)
class AnalyticsTables(BaseTableAccessor):
    """Read accessors for analytics schema tables."""

    def function_metrics(self) -> DuckDBRelation:
        """Return the ``analytics.function_metrics`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.function_metrics``.
        """
        return self._table("analytics.function_metrics")

    def function_types(self) -> DuckDBRelation:
        """Return the ``analytics.function_types`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.function_types``.
        """
        return self._table("analytics.function_types")

    def coverage_functions(self) -> DuckDBRelation:
        """Return the ``analytics.coverage_functions`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.coverage_functions``.
        """
        return self._table("analytics.coverage_functions")

    def coverage_lines(self) -> DuckDBRelation:
        """Return the ``analytics.coverage_lines`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.coverage_lines``.
        """
        return self._table("analytics.coverage_lines")

    def test_catalog(self) -> DuckDBRelation:
        """Return the ``analytics.test_catalog`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.test_catalog``.
        """
        return self._table("analytics.test_catalog")

    def test_coverage_edges(self) -> DuckDBRelation:
        """Return the ``analytics.test_coverage_edges`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.test_coverage_edges``.
        """
        return self._table("analytics.test_coverage_edges")

    def function_profile(self) -> DuckDBRelation:
        """Return the ``analytics.function_profile`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.function_profile``.
        """
        return self._table("analytics.function_profile")

    def goid_risk_factors(self) -> DuckDBRelation:
        """Return the ``analytics.goid_risk_factors`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.goid_risk_factors``.
        """
        return self._table("analytics.goid_risk_factors")

    def config_values(self) -> DuckDBRelation:
        """Return the ``analytics.config_values`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.config_values``.
        """
        return self._table("analytics.config_values")

    def typedness(self) -> DuckDBRelation:
        """Return the ``analytics.typedness`` relation.

        Returns
        -------
        DuckDBRelation
            Relation for ``analytics.typedness``.
        """
        return self._table("analytics.typedness")

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
    ibis: IbisGateway = field(init=False)
    policy: DuckDBPolicyBackend = field(init=False)
    analytics: AnalyticsTables = field(init=False)
    assets: AssetTracking = field(init=False)
    build: BuildTracking = field(init=False)
    core: CoreTables = field(init=False)
    docs: DocsViews = field(init=False)
    graph: GraphTables = field(init=False)
    runs: PipelineRunTracking = field(init=False)

    def __post_init__(self) -> None:
        """Initialize table accessor instances after dataclass init."""
        self.ibis = IbisGateway(self)
        schemas = {
            table_key: contract.schema
            for table_key, contract in self.datasets.by_table_key.items()
            if contract.schema is not None and not contract.is_view
        }
        self.policy = DuckDBPolicyBackend(self, schema_provider=MappingSchemaProvider(schemas))
        self.analytics = AnalyticsTables(self)
        self.assets = AssetTracking(self)
        self.build = BuildTracking(self)
        self.core = CoreTables(self)
        self.docs = DocsViews(self)
        self.graph = GraphTables(self)
        self.runs = PipelineRunTracking(self.con)

    def close(self) -> None:
        """Close the underlying connection."""
        self.con.close()

    def execute(self, sql: str, params: Sequence[object] | None = None) -> DuckDBConnection:
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
