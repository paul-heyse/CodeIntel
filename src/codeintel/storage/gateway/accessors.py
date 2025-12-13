"""Table accessor classes for DuckDB schema access."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.config.datasets import get_table_columns
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.gateway.base_accessor import BaseTableAccessor
from codeintel.storage.gateway.insert_helpers import insert_rows
from codeintel.storage.ibis_adapter import IbisGateway
from codeintel.storage.tracking.asset_tracking import AssetTracking
from codeintel.storage.tracking.build_tracking import BuildTracking
from codeintel.storage.tracking.run_tracking import PipelineRunTracking

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from codeintel.config.datasets.generated_rows.analytics import (
        AnalyticsConfigValuesRow,
        AnalyticsCoverageFunctionsRow,
        AnalyticsCoverageLinesRow,
        AnalyticsFunctionMetricsRow,
        AnalyticsGoidRiskFactorsRow,
        AnalyticsStaticDiagnosticsRow,
        AnalyticsSubsystemModulesRow,
        AnalyticsSubsystemsRow,
        AnalyticsTestCatalogRow,
        AnalyticsTestCoverageEdgesRow,
        AnalyticsTypednessRow,
    )
    from codeintel.config.datasets.generated_rows.core import (
        CoreFileStateRow,
        CoreGoidsRow,
        CoreModulesRow,
        CoreRepoMapRow,
        CoreScipOccurrencesRow,
    )
    from codeintel.config.datasets.generated_rows.graph import (
        GraphCallGraphEdgesRow,
        GraphCallGraphNodesRow,
        GraphCfgBlocksRow,
        GraphCfgEdgesRow,
        GraphDfgEdgesRow,
        GraphImportGraphEdgesRow,
        GraphSymbolUseEdgesRow,
    )
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
    """Accessors for core schema tables."""

    def goids(self) -> DuckDBRelation:
        """Return relation for core.goids.

        Returns
        -------
        DuckDBRelation
            Relation selecting core.goids.
        """
        return self._table("core.goids")

    def file_state(self) -> DuckDBRelation:
        """Return relation for core.file_state.

        Returns
        -------
        DuckDBRelation
            Relation selecting core.file_state.
        """
        return self._table("core.file_state")

    def scip_occurrences(self) -> DuckDBRelation:
        """Return relation for core.scip_occurrences.

        Returns
        -------
        DuckDBRelation
            Relation selecting core.scip_occurrences.
        """
        return self._table("core.scip_occurrences")

    def modules(self) -> DuckDBRelation:
        """Return relation for core.modules.

        Returns
        -------
        DuckDBRelation
            Relation selecting core.modules.
        """
        return self._table("core.modules")

    def repo_map(self) -> DuckDBRelation:
        """Return relation for core.repo_map.

        Returns
        -------
        DuckDBRelation
            Relation selecting core.repo_map.
        """
        return self._table("core.repo_map")

    def insert_repo_map(
        self,
        rows: Iterable[CoreRepoMapRow | Sequence[str | None]],
    ) -> None:
        """Insert rows into core.repo_map.

        Parameters
        ----------
        rows
            Iterable of mapping rows or tuples of
            (repo, commit, modules_json, overlays_json, generated_at_iso).
        """
        self._insert_normalized("core.repo_map", rows)

    def insert_modules(
        self,
        rows: Iterable[CoreModulesRow | Sequence[object]],
    ) -> None:
        """Insert rows into core.modules.

        Normalizes 4-field inputs by applying defaults for language, tags, and owners.

        Raises
        ------
        ValueError
            If a row has an unexpected number of fields.
        """
        columns = get_table_columns("core.modules")
        minimal_len = 4
        full_len = len(columns)
        normalized_rows: list[Mapping[str, object]] = []
        for row_candidate in rows:
            if isinstance(row_candidate, Mapping):
                normalized = dict(row_candidate)
                normalized.setdefault("language", "python")
                normalized.setdefault("tags", "[]")
                normalized.setdefault("owners", "[]")
                normalized_rows.append(normalized)
                continue
            sequence = tuple(row_candidate)
            if len(sequence) == minimal_len:
                module, path, repo, commit = sequence
                normalized_rows.append(
                    {
                        "module": module,
                        "path": path,
                        "repo": repo,
                        "commit": commit,
                        "language": "python",
                        "tags": "[]",
                        "owners": "[]",
                    }
                )
                continue
            if len(sequence) == full_len:
                normalized_rows.append(dict(zip(columns, sequence, strict=True)))
                continue
            message = (
                f"modules rows must have {minimal_len} fields "
                f"or {full_len} fields, got {len(sequence)}: {sequence}"
            )
            raise ValueError(message)
        insert_rows(self.gateway, "core.modules", normalized_rows)

    def insert_file_state(
        self,
        rows: Iterable[CoreFileStateRow | Sequence[object]],
    ) -> None:
        """Insert rows into core.file_state.

        Parameters
        ----------
        rows
            Iterable of mapping rows or tuples matching file_state columns.
        """
        self._insert_normalized("core.file_state", rows)

    def insert_goids(
        self,
        rows: Iterable[CoreGoidsRow | Sequence[object]],
    ) -> None:
        """Insert rows into core.goids.

        Parameters
        ----------
        rows
            Iterable of mapping rows or tuples matching goids columns.
        """
        self._insert_normalized("core.goids", rows)

    def insert_scip_occurrences(
        self,
        rows: Iterable[CoreScipOccurrencesRow | Sequence[object]],
    ) -> None:
        """Insert rows into core.scip_occurrences.

        Parameters
        ----------
        rows
            Iterable of mapping rows or tuples matching scip_occurrences columns.
        """
        self._insert_normalized("core.scip_occurrences", rows)


@dataclass(frozen=True)
class GraphTables(BaseTableAccessor):
    """Accessors for graph schema tables."""

    def call_graph_edges(self) -> DuckDBRelation:
        """Return relation for graph.call_graph_edges.

        Returns
        -------
        DuckDBRelation
            Relation selecting graph.call_graph_edges.
        """
        return self._table("graph.call_graph_edges")

    def insert_call_graph_edges(
        self,
        rows: Iterable[GraphCallGraphEdgesRow | Sequence[object]],
    ) -> None:
        """Insert rows into graph.call_graph_edges."""
        self._insert_normalized("graph.call_graph_edges", rows)

    def call_graph_nodes(self) -> DuckDBRelation:
        """Return relation for graph.call_graph_nodes.

        Returns
        -------
        DuckDBRelation
            Relation selecting graph.call_graph_nodes.
        """
        return self._table("graph.call_graph_nodes")

    def insert_call_graph_nodes(
        self,
        rows: Iterable[GraphCallGraphNodesRow | Sequence[object]],
    ) -> None:
        """Insert rows into graph.call_graph_nodes."""
        self._insert_normalized("graph.call_graph_nodes", rows)

    def import_graph_edges(self) -> DuckDBRelation:
        """Return relation for graph.import_graph_edges.

        Returns
        -------
        DuckDBRelation
            Relation selecting graph.import_graph_edges.
        """
        return self._table("graph.import_graph_edges")

    def insert_import_graph_edges(
        self,
        rows: Iterable[GraphImportGraphEdgesRow | Sequence[object]],
    ) -> None:
        """Insert rows into graph.import_graph_edges."""
        columns = get_table_columns("graph.import_graph_edges")
        normalized_rows: list[Mapping[str, object]] = []
        for row_candidate in rows:
            if isinstance(row_candidate, Mapping):
                normalized_rows.append(row_candidate)
                continue
            sequence = row_candidate
            if len(sequence) == len(columns) - 1:
                sequence = (*sequence, None)
            normalized_rows.append(dict(zip(columns, sequence, strict=True)))
        insert_rows(self.gateway, "graph.import_graph_edges", normalized_rows)

    def symbol_use_edges(self) -> DuckDBRelation:
        """Return relation for graph.symbol_use_edges.

        Returns
        -------
        DuckDBRelation
            Relation selecting graph.symbol_use_edges.
        """
        return self._table("graph.symbol_use_edges")

    def insert_symbol_use_edges(
        self,
        rows: Iterable[GraphSymbolUseEdgesRow | Sequence[object]],
    ) -> None:
        """Insert rows into graph.symbol_use_edges.

        Raises
        ------
        ValueError
            If a row has an unexpected number of fields.
        """
        columns = get_table_columns("graph.symbol_use_edges")
        basic_len = 5
        full_len = len(columns)
        normalized_rows: list[Mapping[str, object]] = []
        for row_candidate in rows:
            if isinstance(row_candidate, Mapping):
                normalized = dict(row_candidate)
                normalized.setdefault("def_goid_h128", None)
                normalized.setdefault("use_goid_h128", None)
                normalized_rows.append(normalized)
                continue
            sequence = tuple(row_candidate)
            if len(sequence) == basic_len:
                symbol, def_path, use_path, same_file, same_module = sequence
                normalized_rows.append(
                    {
                        "symbol": symbol,
                        "def_path": def_path,
                        "use_path": use_path,
                        "same_file": same_file,
                        "same_module": same_module,
                        "def_goid_h128": None,
                        "use_goid_h128": None,
                    }
                )
                continue
            if len(sequence) == full_len:
                normalized_rows.append(dict(zip(columns, sequence, strict=True)))
                continue
            message = (
                "symbol_use_edges rows must have 5 or "
                f"{full_len} fields, "
                f"got {len(sequence)}: {sequence}"
            )
            raise ValueError(message)
        insert_rows(self.gateway, "graph.symbol_use_edges", normalized_rows)

    def insert_cfg_blocks(
        self,
        rows: Iterable[GraphCfgBlocksRow | Sequence[object]],
    ) -> None:
        """Insert rows into graph.cfg_blocks."""
        self._insert_normalized("graph.cfg_blocks", rows)

    def insert_cfg_edges(
        self,
        rows: Iterable[GraphCfgEdgesRow | Sequence[object]],
    ) -> None:
        """Insert rows into graph.cfg_edges."""
        self._insert_normalized("graph.cfg_edges", rows)

    def insert_dfg_edges(
        self,
        rows: Iterable[GraphDfgEdgesRow | Sequence[object]],
    ) -> None:
        """Insert rows into graph.dfg_edges."""
        self._insert_normalized("graph.dfg_edges", rows)


@dataclass(frozen=True)
class DocsViews(BaseTableAccessor):
    """Accessors for docs schema views."""

    def function_summary(self) -> DuckDBRelation:
        """Return relation for docs.v_function_summary.

        Returns
        -------
        DuckDBRelation
            Relation selecting docs.v_function_summary.
        """
        return self._table("docs.v_function_summary")

    def call_graph_enriched(self) -> DuckDBRelation:
        """Return relation for docs.v_call_graph_enriched.

        Returns
        -------
        DuckDBRelation
            Relation selecting docs.v_call_graph_enriched.
        """
        return self._table("docs.v_call_graph_enriched")

    def function_profile(self) -> DuckDBRelation:
        """Return relation for analytics.function_profile.

        Returns
        -------
        DuckDBRelation
            Relation selecting analytics.function_profile.
        """
        return self._table("analytics.function_profile")


@dataclass(frozen=True)
class AnalyticsTables(BaseTableAccessor):
    """Accessors for analytics schema tables."""

    def function_metrics(self) -> DuckDBRelation:
        """Return relation for analytics.function_metrics.

        Returns
        -------
        DuckDBRelation
            Relation selecting analytics.function_metrics.
        """
        return self._table("analytics.function_metrics")

    def function_types(self) -> DuckDBRelation:
        """Return relation for analytics.function_types.

        Returns
        -------
        DuckDBRelation
            Relation selecting analytics.function_types.
        """
        return self._table("analytics.function_types")

    def coverage_functions(self) -> DuckDBRelation:
        """Return relation for analytics.coverage_functions.

        Returns
        -------
        DuckDBRelation
            Relation selecting analytics.coverage_functions.
        """
        return self._table("analytics.coverage_functions")

    def insert_coverage_functions(
        self,
        rows: Iterable[AnalyticsCoverageFunctionsRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.coverage_functions."""
        self._insert_normalized("analytics.coverage_functions", rows)

    def coverage_lines(self) -> DuckDBRelation:
        """Return relation for analytics.coverage_lines.

        Returns
        -------
        DuckDBRelation
            Relation selecting analytics.coverage_lines.
        """
        return self._table("analytics.coverage_lines")

    def insert_coverage_lines(
        self,
        rows: Iterable[AnalyticsCoverageLinesRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.coverage_lines.

        Parameters
        ----------
        rows
            Iterable of mapping rows or tuples matching coverage_lines columns.
        """
        self._insert_normalized("analytics.coverage_lines", rows)

    def test_catalog(self) -> DuckDBRelation:
        """Return relation for analytics.test_catalog.

        Returns
        -------
        DuckDBRelation
            Relation selecting analytics.test_catalog.
        """
        return self._table("analytics.test_catalog")

    def insert_test_catalog(
        self,
        rows: Iterable[AnalyticsTestCatalogRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.test_catalog."""
        self._insert_normalized("analytics.test_catalog", rows)

    def test_coverage_edges(self) -> DuckDBRelation:
        """Return relation for analytics.test_coverage_edges.

        Returns
        -------
        DuckDBRelation
            Relation selecting analytics.test_coverage_edges.
        """
        return self._table("analytics.test_coverage_edges")

    def insert_test_coverage_edges(
        self,
        rows: Iterable[AnalyticsTestCoverageEdgesRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.test_coverage_edges."""
        self._insert_normalized("analytics.test_coverage_edges", rows)

    def insert_function_metrics(
        self,
        rows: Iterable[AnalyticsFunctionMetricsRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.function_metrics."""
        self._insert_normalized("analytics.function_metrics", rows)

    def insert_goid_risk_factors(
        self,
        rows: Iterable[AnalyticsGoidRiskFactorsRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.goid_risk_factors."""
        self._insert_normalized("analytics.goid_risk_factors", rows)

    def insert_config_values(
        self,
        rows: Iterable[AnalyticsConfigValuesRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.config_values."""
        self._insert_normalized("analytics.config_values", rows)

    def insert_typedness(
        self,
        rows: Iterable[AnalyticsTypednessRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.typedness."""
        self._insert_normalized("analytics.typedness", rows)

    def insert_static_diagnostics(
        self,
        rows: Iterable[AnalyticsStaticDiagnosticsRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.static_diagnostics."""
        self._insert_normalized("analytics.static_diagnostics", rows)

    def insert_subsystems(
        self,
        rows: Iterable[AnalyticsSubsystemsRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.subsystems."""
        self._insert_normalized("analytics.subsystems", rows)

    def insert_subsystem_modules(
        self,
        rows: Iterable[AnalyticsSubsystemModulesRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.subsystem_modules."""
        self._insert_normalized("analytics.subsystem_modules", rows)


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
        self.policy = DuckDBPolicyBackend(self)
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
        """
        Execute a SQL statement using the active DuckDB connection.

        Returns
        -------
        DuckDBConnection
            Connection representing the executed query.
        """
        return self.con.execute(sql, params)

    def table(self, name: str) -> DuckDBRelation:
        """
        Return a relation object for the specified table or view.

        Returns
        -------
        DuckDBRelation
            Relation bound to the requested table/view.
        """
        return self.con.table(name)
