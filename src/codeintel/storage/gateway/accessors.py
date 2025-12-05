"""Table accessor classes for DuckDB schema access."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.storage.gateway.base_accessor import BaseTableAccessor
from codeintel.storage.gateway.protocol import DuckDBConnection, DuckDBRelation
from codeintel.storage.tracking.build_tracking import BuildTracking
from codeintel.storage.tracking.run_tracking import PipelineRunTracking

if TYPE_CHECKING:
    from codeintel.storage.datasets import DatasetRegistry
    from codeintel.storage.gateway.config import StorageConfig

__all__ = [
    "AnalyticsTables",
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
        rows: Iterable[tuple[str, str, str, str, str]],
    ) -> None:
        """Insert rows into core.repo_map.

        Parameters
        ----------
        rows
            Iterable of (repo, commit, modules_json, overlays_json, generated_at_iso).
        """
        self._insert_rows("core.repo_map", rows)

    def insert_modules(
        self,
        rows: Iterable[tuple[str, str, str, str]],
    ) -> None:
        """Insert rows into core.modules.

        Normalizes 4-column rows by adding default values for language,
        imports_json, and exports_json columns.

        Parameters
        ----------
        rows
            Iterable of (module, path, repo, commit).
        """
        normalized = [
            (module, path, repo, commit, "python", "[]", "[]")
            for module, path, repo, commit in rows
        ]
        self._insert_rows("core.modules", normalized)

    def insert_goids(
        self,
        rows: Iterable[tuple[int, str, str, str, str, str, str, str, int, int, str]],
    ) -> None:
        """Insert rows into core.goids.

        Parameters
        ----------
        rows
            Iterable of (goid_h128, urn, repo, commit, rel_path, language, kind,
            qualname, start_line, end_line, created_at_iso).
        """
        self._insert_rows("core.goids", rows)


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
        rows: Iterable[tuple[str, str, int, int | None, str, int, int, str, str, str, float, str]],
    ) -> None:
        """Insert rows into graph.call_graph_edges.

        Parameters
        ----------
        rows
            Iterable of (repo, commit, caller_goid_h128, callee_goid_h128,
            callsite_path, callsite_line, callsite_col, language, kind,
            resolved_via, confidence, evidence_json).
        """
        self._insert_rows("graph.call_graph_edges", rows)

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
        rows: Iterable[tuple[int, str, str, int, bool, str]],
    ) -> None:
        """Insert rows into graph.call_graph_nodes.

        Parameters
        ----------
        rows
            Iterable of (goid_h128, language, kind, arity, is_public, rel_path).
        """
        self._insert_rows("graph.call_graph_nodes", rows)

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
        rows: Iterable[tuple[str, str, str, str, int, int, int]],
    ) -> None:
        """Insert rows into graph.import_graph_edges.

        Parameters
        ----------
        rows
            Iterable of (repo, commit, src_module, dst_module, src_fan_out,
            dst_fan_in, cycle_group).
        """
        self._insert_rows("graph.import_graph_edges", rows)

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
        rows: Iterable[Sequence[object]],
    ) -> None:
        """Insert rows into graph.symbol_use_edges.

        Normalizes 5-column rows by adding NULL values for def_goid_h128
        and use_goid_h128 columns.

        Parameters
        ----------
        rows
            Iterable of (symbol, def_path, use_path, same_file, same_module,
            def_goid_h128, use_goid_h128).

        Raises
        ------
        ValueError
            If a row is not length 5 or 7.
        """
        expected_basic_len = 5
        expected_full_len = 7
        normalized_rows = []
        for row in rows:
            if len(row) == expected_basic_len:
                symbol, def_path, use_path, same_file, same_module = row
                normalized_rows.append(
                    (symbol, def_path, use_path, same_file, same_module, None, None)
                )
            elif len(row) == expected_full_len:
                normalized_rows.append(tuple(row))
            else:
                message = f"symbol_use_edges rows must have 5 or 7 fields, got {len(row)}: {row}"
                raise ValueError(message)
        self._insert_rows("graph.symbol_use_edges", normalized_rows)

    def insert_cfg_blocks(
        self,
        rows: Iterable[tuple[int, int, str, str, str, int, int, str, str, int, int]],
    ) -> None:
        """Insert rows into graph.cfg_blocks.

        Parameters
        ----------
        rows
            Iterable of values matching cfg_blocks columns.
        """
        self._insert_rows("graph.cfg_blocks", rows)

    def insert_cfg_edges(
        self,
        rows: Iterable[tuple[int, str, str, str | None]],
    ) -> None:
        """Insert rows into graph.cfg_edges.

        Parameters
        ----------
        rows
            Iterable of (function_goid_h128, src_block_id, dst_block_id, edge_kind).
        """
        self._insert_rows("graph.cfg_edges", rows)

    def insert_dfg_edges(
        self,
        rows: Iterable[
            tuple[int, str, str, str | None, str | None, str | None, bool | None, str | None]
        ],
    ) -> None:
        """Insert rows into graph.dfg_edges.

        Parameters
        ----------
        rows
            Iterable of (function_goid_h128, src_block_id, dst_block_id, src_var,
            dst_var, edge_kind, via_phi, use_kind).
        """
        self._insert_rows("graph.dfg_edges", rows)


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
        rows: Iterable[
            tuple[int, str, str, str, str, str, str, str, int, int, int, int, float, bool, str, str]
        ],
    ) -> None:
        """Insert rows into analytics.coverage_functions.

        Parameters
        ----------
        rows
            Iterable of values matching coverage_functions columns.
        """
        self._insert_rows("analytics.coverage_functions", rows)

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
        rows: Iterable[tuple[str, str, str, int, bool, bool, int, int, str]],
    ) -> None:
        """Insert rows into analytics.coverage_lines.

        Parameters
        ----------
        rows
            Iterable of values matching coverage_lines columns.
        """
        self._insert_rows("analytics.coverage_lines", rows)

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
        rows: Iterable[
            tuple[
                str,
                int | None,
                str | None,
                str,
                str,
                str,
                str,
                str,
                str,
                int | None,
                str,
                bool,
                bool,
                str,
            ]
        ],
    ) -> None:
        """Insert rows into analytics.test_catalog.

        Parameters
        ----------
        rows
            Iterable of values matching test_catalog columns.
        """
        self._insert_rows("analytics.test_catalog", rows)

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
        rows: Iterable[
            tuple[str, int | None, int, str, str, str, str, str, int, int, float, str, str]
        ],
    ) -> None:
        """Insert rows into analytics.test_coverage_edges.

        Parameters
        ----------
        rows
            Iterable of values matching test_coverage_edges columns.
        """
        self._insert_rows("analytics.test_coverage_edges", rows)

    def insert_function_metrics(
        self,
        rows: Iterable[
            tuple[
                int,
                str,
                str,
                str,
                str,
                str,
                str,
                str,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                bool,
                bool,
                bool,
                bool,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                bool,
                str,
                str,
            ]
        ],
    ) -> None:
        """Insert rows into analytics.function_metrics.

        Parameters
        ----------
        rows
            Iterable of values matching function_metrics columns.
        """
        self._insert_rows("analytics.function_metrics", rows)

    def insert_goid_risk_factors(
        self,
        rows: Iterable[
            tuple[
                int,
                str,
                str,
                str,
                str,
                str,
                str,
                str,
                int,
                int,
                int,
                str,
                str,
                str,
                float,
                float,
                int,
                bool,
                int,
                int,
                float,
                bool,
                int,
                int,
                str,
                float,
                str,
                str,
                str,
                str,
            ]
        ],
    ) -> None:
        """Insert rows into analytics.goid_risk_factors.

        Parameters
        ----------
        rows
            Iterable of values matching goid_risk_factors columns.
        """
        self._insert_rows("analytics.goid_risk_factors", rows)

    def insert_config_values(
        self,
        rows: Iterable[tuple[str, str, str, str, str | None, str | None, str | None, int]],
    ) -> None:
        """Insert rows into analytics.config_values.

        Parameters
        ----------
        rows
            Iterable of (repo, commit, config_path, format, key, reference_paths,
            reference_modules, reference_modules_json, reference_count).
        """
        self._insert_rows("analytics.config_values", rows)

    def insert_typedness(
        self,
        rows: Iterable[tuple[str, str, str, int, str, int, bool]],
    ) -> None:
        """Insert rows into analytics.typedness.

        Parameters
        ----------
        rows
            Iterable of values matching typedness columns.
        """
        self._insert_rows("analytics.typedness", rows)

    def insert_static_diagnostics(
        self,
        rows: Iterable[tuple[str, str, str, int, int, int, int, bool]],
    ) -> None:
        """Insert rows into analytics.static_diagnostics.

        Parameters
        ----------
        rows
            Iterable of values matching static_diagnostics columns.
        """
        self._insert_rows("analytics.static_diagnostics", rows)

    def insert_subsystems(
        self,
        rows: Iterable[
            tuple[
                str,
                str,
                str,
                str,
                str | None,
                int,
                str,
                str | None,
                int,
                int,
                int,
                int,
                int,
                float | None,
                float | None,
                int,
                str | None,
                str,
            ]
        ],
    ) -> None:
        """Insert rows into analytics.subsystems.

        Parameters
        ----------
        rows
            Iterable of values matching subsystems columns.
        """
        self._insert_rows("analytics.subsystems", rows)

    def insert_subsystem_modules(
        self,
        rows: Iterable[tuple[str, str, str, str, str | None]],
    ) -> None:
        """Insert rows into analytics.subsystem_modules.

        Parameters
        ----------
        rows
            Iterable of values matching subsystem_modules columns.
        """
        self._insert_rows("analytics.subsystem_modules", rows)


@dataclass
class DuckDBGateway:
    """Concrete StorageGateway implementation."""

    config: StorageConfig
    datasets: DatasetRegistry
    con: DuckDBConnection
    analytics: AnalyticsTables = field(init=False)
    build: BuildTracking = field(init=False)
    core: CoreTables = field(init=False)
    docs: DocsViews = field(init=False)
    graph: GraphTables = field(init=False)
    runs: PipelineRunTracking = field(init=False)

    def __post_init__(self) -> None:
        """Initialize table accessor instances after dataclass init."""
        self.analytics = AnalyticsTables(self.con)
        self.build = BuildTracking(self.con)
        self.core = CoreTables(self.con)
        self.docs = DocsViews(self.con)
        self.graph = GraphTables(self.con)
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
