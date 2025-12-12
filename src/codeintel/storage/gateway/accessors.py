"""Table accessor classes for DuckDB schema access."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.storage.gateway.base_accessor import BaseTableAccessor
from codeintel.storage.gateway.insert_helpers import insert_rows
from codeintel.storage.ibis_adapter import IbisGateway
from codeintel.storage.tracking.build_tracking import BuildTracking
from codeintel.storage.tracking.run_tracking import PipelineRunTracking

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from codeintel.storage.datasets import DatasetRegistry
    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.gateway.protocol import DuckDBConnection, DuckDBRelation
    from codeintel.storage.gateway.rows.analytics import (
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
    from codeintel.storage.gateway.rows.core import (
        CoreFileStateRow,
        CoreGoidsRow,
        CoreModulesRow,
        CoreRepoMapRow,
        CoreScipOccurrencesRow,
    )
    from codeintel.storage.gateway.rows.graph import (
        GraphCallGraphEdgesRow,
        GraphCallGraphNodesRow,
        GraphCfgBlocksRow,
        GraphCfgEdgesRow,
        GraphDfgEdgesRow,
        GraphImportGraphEdgesRow,
        GraphSymbolUseEdgesRow,
    )

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

    _repo_map_columns: Sequence[str] = ("repo", "commit", "modules", "overlays", "generated_at")
    _modules_columns: Sequence[str] = (
        "module",
        "path",
        "repo",
        "commit",
        "language",
        "tags",
        "owners",
    )
    _file_state_columns: Sequence[str] = (
        "repo",
        "commit",
        "rel_path",
        "language",
        "size_bytes",
        "mtime_ns",
        "content_hash",
    )
    _goids_columns: Sequence[str] = (
        "goid_h128",
        "urn",
        "repo",
        "commit",
        "rel_path",
        "language",
        "kind",
        "qualname",
        "start_line",
        "end_line",
        "created_at",
    )
    _scip_occurrences_columns: Sequence[str] = (
        "repo",
        "commit",
        "rel_path",
        "symbol",
        "start_line",
        "start_col",
        "end_line",
        "end_col",
        "roles",
        "created_at",
    )

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
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(row, self._repo_map_columns, "core.repo_map")
            for row in rows
        )
        insert_rows(self.con, "core.repo_map", normalized_rows)

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
        minimal_len = 4
        full_len = len(self._modules_columns)
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
                normalized_rows.append(
                    _normalize_to_mapping(sequence, self._modules_columns, "core.modules")
                )
                continue
            message = (
                f"modules rows must have {minimal_len} fields "
                f"or {full_len} fields, got {len(sequence)}: {sequence}"
            )
            raise ValueError(message)
        insert_rows(self.con, "core.modules", normalized_rows)

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
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(row, self._file_state_columns, "core.file_state")
            for row in rows
        )
        insert_rows(self.con, "core.file_state", normalized_rows)

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
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(row, self._goids_columns, "core.goids")
            for row in rows
        )
        insert_rows(self.con, "core.goids", normalized_rows)

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
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(
                row,
                self._scip_occurrences_columns,
                "core.scip_occurrences",
            )
            for row in rows
        )
        insert_rows(self.con, "core.scip_occurrences", normalized_rows)


@dataclass(frozen=True)
class GraphTables(BaseTableAccessor):
    """Accessors for graph schema tables."""

    _call_graph_nodes_columns: Sequence[str] = (
        "goid_h128",
        "language",
        "kind",
        "arity",
        "is_public",
        "rel_path",
    )
    _call_graph_edges_columns: Sequence[str] = (
        "repo",
        "commit",
        "caller_goid_h128",
        "callee_goid_h128",
        "callsite_path",
        "callsite_line",
        "callsite_col",
        "language",
        "kind",
        "resolved_via",
        "confidence",
        "evidence_json",
    )
    _import_graph_edges_columns: Sequence[str] = (
        "repo",
        "commit",
        "src_module",
        "dst_module",
        "src_fan_out",
        "dst_fan_in",
        "cycle_group",
        "module_layer",
    )
    _symbol_use_edges_columns: Sequence[str] = (
        "symbol",
        "def_path",
        "use_path",
        "same_file",
        "same_module",
        "def_goid_h128",
        "use_goid_h128",
    )
    _cfg_blocks_columns: Sequence[str] = (
        "function_goid_h128",
        "block_idx",
        "block_id",
        "label",
        "file_path",
        "start_line",
        "end_line",
        "kind",
        "stmts_json",
        "in_degree",
        "out_degree",
    )
    _cfg_edges_columns: Sequence[str] = (
        "function_goid_h128",
        "src_block_id",
        "dst_block_id",
        "edge_kind",
    )
    _dfg_edges_columns: Sequence[str] = (
        "function_goid_h128",
        "src_block_id",
        "dst_block_id",
        "src_var",
        "dst_var",
        "edge_kind",
        "via_phi",
        "use_kind",
    )

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
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(
                row, self._call_graph_edges_columns, "graph.call_graph_edges"
            )
            for row in rows
        )
        insert_rows(self.con, "graph.call_graph_edges", normalized_rows)

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
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(
                row, self._call_graph_nodes_columns, "graph.call_graph_nodes"
            )
            for row in rows
        )
        insert_rows(self.con, "graph.call_graph_nodes", normalized_rows)

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
        normalized_rows: list[Mapping[str, object]] = []
        for row_candidate in rows:
            if isinstance(row_candidate, Mapping):
                normalized_rows.append(row_candidate)
                continue
            sequence = row_candidate
            if len(sequence) == len(self._import_graph_edges_columns) - 1:
                sequence = (*sequence, None)
            normalized_rows.append(
                _normalize_to_mapping(
                    sequence, self._import_graph_edges_columns, "graph.import_graph_edges"
                )
            )
        insert_rows(self.con, "graph.import_graph_edges", normalized_rows)

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
        basic_len = 5
        full_len = len(self._symbol_use_edges_columns)
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
                normalized_rows.append(
                    _normalize_to_mapping(
                        sequence, self._symbol_use_edges_columns, "graph.symbol_use_edges"
                    )
                )
                continue
            message = (
                "symbol_use_edges rows must have 5 or "
                f"{full_len} fields, "
                f"got {len(sequence)}: {sequence}"
            )
            raise ValueError(message)
        insert_rows(self.con, "graph.symbol_use_edges", normalized_rows)

    def insert_cfg_blocks(
        self,
        rows: Iterable[GraphCfgBlocksRow | Sequence[object]],
    ) -> None:
        """Insert rows into graph.cfg_blocks."""
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(row, self._cfg_blocks_columns, "graph.cfg_blocks")
            for row in rows
        )
        insert_rows(self.con, "graph.cfg_blocks", normalized_rows)

    def insert_cfg_edges(
        self,
        rows: Iterable[GraphCfgEdgesRow | Sequence[object]],
    ) -> None:
        """Insert rows into graph.cfg_edges."""
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(row, self._cfg_edges_columns, "graph.cfg_edges")
            for row in rows
        )
        insert_rows(self.con, "graph.cfg_edges", normalized_rows)

    def insert_dfg_edges(
        self,
        rows: Iterable[GraphDfgEdgesRow | Sequence[object]],
    ) -> None:
        """Insert rows into graph.dfg_edges."""
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(row, self._dfg_edges_columns, "graph.dfg_edges")
            for row in rows
        )
        insert_rows(self.con, "graph.dfg_edges", normalized_rows)


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

    _coverage_functions_columns: Sequence[str] = (
        "function_goid_h128",
        "urn",
        "repo",
        "commit",
        "rel_path",
        "language",
        "kind",
        "qualname",
        "start_line",
        "end_line",
        "executable_lines",
        "covered_lines",
        "coverage_ratio",
        "tested",
        "untested_reason",
        "created_at",
    )
    _function_metrics_columns: Sequence[str] = (
        "function_goid_h128",
        "urn",
        "repo",
        "commit",
        "rel_path",
        "language",
        "kind",
        "qualname",
        "start_line",
        "end_line",
        "loc",
        "logical_loc",
        "param_count",
        "positional_params",
        "keyword_only_params",
        "has_varargs",
        "has_varkw",
        "is_async",
        "is_generator",
        "return_count",
        "yield_count",
        "raise_count",
        "cyclomatic_complexity",
        "max_nesting_depth",
        "stmt_count",
        "decorator_count",
        "has_docstring",
        "complexity_bucket",
        "created_at",
    )
    _goid_risk_factors_columns: Sequence[str] = (
        "function_goid_h128",
        "urn",
        "repo",
        "commit",
        "rel_path",
        "language",
        "kind",
        "qualname",
        "loc",
        "logical_loc",
        "cyclomatic_complexity",
        "complexity_bucket",
        "typedness_bucket",
        "typedness_source",
        "hotspot_score",
        "file_typed_ratio",
        "static_error_count",
        "has_static_errors",
        "executable_lines",
        "covered_lines",
        "coverage_ratio",
        "tested",
        "test_count",
        "failing_test_count",
        "last_test_status",
        "risk_score",
        "risk_level",
        "tags",
        "owners",
        "created_at",
    )
    _config_values_columns: Sequence[str] = (
        "repo",
        "commit",
        "config_path",
        "format",
        "key",
        "reference_paths",
        "reference_modules",
        "reference_count",
    )
    _static_diagnostics_columns: Sequence[str] = (
        "repo",
        "commit",
        "rel_path",
        "pyrefly_errors",
        "pyright_errors",
        "ruff_errors",
        "total_errors",
        "has_errors",
    )
    _subsystems_columns: Sequence[str] = (
        "repo",
        "commit",
        "subsystem_id",
        "name",
        "description",
        "module_count",
        "modules_json",
        "entrypoints_json",
        "internal_edge_count",
        "external_edge_count",
        "fan_in",
        "fan_out",
        "function_count",
        "avg_risk_score",
        "max_risk_score",
        "high_risk_function_count",
        "risk_level",
        "created_at",
    )
    _subsystem_modules_columns: Sequence[str] = (
        "repo",
        "commit",
        "subsystem_id",
        "module",
        "role",
    )
    _test_catalog_columns: Sequence[str] = (
        "test_id",
        "test_goid_h128",
        "urn",
        "repo",
        "commit",
        "rel_path",
        "qualname",
        "kind",
        "status",
        "duration_ms",
        "markers",
        "parametrized",
        "flaky",
        "created_at",
    )
    _test_coverage_edges_columns: Sequence[str] = (
        "test_id",
        "test_goid_h128",
        "function_goid_h128",
        "urn",
        "repo",
        "commit",
        "rel_path",
        "qualname",
        "covered_lines",
        "executable_lines",
        "coverage_ratio",
        "last_status",
        "created_at",
    )

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
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(
                row, self._coverage_functions_columns, "analytics.coverage_functions"
            )
            for row in rows
        )
        insert_rows(self.con, "analytics.coverage_functions", normalized_rows)

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
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(
                row,
                (
                    "repo",
                    "commit",
                    "rel_path",
                    "line",
                    "is_executable",
                    "is_covered",
                    "hits",
                    "context_count",
                    "created_at",
                ),
                "analytics.coverage_lines",
            )
            for row in rows
        )
        insert_rows(self.con, "analytics.coverage_lines", normalized_rows)

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
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(row, self._test_catalog_columns, "analytics.test_catalog")
            for row in rows
        )
        insert_rows(self.con, "analytics.test_catalog", normalized_rows)

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
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(
                row,
                self._test_coverage_edges_columns,
                "analytics.test_coverage_edges",
            )
            for row in rows
        )
        insert_rows(self.con, "analytics.test_coverage_edges", normalized_rows)

    def insert_function_metrics(
        self,
        rows: Iterable[AnalyticsFunctionMetricsRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.function_metrics."""
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(
                row, self._function_metrics_columns, "analytics.function_metrics"
            )
            for row in rows
        )
        insert_rows(self.con, "analytics.function_metrics", normalized_rows)

    def insert_goid_risk_factors(
        self,
        rows: Iterable[AnalyticsGoidRiskFactorsRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.goid_risk_factors."""
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(
                row, self._goid_risk_factors_columns, "analytics.goid_risk_factors"
            )
            for row in rows
        )
        insert_rows(self.con, "analytics.goid_risk_factors", normalized_rows)

    def insert_config_values(
        self,
        rows: Iterable[AnalyticsConfigValuesRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.config_values."""
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(row, self._config_values_columns, "analytics.config_values")
            for row in rows
        )
        insert_rows(self.con, "analytics.config_values", normalized_rows)

    def insert_typedness(
        self,
        rows: Iterable[AnalyticsTypednessRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.typedness."""
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(
                row,
                (
                    "repo",
                    "commit",
                    "path",
                    "type_error_count",
                    "annotation_ratio",
                    "untyped_defs",
                    "overlay_needed",
                ),
                "analytics.typedness",
            )
            for row in rows
        )
        insert_rows(self.con, "analytics.typedness", normalized_rows)

    def insert_static_diagnostics(
        self,
        rows: Iterable[AnalyticsStaticDiagnosticsRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.static_diagnostics."""
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(
                row, self._static_diagnostics_columns, "analytics.static_diagnostics"
            )
            for row in rows
        )
        insert_rows(self.con, "analytics.static_diagnostics", normalized_rows)

    def insert_subsystems(
        self,
        rows: Iterable[AnalyticsSubsystemsRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.subsystems."""
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(row, self._subsystems_columns, "analytics.subsystems")
            for row in rows
        )
        insert_rows(self.con, "analytics.subsystems", normalized_rows)

    def insert_subsystem_modules(
        self,
        rows: Iterable[AnalyticsSubsystemModulesRow | Sequence[object]],
    ) -> None:
        """Insert rows into analytics.subsystem_modules."""
        normalized_rows = (
            row
            if isinstance(row, Mapping)
            else _normalize_to_mapping(
                row, self._subsystem_modules_columns, "analytics.subsystem_modules"
            )
            for row in rows
        )
        insert_rows(self.con, "analytics.subsystem_modules", normalized_rows)


def _normalize_to_mapping(
    row: Sequence[object],
    columns: Sequence[str],
    table_key: str,
) -> dict[str, object]:
    """Convert a positional row sequence into a mapping keyed by columns.

    Returns
    -------
    dict[str, object]
        Mapping of column names to values from the sequence.

    Raises
    ------
    ValueError
        If the row length does not match the expected columns.
    """
    if len(row) != len(columns):
        message = f"Row for {table_key} has {len(row)} values, expected {len(columns)}"
        raise ValueError(message)
    return {column: row[index] for index, column in enumerate(columns)}


@dataclass
class DuckDBGateway:
    """Concrete StorageGateway implementation."""

    config: StorageConfig
    datasets: DatasetRegistry
    con: DuckDBConnection
    ibis: IbisGateway = field(init=False)
    analytics: AnalyticsTables = field(init=False)
    build: BuildTracking = field(init=False)
    core: CoreTables = field(init=False)
    docs: DocsViews = field(init=False)
    graph: GraphTables = field(init=False)
    runs: PipelineRunTracking = field(init=False)

    def __post_init__(self) -> None:
        """Initialize table accessor instances after dataclass init."""
        self.ibis = IbisGateway(self)
        self.analytics = AnalyticsTables(self.con)
        self.build = BuildTracking(self)
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
