"""Composition root types and dataset registry helpers for DuckDB access."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import duckdb

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.schemas import apply_all_schemas, assert_schema_alignment
from codeintel.storage.views import create_all_views

DOCS_VIEWS: tuple[str, ...] = (
    "docs.v_function_summary",
    "docs.v_call_graph_enriched",
    "docs.v_function_architecture",
    "docs.v_module_architecture",
    "docs.v_symbol_module_graph",
    "docs.v_config_graph_metrics_keys",
    "docs.v_config_graph_metrics_modules",
    "docs.v_config_projection_key_edges",
    "docs.v_config_projection_module_edges",
    "docs.v_config_data_flow",
    "docs.v_subsystem_agreement",
    "docs.v_cfg_block_architecture",
    "docs.v_dfg_block_architecture",
    "docs.v_subsystem_summary",
    "docs.v_module_with_subsystem",
    "docs.v_ide_hints",
    "docs.v_entrypoints",
    "docs.v_external_dependencies",
    "docs.v_external_dependency_calls",
    "docs.v_data_models",
    "docs.v_data_model_usage",
    "docs.v_test_to_function",
    "docs.v_file_summary",
    "docs.v_function_profile",
    "docs.v_file_profile",
    "docs.v_module_profile",
)


@dataclass(frozen=True)
class StorageConfig:
    """Define configuration for opening a CodeIntel DuckDB database."""

    db_path: Path
    read_only: bool = False
    apply_schema: bool = False
    ensure_views: bool = False
    validate_schema: bool = True
    repo: str | None = None
    commit: str | None = None


@dataclass(frozen=True)
class DatasetRegistry:
    """Track known table and view dataset names."""

    mapping: Mapping[str, str]
    tables: tuple[str, ...]
    views: tuple[str, ...]

    @property
    def all_datasets(self) -> tuple[str, ...]:
        """
        Return all registered dataset identifiers.

        Returns
        -------
        tuple[str, ...]
            Combined table and view names.
        """
        return self.tables + self.views

    def resolve(self, name: str) -> str:
        """
        Return a validated dataset name.

        Parameters
        ----------
        name
            Dataset identifier to validate.

        Returns
        -------
        str
            Fully qualified dataset name.

        Raises
        ------
        KeyError
            If the dataset name is unknown.
        """
        if name not in self.mapping:
            message = f"Unknown dataset: {name}"
            raise KeyError(message)
        return self.mapping[name]


class StorageGateway(Protocol):
    """Expose DuckDB access along with dataset registry metadata."""

    config: StorageConfig
    datasets: DatasetRegistry
    core: CoreTables
    graph: GraphTables
    docs: DocsViews
    analytics: AnalyticsTables

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        """
        Return an open DuckDB connection.

        Returns
        -------
        duckdb.DuckDBPyConnection
            Live connection bound to the configured database.
        """
        ...

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        ...


def build_dataset_registry(*, include_views: bool = True) -> DatasetRegistry:
    """
    Build a dataset registry from known tables and docs views.

    Parameters
    ----------
    include_views
        When True, include docs views alongside base tables.

    Returns
    -------
    DatasetRegistry
        Registry containing table/view identifiers and validation helpers.
    """

    def _dataset_name(key: str) -> str:
        _, name = key.split(".", maxsplit=1) if "." in key else ("", key)
        return name

    table_keys = tuple(sorted(TABLE_SCHEMAS.keys()))
    view_keys = DOCS_VIEWS if include_views else ()
    mapping = {_dataset_name(key): key for key in table_keys}
    mapping.update({_dataset_name(key): key for key in view_keys})
    table_names = tuple(_dataset_name(key) for key in table_keys)
    view_names = tuple(_dataset_name(key) for key in view_keys)
    return DatasetRegistry(mapping=mapping, tables=table_names, views=view_names)


@dataclass(frozen=True)
class CoreTables:
    """Accessors for core schema tables."""

    con: duckdb.DuckDBPyConnection

    def goids(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for core.goids.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting core.goids.
        """
        return self.con.table("core.goids")

    def modules(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for core.modules.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting core.modules.
        """
        return self.con.table("core.modules")

    def repo_map(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for core.repo_map.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting core.repo_map.
        """
        return self.con.table("core.repo_map")

    def insert_repo_map(
        self,
        rows: Iterable[tuple[str, str, str, str, str]],
    ) -> None:
        """
        Insert rows into core.repo_map.

        Parameters
        ----------
        rows
            Iterable of (repo, commit, modules_json, overlays_json, generated_at_iso).
        """
        self.con.executemany(
            """
            INSERT INTO core.repo_map (repo, commit, modules, overlays, generated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

    def insert_modules(
        self,
        rows: Iterable[tuple[str, str, str, str]],
    ) -> None:
        """
        Insert rows into core.modules.

        Parameters
        ----------
        rows
            Iterable of (module, path, repo, commit).
        """
        self.con.executemany(
            """
            INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
            VALUES (?, ?, ?, ?, 'python', '[]', '[]')
            """,
            rows,
        )

    def insert_goids(
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
                str,
            ]
        ],
    ) -> None:
        """
        Insert rows into core.goids.

        Parameters
        ----------
        rows
            Iterable of (goid_h128, urn, repo, commit, rel_path, language, kind,
            qualname, start_line, end_line, created_at_iso).
        """
        self.con.executemany(
            """
            INSERT INTO core.goids (
                goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
                start_line, end_line, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


@dataclass(frozen=True)
class GraphTables:
    """Accessors for graph schema tables."""

    con: duckdb.DuckDBPyConnection

    def call_graph_edges(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for graph.call_graph_edges.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting graph.call_graph_edges.
        """
        return self.con.table("graph.call_graph_edges")

    def insert_call_graph_edges(
        self,
        rows: Iterable[
            tuple[
                str,
                str,
                int,
                int | None,
                str,
                int,
                int,
                str,
                str,
                str,
                float,
                str,
            ]
        ],
    ) -> None:
        """
        Insert rows into graph.call_graph_edges.

        Parameters
        ----------
        rows
            Iterable of (repo, commit, caller_goid_h128, callee_goid_h128,
            callsite_path, callsite_line, callsite_col, language, kind,
            resolved_via, confidence, evidence_json).
        """
        self.con.executemany(
            """
            INSERT INTO graph.call_graph_edges (
                repo, commit, caller_goid_h128, callee_goid_h128, callsite_path,
                callsite_line, callsite_col, language, kind, resolved_via, confidence,
                evidence_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def call_graph_nodes(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for graph.call_graph_nodes.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting graph.call_graph_nodes.
        """
        return self.con.table("graph.call_graph_nodes")

    def insert_call_graph_nodes(
        self,
        rows: Iterable[tuple[int, str, str, int, bool, str]],
    ) -> None:
        """
        Insert rows into graph.call_graph_nodes.

        Parameters
        ----------
        rows
            Iterable of (goid_h128, language, kind, arity, is_public, rel_path).
        """
        self.con.executemany(
            """
            INSERT INTO graph.call_graph_nodes (
                goid_h128, language, kind, arity, is_public, rel_path
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def import_graph_edges(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for graph.import_graph_edges.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting graph.import_graph_edges.
        """
        return self.con.table("graph.import_graph_edges")

    def insert_import_graph_edges(
        self,
        rows: Iterable[tuple[str, str, str, str, int, int, int]],
    ) -> None:
        """
        Insert rows into graph.import_graph_edges.

        Parameters
        ----------
        rows
            Iterable of (repo, commit, src_module, dst_module, src_fan_out,
            dst_fan_in, cycle_group).
        """
        self.con.executemany(
            """
            INSERT INTO graph.import_graph_edges (
                repo, commit, src_module, dst_module, src_fan_out, dst_fan_in, cycle_group
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def symbol_use_edges(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for graph.symbol_use_edges.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting graph.symbol_use_edges.
        """
        return self.con.table("graph.symbol_use_edges")

    def insert_symbol_use_edges(
        self,
        rows: Iterable[tuple[str, str, str, bool, bool]],
    ) -> None:
        """
        Insert rows into graph.symbol_use_edges.

        Parameters
        ----------
        rows
            Iterable of (symbol, def_path, use_path, same_file, same_module).
        """
        self.con.executemany(
            """
            INSERT INTO graph.symbol_use_edges (
                symbol, def_path, use_path, same_file, same_module
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

    def insert_cfg_blocks(
        self,
        rows: Iterable[tuple[int, int, str, str, str, int, int, str, str, int, int]],
    ) -> None:
        """
        Insert rows into graph.cfg_blocks.

        Parameters
        ----------
        rows
            Iterable of values matching cfg_blocks columns.
        """
        self.con.executemany(
            """
            INSERT INTO graph.cfg_blocks (
                function_goid_h128, block_idx, block_id, label, file_path, start_line,
                end_line, kind, stmts_json, in_degree, out_degree
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def insert_cfg_edges(
        self,
        rows: Iterable[tuple[int, str, str, str | None]],
    ) -> None:
        """
        Insert rows into graph.cfg_edges.

        Parameters
        ----------
        rows
            Iterable of (function_goid_h128, src_block_id, dst_block_id, edge_kind).
        """
        self.con.executemany(
            """
            INSERT INTO graph.cfg_edges (
                function_goid_h128, src_block_id, dst_block_id, edge_kind
            )
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )

    def insert_dfg_edges(
        self,
        rows: Iterable[tuple[int, str, str, str | None, str | None, str | None]],
    ) -> None:
        """
        Insert rows into graph.dfg_edges.

        Parameters
        ----------
        rows
            Iterable of values matching dfg_edges columns.
        """
        self.con.executemany(
            """
            INSERT INTO graph.dfg_edges (
                function_goid_h128, src_block_id, dst_block_id, src_var, dst_var,
                edge_kind
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


@dataclass(frozen=True)
class DocsViews:
    """Accessors for docs schema views."""

    con: duckdb.DuckDBPyConnection

    def function_summary(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for docs.v_function_summary.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting docs.v_function_summary.
        """
        return self.con.table("docs.v_function_summary")

    def call_graph_enriched(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for docs.v_call_graph_enriched.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting docs.v_call_graph_enriched.
        """
        return self.con.table("docs.v_call_graph_enriched")

    def function_profile(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for docs.v_function_profile.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting docs.v_function_profile.
        """
        return self.con.table("docs.v_function_profile")


@dataclass(frozen=True)
class AnalyticsTables:
    """Accessors for analytics schema tables."""

    con: duckdb.DuckDBPyConnection

    def function_metrics(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for analytics.function_metrics.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting analytics.function_metrics.
        """
        return self.con.table("analytics.function_metrics")

    def function_types(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for analytics.function_types.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting analytics.function_types.
        """
        return self.con.table("analytics.function_types")

    def coverage_functions(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for analytics.coverage_functions.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting analytics.coverage_functions.
        """
        return self.con.table("analytics.coverage_functions")

    def insert_coverage_functions(
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
                float,
                bool,
                str,
                str,
            ]
        ],
    ) -> None:
        """
        Insert rows into analytics.coverage_functions.

        Parameters
        ----------
        rows
            Iterable of values matching coverage_functions columns.
        """
        self.con.executemany(
            """
            INSERT INTO analytics.coverage_functions (
                function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
                start_line, end_line, executable_lines, covered_lines, coverage_ratio,
                tested, untested_reason, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def coverage_lines(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for analytics.coverage_lines.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting analytics.coverage_lines.
        """
        return self.con.table("analytics.coverage_lines")

    def insert_coverage_lines(
        self,
        rows: Iterable[tuple[str, str, str, int, bool, bool, int, int, str]],
    ) -> None:
        """
        Insert rows into analytics.coverage_lines.

        Parameters
        ----------
        rows
            Iterable of values matching coverage_lines columns.
        """
        self.con.executemany(
            """
            INSERT INTO analytics.coverage_lines (
                repo, commit, rel_path, line, is_executable, is_covered, hits,
                context_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def test_catalog(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for analytics.test_catalog.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting analytics.test_catalog.
        """
        return self.con.table("analytics.test_catalog")

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
        """
        Insert rows into analytics.test_catalog.

        Parameters
        ----------
        rows
            Iterable of values matching test_catalog columns.
        """
        self.con.executemany(
            """
            INSERT INTO analytics.test_catalog (
                test_id, test_goid_h128, urn, repo, commit, rel_path, qualname, kind,
                status, duration_ms, markers, parametrized, flaky, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def test_coverage_edges(self) -> duckdb.DuckDBPyRelation:
        """
        Return relation for analytics.test_coverage_edges.

        Returns
        -------
        duckdb.DuckDBPyRelation
            Relation selecting analytics.test_coverage_edges.
        """
        return self.con.table("analytics.test_coverage_edges")

    def insert_test_coverage_edges(
        self,
        rows: Iterable[
            tuple[
                str,
                int | None,
                int,
                str,
                str,
                str,
                str,
                str,
                int,
                int,
                float,
                str,
                str,
            ]
        ],
    ) -> None:
        """
        Insert rows into analytics.test_coverage_edges.

        Parameters
        ----------
        rows
            Iterable of values matching test_coverage_edges columns.
        """
        self.con.executemany(
            """
            INSERT INTO analytics.test_coverage_edges (
                test_id, test_goid_h128, function_goid_h128, urn, repo, commit, rel_path,
                qualname, covered_lines, executable_lines, coverage_ratio, last_status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

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
        """
        Insert rows into analytics.function_metrics.

        Parameters
        ----------
        rows
            Iterable of values matching function_metrics columns.
        """
        self.con.executemany(
            """
            INSERT INTO analytics.function_metrics (
                function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
                start_line, end_line, loc, logical_loc, param_count, positional_params,
                keyword_only_params, has_varargs, has_varkw, is_async, is_generator,
                return_count, yield_count, raise_count, cyclomatic_complexity,
                max_nesting_depth, stmt_count, decorator_count, has_docstring,
                complexity_bucket, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

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
        """
        Insert rows into analytics.goid_risk_factors.

        Parameters
        ----------
        rows
            Iterable of values matching goid_risk_factors columns.
        """
        self.con.executemany(
            """
            INSERT INTO analytics.goid_risk_factors (
                function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
                loc, logical_loc, cyclomatic_complexity, complexity_bucket,
                typedness_bucket, typedness_source, hotspot_score, file_typed_ratio,
                static_error_count, has_static_errors, executable_lines, covered_lines,
                coverage_ratio, tested, test_count, failing_test_count, last_test_status,
                risk_score, risk_level, tags, owners, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def insert_config_values(
        self,
        rows: Iterable[tuple[str, str, str, str | None, str | None, int]],
    ) -> None:
        """
        Insert rows into analytics.config_values.

        Parameters
        ----------
        rows
            Iterable of (config_path, format, key, reference_paths, reference_modules,
            reference_count).
        """
        self.con.executemany(
            """
            INSERT INTO analytics.config_values (
                config_path, format, key, reference_paths, reference_modules,
                reference_count
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def insert_typedness(
        self,
        rows: Iterable[tuple[str, str, str, int, str, int, bool]],
    ) -> None:
        """
        Insert rows into analytics.typedness.

        Parameters
        ----------
        rows
            Iterable of values matching typedness columns.
        """
        self.con.executemany(
            """
            INSERT INTO analytics.typedness (
                repo, commit, path, type_error_count, annotation_ratio, untyped_defs,
                overlay_needed
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def insert_static_diagnostics(
        self,
        rows: Iterable[tuple[str, str, str, int, int, int, int, bool]],
    ) -> None:
        """
        Insert rows into analytics.static_diagnostics.

        Parameters
        ----------
        rows
            Iterable of values matching static_diagnostics columns.
        """
        self.con.executemany(
            """
            INSERT INTO analytics.static_diagnostics (
                repo, commit, rel_path, pyrefly_errors, pyright_errors, ruff_errors,
                total_errors, has_errors
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def insert_graph_metrics_functions(
        self,
        rows: Iterable[
            tuple[
                str,
                str,
                int,
                int,
                int,
                int,
                int,
                float | None,
                float | None,
                float | None,
                bool,
                int | None,
                int | None,
                str,
            ]
        ],
    ) -> None:
        """
        Insert rows into analytics.graph_metrics_functions.

        Parameters
        ----------
        rows
            Iterable of values matching graph_metrics_functions columns.
        """
        self.con.executemany(
            """
            INSERT INTO analytics.graph_metrics_functions (
                repo, commit, function_goid_h128, call_fan_in, call_fan_out,
                call_in_degree, call_out_degree, call_pagerank, call_betweenness,
                call_closeness, call_cycle_member, call_cycle_id, call_layer, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def insert_graph_metrics_modules(
        self,
        rows: Iterable[
            tuple[
                str,
                str,
                str,
                int,
                int,
                int,
                int,
                float | None,
                float | None,
                float | None,
                bool,
                int | None,
                int | None,
                int,
                int,
                str,
            ]
        ],
    ) -> None:
        """
        Insert rows into analytics.graph_metrics_modules.

        Parameters
        ----------
        rows
            Iterable of values matching graph_metrics_modules columns.
        """
        self.con.executemany(
            """
            INSERT INTO analytics.graph_metrics_modules (
                repo, commit, module, import_fan_in, import_fan_out, import_in_degree,
                import_out_degree, import_pagerank, import_betweenness, import_closeness,
                import_cycle_member, import_cycle_id, import_layer, symbol_fan_in,
                symbol_fan_out, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

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
        """
        Insert rows into analytics.subsystems.

        Parameters
        ----------
        rows
            Iterable of values matching subsystems columns.
        """
        self.con.executemany(
            """
            INSERT INTO analytics.subsystems (
                repo, commit, subsystem_id, name, description, module_count, modules_json,
                entrypoints_json, internal_edge_count, external_edge_count, fan_in,
                fan_out, function_count, avg_risk_score, max_risk_score,
                high_risk_function_count, risk_level, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def insert_subsystem_modules(
        self,
        rows: Iterable[tuple[str, str, str, str, str | None]],
    ) -> None:
        """
        Insert rows into analytics.subsystem_modules.

        Parameters
        ----------
        rows
            Iterable of values matching subsystem_modules columns.
        """
        self.con.executemany(
            """
            INSERT INTO analytics.subsystem_modules (
                repo, commit, subsystem_id, module, role
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )


@dataclass
class _DuckDBGateway:
    """Concrete StorageGateway implementation."""

    config: StorageConfig
    datasets: DatasetRegistry
    con: duckdb.DuckDBPyConnection
    core: CoreTables = field(init=False)
    graph: GraphTables = field(init=False)
    docs: DocsViews = field(init=False)
    analytics: AnalyticsTables = field(init=False)

    def __post_init__(self) -> None:
        self.core = CoreTables(self.con)
        self.graph = GraphTables(self.con)
        self.docs = DocsViews(self.con)
        self.analytics = AnalyticsTables(self.con)

    def close(self) -> None:
        """Close the underlying connection."""
        self.con.close()


def _connect(config: StorageConfig) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection using the provided configuration.

    Parameters
    ----------
    config
        Storage configuration controlling path, schema application, and validation.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Live DuckDB connection with optional schema/views applied.
    """
    if not config.read_only and config.db_path != Path(":memory:"):
        config.db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(config.db_path), read_only=config.read_only)
    if config.apply_schema and not config.read_only:
        apply_all_schemas(con)
    if config.ensure_views and not config.read_only:
        create_all_views(con)
    if config.validate_schema:
        assert_schema_alignment(con, strict=True)
    return con


def open_gateway(config: StorageConfig) -> StorageGateway:
    """
    Create a StorageGateway bound to a DuckDB database.

    Parameters
    ----------
    config
        Storage configuration describing connection options.

    Returns
    -------
    StorageGateway
        Gateway exposing typed accessors and dataset registry.
    """
    datasets = build_dataset_registry()
    con = _connect(config)
    return _DuckDBGateway(config=config, datasets=datasets, con=con)


def open_memory_gateway(
    *,
    apply_schema: bool = True,
    ensure_views: bool = False,
    validate_schema: bool = True,
) -> StorageGateway:
    """
    Create an in-memory StorageGateway for tests.

    Parameters
    ----------
    apply_schema
        When True, apply all table schemas to the in-memory database.
    ensure_views
        When True, create docs views after schema application.
    validate_schema
        When True, validate schema alignment after setup.

    Returns
    -------
    StorageGateway
        Gateway backed by an in-memory DuckDB connection.
    """
    cfg = StorageConfig(
        db_path=Path(":memory:"),
        read_only=False,
        apply_schema=apply_schema,
        ensure_views=ensure_views,
        validate_schema=validate_schema,
    )
    return open_gateway(cfg)
