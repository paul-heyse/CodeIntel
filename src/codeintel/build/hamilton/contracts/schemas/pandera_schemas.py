"""Pandera schema definitions for CodeIntel datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from pandera import Check, Column, DataFrameSchema
from pandera.errors import SchemaErrors

from codeintel.build.schemas.registry import get_schema_provider
from codeintel.core.schemas.pandera_types import PanderaDtype, dtype_for_column_type

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from codeintel.core.schemas.primitives import ColumnType, TableSchema

__all__ = [
    "ValidationResult",
    "validate_with_result",
]


_DATAFRAME_CHECKS: dict[str, list[Check]] = {
    "core.goids": [
        Check(
            lambda df: ~df.duplicated(subset=["repo", "commit", "goid_h128"]).any(),
            error="Duplicate (repo, commit, goid_h128) in core.goids",
        ),
        Check(
            lambda df: ~df.duplicated(subset=["repo", "commit", "urn"]).any(),
            error="Duplicate (repo, commit, urn) in core.goids",
        ),
        Check(
            lambda df: df["end_line"].isna() | (df["end_line"] >= df["start_line"]),
            error="end_line must be >= start_line when present",
        ),
    ],
    "core.goid_crosswalk": [
        Check(
            lambda df: df["end_line"].isna() | (df["end_line"] >= df["start_line"]),
            error="end_line must be >= start_line when present",
        ),
    ],
    "core.docstrings": [
        Check(
            lambda df: df["end_lineno"].isna() | (df["end_lineno"] >= df["lineno"]),
            error="end_lineno must be >= lineno when present",
        ),
    ],
    "analytics.function_metrics": [
        Check(
            lambda df: df["end_line"].isna() | (df["end_line"] >= df["start_line"]),
            error="end_line must be >= start_line when present",
        ),
    ],
    "analytics.function_types": [
        Check(
            lambda df: df["end_line"].isna() | (df["end_line"] >= df["start_line"]),
            error="end_line must be >= start_line when present",
        ),
        Check(
            lambda df: df["annotated_params"].isna()
            | df["total_params"].isna()
            | (df["annotated_params"] <= df["total_params"]),
            error="annotated_params must be <= total_params",
        ),
    ],
    "analytics.function_profile": [
        Check(
            lambda df: df["covered_lines"].isna()
            | df["executable_lines"].isna()
            | (df["covered_lines"] <= df["executable_lines"]),
            error="covered_lines must be <= executable_lines",
        ),
    ],
    "analytics.coverage_lines": [
        Check(
            lambda df: df["end_line"].isna() | (df["end_line"] >= df["start_line"]),
            error="end_line must be >= start_line when present",
        ),
    ],
    "analytics.coverage_functions": [
        Check(
            lambda df: df["covered_lines"].isna()
            | df["executable_lines"].isna()
            | (df["covered_lines"] <= df["executable_lines"]),
            error="covered_lines must be <= executable_lines",
        ),
    ],
    "graph.call_graph_nodes": [
        Check(
            lambda df: df["end_line"].isna() | (df["end_line"] >= df["start_line"]),
            error="end_line must be >= start_line when present",
        ),
    ],
    "graph.cfg_blocks": [
        Check(
            lambda df: df["end_lineno"].isna() | (df["end_lineno"] >= df["lineno"]),
            error="end_lineno must be >= lineno when present",
        ),
    ],
    "graph.dfg_edges": [
        Check(
            lambda df: df["def_line"].isna() | df["use_line"].isna() | True,
            error="",
        ),
    ],
    "analytics.subsystem_agreement": [
        Check(
            lambda df: df["disagree_count"].isna()
            | df["agree_count"].isna()
            | df["disagree_count"].isna()
            | True,
            error="",
        ),
    ],
    "analytics.test_graph_metrics_functions": [
        Check(
            lambda df: df["failed_test_count"].isna()
            | df["test_count"].isna()
            | (df["failed_test_count"] <= df["test_count"]),
            error="failed_test_count must be <= test_count",
        ),
    ],
    "analytics.subsystem_coverage_cache": [
        Check(
            lambda df: df["failed_test_count"].isna()
            | df["test_count"].isna()
            | (df["failed_test_count"] <= df["test_count"]),
            error="failed_test_count must be <= test_count",
        ),
    ],
}


def _check_non_negative() -> Check:
    """
    Create a check for non-negative values (allows NA).

    Returns
    -------
    Check
        Pandera check that validates values >= 0.
    """
    return Check(lambda s: s.isna() | (s >= 0))


def _check_positive() -> Check:
    """
    Create a check for positive values >= 1 (allows NA).

    Returns
    -------
    Check
        Pandera check that validates values >= 1.
    """
    return Check(lambda s: s.isna() | (s >= 1))


def _check_ratio() -> Check:
    """
    Create a check for ratio values between 0 and 1 (allows NA).

    Returns
    -------
    Check
        Pandera check that validates 0 <= value <= 1.
    """
    return Check(lambda s: s.isna() | ((s >= 0) & (s <= 1)))


def _check_confidence() -> Check:
    """
    Create a check for confidence values between 0 and 1 (allows NA).

    Returns
    -------
    Check
        Pandera check that validates 0 <= value <= 1.
    """
    return Check(lambda s: s.isna() | ((s >= 0) & (s <= 1)))


_COLUMN_CHECKS: dict[str, dict[str, list[Check]]] = {
    "core.goids": {
        "goid_h128": [_check_non_negative()],
        "start_line": [_check_positive()],
        "end_line": [_check_positive()],
    },
    "core.goid_crosswalk": {
        "start_line": [_check_positive()],
        "end_line": [_check_positive()],
    },
    "core.ast_nodes": {
        "lineno": [_check_positive()],
        "end_lineno": [_check_positive()],
        "col_offset": [_check_non_negative()],
        "end_col_offset": [_check_non_negative()],
        "decorator_start_line": [_check_positive()],
        "decorator_end_line": [_check_positive()],
    },
    "core.ast_metrics": {
        "node_count": [_check_non_negative()],
        "function_count": [_check_non_negative()],
        "class_count": [_check_non_negative()],
        "avg_depth": [_check_non_negative()],
        "max_depth": [_check_non_negative()],
        "complexity": [_check_non_negative()],
    },
    "core.docstrings": {
        "lineno": [_check_positive()],
        "end_lineno": [_check_positive()],
    },
    "core.file_state": {
        "size_bytes": [_check_non_negative()],
        "mtime_ns": [_check_non_negative()],
    },
    "core.ingest_runs": {
        "duration_s": [_check_non_negative()],
        "rows_inserted": [_check_non_negative()],
        "rows_deleted": [_check_non_negative()],
        "modules_total": [_check_non_negative()],
        "modules_changed": [_check_non_negative()],
        "modules_deleted": [_check_non_negative()],
        "modules_changed_ratio": [_check_ratio()],
        "modules_deleted_ratio": [_check_ratio()],
    },
    "core.scip_occurrences": {
        "start_line": [_check_non_negative()],
        "start_col": [_check_non_negative()],
        "end_line": [_check_non_negative()],
        "end_col": [_check_non_negative()],
        "roles": [_check_non_negative()],
    },
    "core.scip_symbol_information": {
        "kind": [_check_non_negative()],
    },
    "core.scip_symbol_relationships": {
        "relationship_kind": [
            Check.isin(
                [
                    "definition",
                    "implementation",
                    "reference",
                    "type_definition",
                ]
            )
        ],
    },
    "core.scip_diagnostics": {
        "start_line": [_check_non_negative()],
        "start_col": [_check_non_negative()],
        "end_line": [_check_non_negative()],
        "end_col": [_check_non_negative()],
    },
    "core.scip_external_symbols": {
        "symbol": [Check.str_length(min_value=1)],
    },
    "core.scip_module_state": {
        "rel_path": [Check.str_length(min_value=1)],
        "content_hash": [Check.str_length(min_value=1)],
        "shard_path": [Check.str_length(min_value=1)],
    },
    "analytics.function_metrics": {
        "function_goid_h128": [_check_non_negative()],
        "loc": [_check_non_negative()],
        "logical_loc": [_check_non_negative()],
        "cyclomatic_complexity": [_check_non_negative()],
        "start_line": [_check_positive()],
        "end_line": [_check_positive()],
        "param_count": [_check_non_negative()],
        "positional_params": [_check_non_negative()],
        "keyword_only_params": [_check_non_negative()],
        "return_count": [_check_non_negative()],
        "yield_count": [_check_non_negative()],
        "raise_count": [_check_non_negative()],
        "max_nesting_depth": [_check_non_negative()],
        "stmt_count": [_check_non_negative()],
        "decorator_count": [_check_non_negative()],
    },
    "analytics.function_types": {
        "function_goid_h128": [_check_non_negative()],
        "start_line": [_check_positive()],
        "end_line": [_check_positive()],
        "total_params": [_check_non_negative()],
        "annotated_params": [_check_non_negative()],
        "unannotated_params": [_check_non_negative()],
        "param_typed_ratio": [_check_ratio()],
    },
    "analytics.function_effects": {
        "function_goid_h128": [_check_non_negative()],
    },
    "analytics.function_contracts": {
        "function_goid_h128": [_check_non_negative()],
    },
    "analytics.function_validation": {
        "function_goid_h128": [_check_non_negative()],
    },
    "analytics.function_profile": {
        "function_goid_h128": [_check_non_negative()],
        "loc": [_check_non_negative()],
        "logical_loc": [_check_non_negative()],
        "cyclomatic_complexity": [_check_non_negative()],
        "risk_score": [_check_non_negative()],
        "coverage_ratio": [_check_ratio()],
        "file_typed_ratio": [_check_ratio()],
        "param_typed_ratio": [_check_ratio()],
        "hotspot_score": [_check_non_negative()],
        "call_fan_in": [_check_non_negative()],
        "call_fan_out": [_check_non_negative()],
        "test_count": [_check_non_negative()],
        "failing_test_count": [_check_non_negative()],
    },
    "analytics.function_history": {
        "function_goid_h128": [_check_non_negative()],
        "commit_count": [_check_non_negative()],
        "author_count": [_check_non_negative()],
        "lines_added": [_check_non_negative()],
        "lines_deleted": [_check_non_negative()],
        "days_since_last_change": [_check_non_negative()],
    },
    "analytics.function_ast_features": {
        "function_goid_h128": [_check_non_negative()],
        "control_flow_depth": [_check_non_negative()],
        "exception_handler_count": [_check_non_negative()],
        "loop_count": [_check_non_negative()],
        "conditional_count": [_check_non_negative()],
        "assertion_count": [_check_non_negative()],
        "call_count": [_check_non_negative()],
    },
    "analytics.goid_risk_factors": {
        "function_goid_h128": [_check_non_negative()],
        "cyclomatic_complexity": [_check_non_negative()],
        "fan_in_count": [_check_non_negative()],
        "fan_out_count": [_check_non_negative()],
        "risk_score": [_check_non_negative()],
    },
    "analytics.coverage_lines": {
        "start_line": [_check_positive()],
        "end_line": [_check_positive()],
        "hit_count": [_check_non_negative()],
    },
    "analytics.coverage_functions": {
        "function_goid_h128": [_check_non_negative()],
        "executable_lines": [_check_non_negative()],
        "covered_lines": [_check_non_negative()],
        "coverage_ratio": [_check_ratio()],
    },
    "analytics.hotspots": {
        "commit_count": [_check_non_negative()],
        "author_count": [_check_non_negative()],
        "lines_added": [_check_non_negative()],
        "lines_deleted": [_check_non_negative()],
        "complexity": [_check_non_negative()],
        "score": [_check_non_negative()],
    },
    "analytics.typedness": {
        "function_count": [_check_non_negative()],
        "typed_function_count": [_check_non_negative()],
        "partial_typed_count": [_check_non_negative()],
        "untyped_function_count": [_check_non_negative()],
        "typed_ratio": [_check_ratio()],
    },
    "analytics.graph_metrics_functions": {
        "function_goid_h128": [_check_non_negative()],
        "call_fan_in": [_check_non_negative()],
        "call_fan_out": [_check_non_negative()],
        "call_in_degree": [_check_non_negative()],
        "call_out_degree": [_check_non_negative()],
        "call_pagerank": [_check_non_negative()],
    },
    "analytics.graph_metrics_functions_ext": {
        "function_goid_h128": [_check_non_negative()],
        "call_betweenness": [_check_non_negative()],
        "call_closeness": [_check_non_negative()],
        "call_eigenvector": [_check_non_negative()],
        "call_harmonic": [_check_non_negative()],
        "call_triangle_count": [_check_non_negative()],
        "call_k_core": [_check_non_negative()],
    },
    "analytics.graph_metrics_modules": {
        "import_fan_in": [_check_non_negative()],
        "import_fan_out": [_check_non_negative()],
        "import_in_degree": [_check_non_negative()],
        "import_out_degree": [_check_non_negative()],
        "import_pagerank": [_check_non_negative()],
        "symbol_fan_in": [_check_non_negative()],
        "symbol_fan_out": [_check_non_negative()],
    },
    "analytics.graph_metrics_modules_ext": {
        "import_betweenness": [_check_non_negative()],
        "import_closeness": [_check_non_negative()],
        "import_eigenvector": [_check_non_negative()],
        "import_harmonic": [_check_non_negative()],
        "import_k_core": [_check_non_negative()],
        "import_constraint": [_check_non_negative()],
        "import_effective_size": [_check_non_negative()],
        "import_shell_index": [_check_non_negative()],
        "import_component_size": [_check_non_negative()],
        "import_scc_size": [_check_non_negative()],
    },
    "analytics.subsystem_graph_metrics": {
        "import_fan_in": [_check_non_negative()],
        "import_fan_out": [_check_non_negative()],
        "import_in_degree": [_check_non_negative()],
        "import_out_degree": [_check_non_negative()],
        "import_pagerank": [_check_non_negative()],
        "import_betweenness": [_check_non_negative()],
        "import_closeness": [_check_non_negative()],
        "import_layer": [_check_non_negative()],
    },
    "analytics.symbol_graph_metrics_modules": {
        "symbol_fan_in": [_check_non_negative()],
        "symbol_fan_out": [_check_non_negative()],
        "symbol_in_degree": [_check_non_negative()],
        "symbol_out_degree": [_check_non_negative()],
        "symbol_pagerank": [_check_non_negative()],
    },
    "analytics.symbol_graph_metrics_functions": {
        "function_goid_h128": [_check_non_negative()],
        "symbol_fan_in": [_check_non_negative()],
        "symbol_fan_out": [_check_non_negative()],
    },
    "analytics.config_graph_metrics_keys": {
        "fan_in": [_check_non_negative()],
        "fan_out": [_check_non_negative()],
        "pagerank": [_check_non_negative()],
    },
    "analytics.config_graph_metrics_modules": {
        "config_fan_in": [_check_non_negative()],
        "config_fan_out": [_check_non_negative()],
        "config_pagerank": [_check_non_negative()],
    },
    "analytics.graph_stats": {
        "node_count": [_check_non_negative()],
        "edge_count": [_check_non_negative()],
        "density": [_check_non_negative()],
        "avg_degree": [_check_non_negative()],
        "max_in_degree": [_check_non_negative()],
        "max_out_degree": [_check_non_negative()],
        "scc_count": [_check_non_negative()],
        "largest_scc_size": [_check_non_negative()],
    },
    "analytics.test_catalog": {
        "duration_ms": [_check_non_negative()],
    },
    "analytics.test_coverage_edges": {
        "function_goid_h128": [_check_non_negative()],
    },
    "analytics.test_profile": {
        "duration_ms": [_check_non_negative()],
        "functions_covered_count": [_check_non_negative()],
    },
    "analytics.behavioral_coverage": {
        "test_goid_h128": [_check_non_negative()],
    },
    "analytics.test_graph_metrics_tests": {
        "functions_covered": [_check_non_negative()],
        "pagerank": [_check_non_negative()],
    },
    "analytics.test_graph_metrics_functions": {
        "function_goid_h128": [_check_non_negative()],
        "test_count": [_check_non_negative()],
        "passed_test_count": [_check_non_negative()],
        "failed_test_count": [_check_non_negative()],
        "pagerank": [_check_non_negative()],
    },
    "analytics.cfg_block_metrics": {
        "in_degree": [_check_non_negative()],
        "out_degree": [_check_non_negative()],
        "depth": [_check_non_negative()],
    },
    "analytics.cfg_function_metrics": {
        "function_goid_h128": [_check_non_negative()],
        "block_count": [_check_non_negative()],
        "edge_count": [_check_non_negative()],
        "cyclomatic_complexity": [_check_non_negative()],
        "max_depth": [_check_non_negative()],
    },
    "analytics.cfg_function_metrics_ext": {
        "function_goid_h128": [_check_non_negative()],
    },
    "analytics.dfg_block_metrics": {
        "in_degree": [_check_non_negative()],
        "out_degree": [_check_non_negative()],
    },
    "analytics.dfg_function_metrics": {
        "function_goid_h128": [_check_non_negative()],
        "def_count": [_check_non_negative()],
        "use_count": [_check_non_negative()],
        "edge_count": [_check_non_negative()],
    },
    "analytics.dfg_function_metrics_ext": {
        "function_goid_h128": [_check_non_negative()],
    },
    "analytics.subsystems": {
        "module_count": [_check_non_negative()],
        "internal_edge_count": [_check_non_negative()],
        "external_edge_count": [_check_non_negative()],
        "fan_in": [_check_non_negative()],
        "fan_out": [_check_non_negative()],
    },
    "analytics.subsystem_modules": {},
    "analytics.subsystem_agreement": {
        "agree_count": [_check_non_negative()],
        "disagree_count": [_check_non_negative()],
        "agreement_ratio": [_check_ratio()],
    },
    "analytics.subsystem_profile_cache": {
        "module_count": [_check_non_negative()],
        "function_count": [_check_non_negative()],
        "avg_risk_score": [_check_non_negative()],
        "max_risk_score": [_check_non_negative()],
        "high_risk_function_count": [_check_non_negative()],
    },
    "analytics.subsystem_coverage_cache": {
        "module_count": [_check_non_negative()],
        "function_count": [_check_non_negative()],
        "test_count": [_check_non_negative()],
        "passed_test_count": [_check_non_negative()],
        "failed_test_count": [_check_non_negative()],
        "skipped_test_count": [_check_non_negative()],
        "total_functions_covered": [_check_non_negative()],
        "avg_functions_covered": [_check_non_negative()],
        "function_coverage_ratio": [_check_ratio()],
    },
    "analytics.file_profile": {
        "function_count": [_check_non_negative()],
        "class_count": [_check_non_negative()],
        "loc": [_check_non_negative()],
        "complexity": [_check_non_negative()],
        "avg_risk_score": [_check_non_negative()],
        "max_risk_score": [_check_non_negative()],
        "high_risk_function_count": [_check_non_negative()],
        "coverage_ratio": [_check_ratio()],
        "typed_ratio": [_check_ratio()],
        "hotspot_score": [_check_non_negative()],
    },
    "analytics.module_profile": {
        "function_count": [_check_non_negative()],
        "import_fan_in": [_check_non_negative()],
        "import_fan_out": [_check_non_negative()],
        "symbol_fan_in": [_check_non_negative()],
        "symbol_fan_out": [_check_non_negative()],
        "avg_risk_score": [_check_non_negative()],
        "max_risk_score": [_check_non_negative()],
        "high_risk_function_count": [_check_non_negative()],
    },
    "analytics.history_timeseries": {
        "commit_count": [_check_non_negative()],
        "author_count": [_check_non_negative()],
        "lines_added": [_check_non_negative()],
        "lines_deleted": [_check_non_negative()],
    },
    "analytics.entrypoints": {
        "confidence": [_check_confidence()],
    },
    "analytics.external_dependencies": {
        "usage_count": [_check_non_negative()],
    },
    "analytics.external_dependency_calls": {
        "call_count": [_check_non_negative()],
        "function_goid_h128": [_check_non_negative()],
    },
    "analytics.semantic_roles_functions": {
        "function_goid_h128": [_check_non_negative()],
        "confidence": [_check_confidence()],
    },
    "analytics.semantic_roles_modules": {
        "confidence": [_check_confidence()],
    },
    "analytics.data_models": {
        "field_count": [_check_non_negative()],
    },
    "analytics.data_model_usage": {
        "usage_count": [_check_non_negative()],
    },
    "analytics.config_values": {
        "lineno": [_check_positive()],
    },
    "analytics.config_data_flow": {
        "source_line": [_check_positive()],
        "sink_line": [_check_positive()],
    },
    "analytics.static_diagnostics": {
        "line": [_check_positive()],
        "column": [_check_non_negative()],
    },
    "analytics.graph_validation": {
        "function_goid_h128": [_check_non_negative()],
    },
    "graph.call_graph_nodes": {
        "goid_h128": [_check_non_negative()],
        "start_line": [_check_positive()],
        "end_line": [_check_positive()],
    },
    "graph.call_graph_edges": {
        "caller_goid_h128": [_check_non_negative()],
        "callee_goid_h128": [_check_non_negative()],
        "callsite_line": [_check_positive()],
        "callsite_col": [_check_non_negative()],
        "confidence": [_check_confidence()],
    },
    "graph.import_graph_edges": {
        "src_fan_out": [_check_non_negative()],
        "dst_fan_in": [_check_non_negative()],
        "cycle_group": [_check_non_negative()],
        "module_layer": [_check_non_negative()],
    },
    "graph.import_modules": {
        "scc_id": [_check_non_negative()],
        "component_size": [_check_non_negative()],
        "layer": [_check_non_negative()],
        "cycle_group": [_check_non_negative()],
    },
    "graph.cfg_blocks": {
        "lineno": [_check_positive()],
        "end_lineno": [_check_positive()],
    },
    "graph.cfg_edges": {},
    "graph.dfg_edges": {
        "def_line": [_check_positive()],
        "use_line": [_check_positive()],
    },
    "graph.symbol_use_edges": {
        "src_line": [_check_positive()],
        "src_col": [_check_non_negative()],
        "dst_line": [_check_positive()],
        "dst_col": [_check_non_negative()],
    },
    "build.runs": {
        "duration_s": [_check_non_negative()],
        "rows_inserted": [_check_non_negative()],
        "rows_deleted": [_check_non_negative()],
    },
}


def _dtype_for_column_type(col_type: ColumnType | str) -> PanderaDtype:
    """Map a DuckDB column type to a Pandera-compatible dtype.

    Parameters
    ----------
    col_type
        DuckDB column type string (e.g., "VARCHAR", "INTEGER").

    Returns
    -------
    PanderaDtype
        A dtype that satisfies Pandera's Column dtype parameter.
    """
    return dtype_for_column_type(col_type)


def _build_columns(
    schema: TableSchema,
    *,
    column_checks: Mapping[str, list[Check]],
) -> dict[str, Column]:
    columns: dict[str, Column] = {}
    for col in schema.columns:
        checks = list(column_checks.get(col.name, ()))
        if col.type.upper() == "JSON":
            checks = []
        metadata = {"codeintel_column_type": col.type}
        columns[col.name] = Column(
            _dtype_for_column_type(col.type),
            nullable=col.nullable,
            checks=checks,
            metadata=metadata,
        )
    return columns


def _build_schema_from_table_schema(
    *,
    table_key: str,
    table_schema: TableSchema,
) -> DataFrameSchema:
    column_checks = _COLUMN_CHECKS.get(table_key, {})
    columns = _build_columns(table_schema, column_checks=column_checks)
    dataframe_checks = list(_DATAFRAME_CHECKS.get(table_key, ()))
    if table_schema.primary_key:
        dataframe_checks.append(
            Check(
                lambda df, subset=tuple(table_schema.primary_key): ~df.duplicated(
                    subset=subset
                ).any(),
                error=f"Duplicate primary key rows in {table_key}",
            )
        )
    return DataFrameSchema(
        columns,
        strict=True,
        coerce=True,
        checks=dataframe_checks,
        name=table_key,
    )


def _materialize_schemas() -> dict[str, DataFrameSchema]:
    schemas: dict[str, DataFrameSchema] = {}

    provider = get_schema_provider()
    for table_schema in provider.iter_table_schemas():
        schemas[table_schema.table_key] = _build_schema_from_table_schema(
            table_key=table_schema.table_key,
            table_schema=table_schema,
        )
    return schemas


def _analytics_view_schemas() -> dict[str, DataFrameSchema]:
    """
    Build Pandera schemas for analytics views.

    Returns
    -------
    dict[str, DataFrameSchema]
        View schemas keyed by view name.
    """
    view_schemas: dict[str, DataFrameSchema] = {}

    view_schemas["analytics.v_function_summary"] = DataFrameSchema(
        {
            "function_goid_h128": Column(_dtype_for_column_type("DECIMAL(38,0)"), nullable=True),
            "repo": Column(_dtype_for_column_type("VARCHAR")),
            "commit": Column(_dtype_for_column_type("VARCHAR")),
            "rel_path": Column(_dtype_for_column_type("VARCHAR")),
            "language": Column(_dtype_for_column_type("VARCHAR")),
            "kind": Column(_dtype_for_column_type("VARCHAR")),
            "qualname": Column(_dtype_for_column_type("VARCHAR")),
            "loc": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "logical_loc": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "param_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "positional_params": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "keyword_only_params": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "has_varargs": Column(_dtype_for_column_type("BOOLEAN"), nullable=True),
            "has_varkw": Column(_dtype_for_column_type("BOOLEAN"), nullable=True),
            "is_async": Column(_dtype_for_column_type("BOOLEAN"), nullable=True),
            "is_generator": Column(_dtype_for_column_type("BOOLEAN"), nullable=True),
            "return_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "yield_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "raise_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "cyclomatic_complexity": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "complexity_bucket": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "complexity_band": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "max_nesting_depth": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "stmt_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "decorator_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "has_docstring": Column(_dtype_for_column_type("BOOLEAN"), nullable=True),
            "created_at": Column(_dtype_for_column_type("TIMESTAMP"), nullable=True),
            "loc_bucket": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "param_typed_ratio": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "typedness_bucket": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "typedness_source": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "return_type": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "has_return_annotation": Column(_dtype_for_column_type("BOOLEAN"), nullable=True),
        },
        strict=True,
        coerce=True,
        name="analytics.v_function_summary",
    )

    view_schemas["analytics.v_function_hotspots"] = DataFrameSchema(
        {
            "function_goid_h128": Column(_dtype_for_column_type("DECIMAL(38,0)"), nullable=True),
            "repo": Column(_dtype_for_column_type("VARCHAR")),
            "commit": Column(_dtype_for_column_type("VARCHAR")),
            "rel_path": Column(_dtype_for_column_type("VARCHAR")),
            "language": Column(_dtype_for_column_type("VARCHAR")),
            "kind": Column(_dtype_for_column_type("VARCHAR")),
            "qualname": Column(_dtype_for_column_type("VARCHAR")),
            "hotspot_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "hotspot_normalized": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "cyclomatic_complexity": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "coverage_ratio": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "risk_level": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "complexity_bucket": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "typedness_bucket": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "created_at": Column(_dtype_for_column_type("TIMESTAMP"), nullable=True),
        },
        strict=True,
        coerce=True,
        name="analytics.v_function_hotspots",
    )

    return view_schemas


def _graph_view_schemas() -> dict[str, DataFrameSchema]:
    """
    Build Pandera schemas for graph views.

    Returns
    -------
    dict[str, DataFrameSchema]
        View schemas keyed by view name.
    """
    view_schemas: dict[str, DataFrameSchema] = {}

    view_schemas["graph.v_call_graph_degree"] = DataFrameSchema(
        {
            "repo": Column(_dtype_for_column_type("VARCHAR")),
            "commit": Column(_dtype_for_column_type("VARCHAR")),
            "function_goid_h128": Column(_dtype_for_column_type("DECIMAL(38,0)"), nullable=True),
            "call_out_degree": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "call_in_degree": Column(_dtype_for_column_type("INTEGER"), nullable=True),
        },
        strict=True,
        coerce=True,
        name="graph.v_call_graph_degree",
    )

    view_schemas["graph.v_import_graph_degree"] = DataFrameSchema(
        {
            "repo": Column(_dtype_for_column_type("VARCHAR")),
            "commit": Column(_dtype_for_column_type("VARCHAR")),
            "module": Column(_dtype_for_column_type("VARCHAR")),
            "import_out_degree": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "import_in_degree": Column(_dtype_for_column_type("INTEGER"), nullable=True),
        },
        strict=True,
        coerce=True,
        name="graph.v_import_graph_degree",
    )

    return view_schemas


def _core_view_schemas() -> dict[str, DataFrameSchema]:
    """
    Build Pandera schemas for core views.

    Returns
    -------
    dict[str, DataFrameSchema]
        View schemas keyed by view name.
    """
    view_schemas: dict[str, DataFrameSchema] = {}

    view_schemas["core.v_goid_crosswalk_join"] = DataFrameSchema(
        {
            "repo": Column(_dtype_for_column_type("VARCHAR")),
            "commit": Column(_dtype_for_column_type("VARCHAR")),
            "goid_h128": Column(_dtype_for_column_type("DECIMAL(38,0)"), nullable=True),
            "urn": Column(_dtype_for_column_type("VARCHAR")),
            "rel_path": Column(_dtype_for_column_type("VARCHAR")),
            "language": Column(_dtype_for_column_type("VARCHAR")),
            "kind": Column(_dtype_for_column_type("VARCHAR")),
            "qualname": Column(_dtype_for_column_type("VARCHAR")),
            "start_line": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "end_line": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "crosswalk_lang": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "module_path": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "file_path": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "ast_qualname": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "scip_symbol": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "updated_at": Column(_dtype_for_column_type("TIMESTAMP"), nullable=True),
        },
        strict=True,
        coerce=True,
        name="core.v_goid_crosswalk_join",
    )

    view_schemas["core.v_goid_crosswalk_mismatches"] = DataFrameSchema(
        {
            "repo": Column(_dtype_for_column_type("VARCHAR")),
            "commit": Column(_dtype_for_column_type("VARCHAR")),
            "goid_h128": Column(_dtype_for_column_type("DECIMAL(38,0)"), nullable=True),
            "urn": Column(_dtype_for_column_type("VARCHAR")),
            "crosswalk_urn": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "goid_language": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "crosswalk_language": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "goid_rel_path": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "crosswalk_file_path": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "goid_qualname": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "crosswalk_qualname": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "updated_at": Column(_dtype_for_column_type("TIMESTAMP"), nullable=True),
        },
        strict=True,
        coerce=True,
        name="core.v_goid_crosswalk_mismatches",
    )

    return view_schemas


def _docs_view_schemas() -> dict[str, DataFrameSchema]:
    """
    Build Pandera schemas for docs views.

    Returns
    -------
    dict[str, DataFrameSchema]
        View schemas keyed by view name.
    """
    view_schemas: dict[str, DataFrameSchema] = {}
    view_schemas["docs.v_function_summary"] = DataFrameSchema(
        {
            "function_goid_h128": Column(_dtype_for_column_type("DECIMAL(38,0)"), nullable=True),
            "urn": Column(_dtype_for_column_type("VARCHAR")),
            "repo": Column(_dtype_for_column_type("VARCHAR")),
            "commit": Column(_dtype_for_column_type("VARCHAR")),
            "rel_path": Column(_dtype_for_column_type("VARCHAR")),
            "language": Column(_dtype_for_column_type("VARCHAR")),
            "kind": Column(_dtype_for_column_type("VARCHAR")),
            "qualname": Column(_dtype_for_column_type("VARCHAR")),
            "loc": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "logical_loc": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "cyclomatic_complexity": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "complexity_bucket": Column(_dtype_for_column_type("VARCHAR")),
            "param_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "positional_params": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "keyword_only_params": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "has_varargs": Column(_dtype_for_column_type("BOOLEAN"), nullable=True),
            "has_varkw": Column(_dtype_for_column_type("BOOLEAN"), nullable=True),
            "is_async": Column(_dtype_for_column_type("BOOLEAN"), nullable=True),
            "is_generator": Column(_dtype_for_column_type("BOOLEAN"), nullable=True),
            "return_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "yield_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "raise_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "typedness_bucket": Column(_dtype_for_column_type("VARCHAR")),
            "typedness_source": Column(_dtype_for_column_type("VARCHAR")),
            "hotspot_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "file_typed_ratio": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "static_error_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "has_static_errors": Column(_dtype_for_column_type("BOOLEAN"), nullable=True),
            "executable_lines": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "covered_lines": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "coverage_ratio": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "tested": Column(_dtype_for_column_type("BOOLEAN"), nullable=True),
            "test_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "failing_test_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "last_test_status": Column(_dtype_for_column_type("VARCHAR")),
            "risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "risk_level": Column(_dtype_for_column_type("VARCHAR")),
            "tags": Column(_dtype_for_column_type("JSON")),
            "owners": Column(_dtype_for_column_type("JSON")),
            "created_at": Column(_dtype_for_column_type("TIMESTAMP"), nullable=True),
        },
        strict=True,
        coerce=True,
        name="docs.v_function_summary",
    )
    view_schemas["docs.v_call_graph_enriched"] = DataFrameSchema(
        {
            "caller_goid_h128": Column(_dtype_for_column_type("DECIMAL(38,0)"), nullable=True),
            "caller_repo": Column(_dtype_for_column_type("VARCHAR")),
            "caller_commit": Column(_dtype_for_column_type("VARCHAR")),
            "caller_urn": Column(_dtype_for_column_type("VARCHAR")),
            "caller_rel_path": Column(_dtype_for_column_type("VARCHAR")),
            "caller_qualname": Column(_dtype_for_column_type("VARCHAR")),
            "caller_risk_level": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "caller_risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "callee_goid_h128": Column(_dtype_for_column_type("DECIMAL(38,0)"), nullable=True),
            "callee_repo": Column(_dtype_for_column_type("VARCHAR")),
            "callee_commit": Column(_dtype_for_column_type("VARCHAR")),
            "callee_urn": Column(_dtype_for_column_type("VARCHAR")),
            "callee_rel_path": Column(_dtype_for_column_type("VARCHAR")),
            "callee_qualname": Column(_dtype_for_column_type("VARCHAR")),
            "callee_risk_level": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "callee_risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "callsite_path": Column(_dtype_for_column_type("VARCHAR")),
            "callsite_line": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "callsite_col": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "language": Column(_dtype_for_column_type("VARCHAR")),
            "kind": Column(_dtype_for_column_type("VARCHAR")),
            "resolved_via": Column(_dtype_for_column_type("VARCHAR")),
            "confidence": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "evidence_json": Column(_dtype_for_column_type("JSON")),
        },
        strict=True,
        coerce=True,
        name="docs.v_call_graph_enriched",
    )
    view_schemas["docs.v_subsystem_summary"] = DataFrameSchema(
        {
            "repo": Column(_dtype_for_column_type("VARCHAR")),
            "commit": Column(_dtype_for_column_type("VARCHAR")),
            "subsystem_id": Column(_dtype_for_column_type("VARCHAR")),
            "name": Column(_dtype_for_column_type("VARCHAR")),
            "description": Column(_dtype_for_column_type("VARCHAR")),
            "module_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "modules_json": Column(_dtype_for_column_type("JSON"), nullable=True),
            "entrypoints_json": Column(_dtype_for_column_type("JSON"), nullable=True),
            "internal_edge_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "external_edge_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "fan_in": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "fan_out": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "function_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "avg_risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "max_risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "high_risk_function_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "risk_level": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "subsystem_disagree_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "subsystem_member_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "subsystem_agreement_ratio": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "created_at": Column(_dtype_for_column_type("TIMESTAMP"), nullable=True),
        },
        strict=True,
        coerce=True,
        name="docs.v_subsystem_summary",
    )
    view_schemas["docs.v_module_with_subsystem"] = DataFrameSchema(
        {
            "repo": Column(_dtype_for_column_type("VARCHAR")),
            "commit": Column(_dtype_for_column_type("VARCHAR")),
            "subsystem_id": Column(_dtype_for_column_type("VARCHAR")),
            "subsystem_name": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "module": Column(_dtype_for_column_type("VARCHAR")),
            "role": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "rel_path": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "tags": Column(_dtype_for_column_type("JSON"), nullable=True),
            "owners": Column(_dtype_for_column_type("JSON"), nullable=True),
            "import_fan_in": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "import_fan_out": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "symbol_fan_in": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "symbol_fan_out": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "risk_level": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "avg_risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "max_risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
        },
        strict=True,
        coerce=True,
        name="docs.v_module_with_subsystem",
    )
    view_schemas["docs.v_subsystem_profile"] = DataFrameSchema(
        {
            "repo": Column(_dtype_for_column_type("VARCHAR")),
            "commit": Column(_dtype_for_column_type("VARCHAR")),
            "subsystem_id": Column(_dtype_for_column_type("VARCHAR")),
            "name": Column(_dtype_for_column_type("VARCHAR")),
            "description": Column(_dtype_for_column_type("VARCHAR")),
            "module_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "modules_json": Column(_dtype_for_column_type("JSON"), nullable=True),
            "entrypoints_json": Column(_dtype_for_column_type("JSON"), nullable=True),
            "internal_edge_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "external_edge_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "fan_in": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "fan_out": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "function_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "avg_risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "max_risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "high_risk_function_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "risk_level": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "import_in_degree": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "import_out_degree": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "import_pagerank": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "import_betweenness": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "import_closeness": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "import_layer": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "created_at": Column(_dtype_for_column_type("TIMESTAMP"), nullable=True),
        },
        strict=True,
        coerce=True,
        name="docs.v_subsystem_profile",
    )
    view_schemas["docs.v_subsystem_coverage"] = DataFrameSchema(
        {
            "repo": Column(_dtype_for_column_type("VARCHAR")),
            "commit": Column(_dtype_for_column_type("VARCHAR")),
            "subsystem_id": Column(_dtype_for_column_type("VARCHAR")),
            "name": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "description": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "module_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "function_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "risk_level": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "avg_risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "max_risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "test_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "passed_test_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "failed_test_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "skipped_test_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "xfail_test_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "flaky_test_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "total_functions_covered": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "avg_functions_covered": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "max_functions_covered": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "min_functions_covered": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "function_coverage_ratio": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "created_at": Column(_dtype_for_column_type("TIMESTAMP"), nullable=True),
        },
        strict=True,
        coerce=True,
        name="docs.v_subsystem_coverage",
    )

    view_schemas["docs.v_file_summary"] = DataFrameSchema(
        {
            "repo": Column(_dtype_for_column_type("VARCHAR")),
            "commit": Column(_dtype_for_column_type("VARCHAR")),
            "rel_path": Column(_dtype_for_column_type("VARCHAR")),
            "module": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "language": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "function_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "class_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "loc": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "complexity": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "avg_risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "max_risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "high_risk_function_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "coverage_ratio": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "typed_ratio": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "hotspot_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "static_error_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "tags": Column(_dtype_for_column_type("JSON"), nullable=True),
            "owners": Column(_dtype_for_column_type("JSON"), nullable=True),
        },
        strict=True,
        coerce=True,
        name="docs.v_file_summary",
    )

    view_schemas["docs.v_module_architecture"] = DataFrameSchema(
        {
            "repo": Column(_dtype_for_column_type("VARCHAR")),
            "commit": Column(_dtype_for_column_type("VARCHAR")),
            "module": Column(_dtype_for_column_type("VARCHAR")),
            "rel_path": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "function_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "import_fan_in": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "import_fan_out": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "import_pagerank": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "import_betweenness": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "import_closeness": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "symbol_fan_in": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "symbol_fan_out": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "avg_risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "max_risk_score": Column(_dtype_for_column_type("DOUBLE"), nullable=True),
            "risk_level": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "subsystem_id": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "subsystem_name": Column(_dtype_for_column_type("VARCHAR"), nullable=True),
            "layer": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "tags": Column(_dtype_for_column_type("JSON"), nullable=True),
            "owners": Column(_dtype_for_column_type("JSON"), nullable=True),
        },
        strict=True,
        coerce=True,
        name="docs.v_module_architecture",
    )

    view_schemas["docs.v_validation_summary"] = DataFrameSchema(
        {
            "repo": Column(_dtype_for_column_type("VARCHAR")),
            "commit": Column(_dtype_for_column_type("VARCHAR")),
            "validation_type": Column(_dtype_for_column_type("VARCHAR")),
            "issue_count": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "affected_files": Column(_dtype_for_column_type("INTEGER"), nullable=True),
            "affected_functions": Column(_dtype_for_column_type("INTEGER"), nullable=True),
        },
        strict=True,
        coerce=True,
        name="docs.v_validation_summary",
    )

    return view_schemas


@lru_cache(maxsize=1)
def _get_dataset_schemas() -> dict[str, DataFrameSchema]:
    """Lazily build and return the dataset schemas registry.

    Returns
    -------
    dict[str, DataFrameSchema]
        All registered dataset schemas.
    """
    schemas = _materialize_schemas()
    for view_map in (
        _analytics_view_schemas(),
        _graph_view_schemas(),
        _core_view_schemas(),
        _docs_view_schemas(),
    ):
        for table_key, view_schema in view_map.items():
            schemas.setdefault(table_key, view_schema)
    return schemas


@dataclass
class ValidationResult:
    """Result of Pandera validation with detailed error information."""

    success: bool
    validated_df: pd.DataFrame | None
    errors: list[str]
    error_count: int
    table_key: str

    @classmethod
    def ok(cls, table_key: str, df: pd.DataFrame) -> ValidationResult:
        """
        Create a successful validation result.

        Parameters
        ----------
        table_key
            Fully qualified table name.
        df
            Validated DataFrame.

        Returns
        -------
        ValidationResult
            Success result with validated DataFrame.
        """
        return cls(
            success=True,
            validated_df=df,
            errors=[],
            error_count=0,
            table_key=table_key,
        )

    @classmethod
    def failed(cls, table_key: str, errors: list[str], error_count: int) -> ValidationResult:
        """
        Create a failed validation result.

        Parameters
        ----------
        table_key
            Fully qualified table name.
        errors
            List of error messages.
        error_count
            Count of validation failures.

        Returns
        -------
        ValidationResult
            Failure result with error details.
        """
        return cls(
            success=False,
            validated_df=None,
            errors=errors,
            error_count=error_count,
            table_key=table_key,
        )


def validate_with_result(
    table_key: str,
    df: pd.DataFrame,
    *,
    strict: bool = True,
) -> ValidationResult:
    """
    Validate a DataFrame and return a detailed result object.

    Parameters
    ----------
    table_key
        Fully qualified table name.
    df
        DataFrame to validate.
    strict
        If True, validation errors are captured but not raised.
        If False, validation passes through gracefully on error.

    Returns
    -------
    ValidationResult
        Detailed validation outcome with errors when applicable.
    """
    log = logging.getLogger(__name__)
    schema = _get_dataset_schemas().get(table_key)
    if schema is None:
        return ValidationResult.ok(table_key, df)

    try:
        validated = schema.validate(df, lazy=True)
        return ValidationResult.ok(table_key, validated)
    except SchemaErrors as exc:
        errors = [str(case) for case in exc.failure_cases.itertuples()]
        error_count = len(exc.failure_cases)
        log.warning(
            "Pandera validation failed for %s: %d errors",
            table_key,
            error_count,
        )
        if strict:
            return ValidationResult.failed(table_key, errors, error_count)
        return ValidationResult.ok(table_key, df)


def _json_type_for_dtype(dtype: object) -> tuple[str, str | None]:
    dtype_str = str(dtype).lower()
    if "bool" in dtype_str:
        return "boolean", None
    if "int" in dtype_str:
        return "integer", None
    if "float" in dtype_str or "double" in dtype_str:
        return "number", None
    if "datetime" in dtype_str:
        return "string", "date-time"
    return "string", None


def _extract_column_constraints(column: Column) -> dict[str, Any]:
    """
    Extract constraint metadata from Pandera column checks.

    Parameters
    ----------
    column
        Pandera column definition.

    Returns
    -------
    dict[str, Any]
        JSON Schema constraint properties (minimum, maximum, etc.).
    """
    constraints: dict[str, Any] = {}

    if column.checks is None:
        return constraints

    for check in column.checks:
        check_str = str(check)

        if ">= 0" in check_str or "(s >= 0)" in check_str:
            constraints["minimum"] = 0
        elif ">= 1" in check_str or "(s >= 1)" in check_str:
            constraints["minimum"] = 1
        elif "<= 1" in check_str and ">= 0" in check_str:
            constraints["minimum"] = 0
            constraints["maximum"] = 1

    return constraints


def _column_metadata_type(column: Column) -> str | None:
    metadata = getattr(column, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get("codeintel_column_type")
    return None


def _json_value_types(*, nullable: bool) -> list[str]:
    types = ["object", "array", "string", "number", "boolean"]
    if nullable:
        types.append("null")
    return types


def _build_field_schema(column: Column, *, include_constraints: bool) -> dict[str, Any]:
    if _column_metadata_type(column) == "JSON":
        return {"type": _json_value_types(nullable=column.nullable)}

    json_type, fmt = _json_type_for_dtype(column.dtype)
    types = [json_type]
    if column.nullable:
        types.append("null")
    field_schema: dict[str, Any] = {"type": types}
    if fmt is not None:
        field_schema["format"] = fmt
    if include_constraints:
        constraints = _extract_column_constraints(column)
        field_schema.update(constraints)
    return field_schema


def _build_json_schema_properties(
    df_schema: DataFrameSchema,
    *,
    include_constraints: bool,
) -> tuple[dict[str, Any], list[str]]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, column in df_schema.columns.items():
        properties[name] = _build_field_schema(
            column,
            include_constraints=include_constraints,
        )
        if not column.nullable:
            required.append(name)
    return properties, required


def _apply_schema_metadata(
    schema: dict[str, Any],
    *,
    include_metadata: bool,
    schema_name: str | None,
) -> None:
    if include_metadata and schema_name:
        schema["title"] = schema_name
        if schema_name in _SCHEMA_DESCRIPTIONS:
            schema["description"] = _SCHEMA_DESCRIPTIONS[schema_name]


def pandera_to_json_schema(
    df_schema: DataFrameSchema,
    *,
    include_constraints: bool = True,
    include_metadata: bool = True,
) -> dict[str, Any]:
    """
    Convert a Pandera DataFrameSchema to a JSON Schema draft 2020-12 mapping.

    Parameters
    ----------
    df_schema
        Pandera schema to convert.
    include_constraints
        Whether to include numeric constraints from checks.
    include_metadata
        Whether to include schema metadata (name, description).

    Returns
    -------
    dict[str, Any]
        JSON Schema describing the dataframe structure.
    """
    properties, required = _build_json_schema_properties(
        df_schema,
        include_constraints=include_constraints,
    )
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    _apply_schema_metadata(
        schema,
        include_metadata=include_metadata,
        schema_name=df_schema.name,
    )

    return schema


_SCHEMA_DESCRIPTIONS: dict[str, str] = {
    "core.goids": "Global Object Identifiers for all tracked code entities.",
    "core.goid_crosswalk": "Cross-reference table linking GOIDs to AST/SCIP symbols.",
    "analytics.function_metrics": "Structural complexity metrics for functions and methods.",
    "analytics.function_types": "Type annotation coverage for functions and methods.",
    "analytics.goid_risk_factors": "Composite risk factors per function GOID.",
    "graph.call_graph_nodes": "Nodes in the function call graph.",
    "graph.call_graph_edges": "Edges representing function calls.",
    "graph.import_graph_edges": "Module-level import dependencies.",
    "docs.v_function_summary": "Enriched function view for documentation.",
    "docs.v_call_graph_enriched": "Call graph edges with caller/callee metadata.",
    "docs.v_subsystem_summary": "Subsystem overview with structure and risk profile.",
}
