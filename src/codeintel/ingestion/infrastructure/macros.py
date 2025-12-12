"""Ingestion macro utilities.

This module provides utilities for working with DuckDB ingestion macros,
including the canonical list of tables that have ingest macros defined.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.storage.gateway.protocol import DuckDBError

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import DuckDBConnection


INGEST_MACRO_TABLES: frozenset[str] = frozenset(
    {
        "analytics.coverage_lines",
        "analytics.function_metrics",
        "analytics.function_types",
        "analytics.goid_risk_factors",
        "analytics.static_diagnostics",
        "analytics.test_catalog",
        "analytics.test_coverage_edges",
        "analytics.typedness",
        "core.ast_metrics",
        "core.ast_nodes",
        "core.cst_nodes",
        "core.docstrings",
        "core.file_state",
        "core.goid_crosswalk",
        "core.goids",
        "core.modules",
        "core.repo_map",
        "graph.call_graph_edges",
        "graph.call_graph_nodes",
        "graph.cfg_blocks",
        "graph.cfg_edges",
        "graph.dfg_edges",
        "graph.import_graph_edges",
        "graph.symbol_use_edges",
    }
)


def macro_exists(con: DuckDBConnection, table_key: str) -> bool:
    """Check if an ingest macro exists for the given table key.

    Parameters
    ----------
    con
        DuckDB connection.
    table_key
        Table key in format 'schema.table_name' (e.g., 'analytics.coverage_lines').

    Returns
    -------
    bool
        True if macro exists.
    """
    _, table_name = table_key.split(".", maxsplit=1)
    macro_name = f"ingest_{table_name}"
    try:
        result = con.execute(
            "SELECT * FROM duckdb_functions() WHERE function_name = ? AND schema_name = 'metadata'",
            [macro_name],
        ).fetchone()
    except DuckDBError:
        return False
    return result is not None


__all__ = [
    "INGEST_MACRO_TABLES",
    "macro_exists",
]
