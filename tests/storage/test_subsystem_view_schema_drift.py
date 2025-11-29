"""Guardrail tests to detect schema drift in subsystem docs views."""

from __future__ import annotations

import duckdb
import pytest

from codeintel.storage.schemas import apply_all_schemas
from codeintel.storage.views import create_all_views

EXPECTED_SCHEMAS: dict[str, list[tuple[str, str]]] = {
    "docs.v_subsystem_profile": [
        ("repo", "VARCHAR"),
        ("commit", "VARCHAR"),
        ("subsystem_id", "VARCHAR"),
        ("name", "VARCHAR"),
        ("description", "VARCHAR"),
        ("module_count", "INTEGER"),
        ("modules_json", "JSON"),
        ("entrypoints_json", "JSON"),
        ("internal_edge_count", "INTEGER"),
        ("external_edge_count", "INTEGER"),
        ("fan_in", "INTEGER"),
        ("fan_out", "INTEGER"),
        ("function_count", "INTEGER"),
        ("avg_risk_score", "DOUBLE"),
        ("max_risk_score", "DOUBLE"),
        ("high_risk_function_count", "INTEGER"),
        ("risk_level", "VARCHAR"),
        ("import_in_degree", "DOUBLE"),
        ("import_out_degree", "DOUBLE"),
        ("import_pagerank", "DOUBLE"),
        ("import_betweenness", "DOUBLE"),
        ("import_closeness", "DOUBLE"),
        ("import_layer", "INTEGER"),
        ("created_at", "TIMESTAMP"),
    ],
    "docs.v_subsystem_coverage": [
        ("repo", "VARCHAR"),
        ("commit", "VARCHAR"),
        ("subsystem_id", "VARCHAR"),
        ("name", "VARCHAR"),
        ("description", "VARCHAR"),
        ("module_count", "INTEGER"),
        ("function_count", "INTEGER"),
        ("risk_level", "VARCHAR"),
        ("avg_risk_score", "DOUBLE"),
        ("max_risk_score", "DOUBLE"),
        ("test_count", "BIGINT"),
        ("passed_test_count", "HUGEINT"),
        ("failed_test_count", "HUGEINT"),
        ("skipped_test_count", "HUGEINT"),
        ("xfail_test_count", "HUGEINT"),
        ("flaky_test_count", "HUGEINT"),
        ("total_functions_covered", "HUGEINT"),
        ("avg_functions_covered", "DOUBLE"),
        ("max_functions_covered", "DOUBLE"),
        ("min_functions_covered", "DOUBLE"),
        ("function_coverage_ratio", "DOUBLE"),
        ("created_at", "TIMESTAMP"),
    ],
}


def _fetch_schema(con: duckdb.DuckDBPyConnection, view_key: str) -> list[tuple[str, str]]:
    schema_name, table_name = view_key.split(".", maxsplit=1)
    rows = con.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ?
        ORDER BY ordinal_position
        """,
        [schema_name, table_name],
    ).fetchall()
    return [(str(name), str(dtype)) for name, dtype in rows]


def test_subsystem_docs_views_schema_stable() -> None:
    """Detect unintended column drift for subsystem docs views."""
    con = duckdb.connect(database=":memory:")
    apply_all_schemas(con)
    create_all_views(con)
    for view_key, expected in EXPECTED_SCHEMAS.items():
        actual = _fetch_schema(con, view_key)
        if actual != expected:
            pytest.fail(f"{view_key} schema drift detected: {actual} != {expected}")
