"""Tests for normalized macros execution, schema validation, and helpers.

This module consolidates tests for:
- Normalized macro execution with zero-row limits
- render_macro DDL generation
- Dataset rows macro functionality
- Schema validation (macro outputs vs contract schemas)
- Drift detection (macros defined vs expected)
- Performance guardrails

Consolidated from:
- test_normalized_macros.py (original)
- test_normalized_macros_helper.py
- test_macro_schemas.py
- test_macro_performance.py
"""

from __future__ import annotations

import io
import re
from time import perf_counter

import pytest

from codeintel.config.datasets import get_dataset_contracts_by_table_key
from codeintel.export.export_jsonl import NORMALIZED_MACROS
from codeintel.storage.gateway import DuckDBError, StorageGateway
from codeintel.storage.macros.generation import render_macro
from codeintel.storage.metadata import (
    METADATA_SCHEMA_DDL,
    dataset_rows_only_entries,
    validate_normalized_macro_schemas,
)
from codeintel.storage.metadata import NORMALIZED_MACROS as BOOTSTRAP_MACROS
from codeintel.storage.sql import safe_macro_call

pytestmark = pytest.mark.smoke


def test_normalized_macros_execute(fresh_gateway: StorageGateway) -> None:
    """
    Every normalized macro should execute with a zero-row limit.

    This guards against missing macro definitions or signature drift.
    """
    con = fresh_gateway.con
    failures: list[str] = []
    for table_key, macro in sorted(NORMALIZED_MACROS.items()):
        try:
            sql, params = safe_macro_call(
                macro, [table_key, 0], allowed=set(NORMALIZED_MACROS.values())
            )
            con.execute(sql, params)
        except (DuckDBError, RuntimeError, ValueError) as exc:
            failures.append(f"{table_key} via {macro}: {exc}")
    if failures:
        message = "Normalized macro failures: " + "; ".join(failures)
        pytest.fail(message)


def test_render_macro_valid_table_key() -> None:
    """Verify render_macro returns RenderedMacro for valid table key."""
    result = render_macro("core.ast_nodes")

    assert result.macro_name.startswith("metadata.normalized_")
    assert "CREATE OR REPLACE MACRO" in result.ddl


def test_render_macro_unknown_table_key_raises() -> None:
    """Verify render_macro raises KeyError for unknown table key."""
    with pytest.raises(KeyError, match="Unknown table key"):
        render_macro("unknown.table_that_does_not_exist")


def test_render_macro_with_custom_limit() -> None:
    """Verify render_macro accepts custom default_limit."""
    result = render_macro("core.ast_nodes", default_limit=1000)

    assert ":= 1000" in result.ddl


def test_render_macro_includes_date_cast() -> None:
    """Verify render_macro correctly casts DATE columns through internal _cast_expression."""
    # Find a table with DATE columns (core.file_state likely has date columns)
    # We test the output DDL to verify DATE handling
    result = render_macro("analytics.function_history")

    # The DDL should contain proper casting for timestamp columns
    assert "CAST" in result.ddl


def test_render_macro_includes_goid_cast() -> None:
    """Verify render_macro casts goid_h128 columns to BIGINT."""
    result = render_macro("analytics.function_profile")

    # Should include BIGINT cast for goid columns
    assert "AS BIGINT" in result.ddl
    assert "goid_h128" in result.ddl.lower()


def test_render_macro_outputs_ddl_to_buffer() -> None:
    """Verify render_macro produces DDL strings for provided tables."""
    buffer = io.StringIO()

    for table in ("core.ast_nodes", "core.goids"):
        rendered = render_macro(table)
        buffer.write(rendered.ddl)

    output = buffer.getvalue()
    assert "metadata.normalized_ast_nodes" in output
    assert "metadata.normalized_goids" in output


# =============================================================================
# Dataset Rows Macro Tests (merged from test_metadata_dataset_rows_macro.py)
# =============================================================================


def test_dataset_rows_macro_handles_registry_datasets(
    fresh_gateway: StorageGateway,
) -> None:
    """
    Verify metadata.dataset_rows works for every dataset table_key in the registry.

    Uses a zero-row limit to avoid materializing data while exercising the macro.
    """
    con = fresh_gateway.con
    failures: list[str] = []
    for dataset_name, table_key in sorted(fresh_gateway.datasets.mapping.items()):
        try:
            con.execute(
                """
                SELECT 1
                FROM metadata.dataset_rows(?, 0, 0)
                LIMIT 0
                """,
                [table_key],
            )
        except (DuckDBError, RuntimeError, ValueError) as exc:
            failures.append(f"{dataset_name} ({table_key}): {exc}")

    if failures:
        message = "dataset_rows macro failures: " + "; ".join(failures)
        pytest.fail(message)


# =============================================================================
# Drift Detection Tests (merged from test_normalized_macros_helper.py)
# =============================================================================


def test_normalized_macros_defined(fresh_gateway: StorageGateway) -> None:
    """Ensure every macro referenced in NORMALIZED_MACROS exists in DuckDB."""
    _ = fresh_gateway  # Gateway ensures bootstrap runs.

    ddl_text = "\n".join(METADATA_SCHEMA_DDL)
    defined = {
        match.group(1).lower()
        for match in re.finditer(
            r"CREATE\s+OR\s+REPLACE\s+MACRO\s+([\w\.]+)", ddl_text, re.IGNORECASE
        )
    }
    missing: list[str] = []
    for macro in sorted(set(NORMALIZED_MACROS.values())):
        macro_lower = macro.lower()
        macro_name = macro_lower.split(".")[-1]
        if macro_lower not in defined and macro_name not in defined:
            missing.append(macro)
    if missing:
        message = "Missing normalized macros: " + ", ".join(missing)
        pytest.fail(message)


def test_normalized_macros_match_expected_sets() -> None:
    """Catch drift when adding datasets without macros or allowlisting explicitly."""
    datasets = {
        key
        for key, contract in get_dataset_contracts_by_table_key().items()
        if contract.schema is not None
    }
    macro_backed = set(NORMALIZED_MACROS)
    dataset_rows_only = set(dataset_rows_only_entries())
    unexpected_dataset_rows = datasets - macro_backed - dataset_rows_only
    if unexpected_dataset_rows:
        message = "Datasets missing normalized macros or allowlist entries: " + ", ".join(
            sorted(unexpected_dataset_rows)
        )
        pytest.fail(message)


def test_normalized_macro_schema_validation(fresh_gateway: StorageGateway) -> None:
    """Ensure schema validation helper raises on drift (no drift expected)."""
    con = fresh_gateway.con
    # Provide a sanity check that the helper executes successfully.
    validate_normalized_macro_schemas(con)
    # Keep export mapping aligned with bootstrap mapping.
    if set(BOOTSTRAP_MACROS) != set(NORMALIZED_MACROS):
        pytest.fail("Export and bootstrap macro mappings diverged")


def test_dataset_rows_only_tables_parse(fresh_gateway: StorageGateway) -> None:
    """Ensure dataset_rows-only tables at least parse with zero-row selects."""
    con = fresh_gateway.con
    failures: list[str] = []
    for table_key in dataset_rows_only_entries():
        try:
            sql, params = safe_macro_call(
                "metadata.dataset_rows", [table_key, 0, 0], allowed={"metadata.dataset_rows"}
            )
            con.sql(sql, params=params).fetchall()
        except (DuckDBError, RuntimeError, ValueError) as exc:
            failures.append(f"{table_key}: {exc}")
    if failures:
        pytest.fail("; ".join(failures))


# =============================================================================
# Schema Validation Tests (merged from test_macro_schemas.py)
# =============================================================================


def _canonical_type(type_str: str) -> str:
    """
    Canonicalize a SQL type string for comparison.

    Returns
    -------
    str
        Canonical type representation.
    """
    upper = type_str.upper()
    if upper in {"TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"}:
        return "TIMESTAMPTZ"
    if upper.startswith("DECIMAL") or upper == "BIGINT":
        return "BIGINT"
    return upper


def test_macro_schemas_match_table_definitions(fresh_gateway: StorageGateway) -> None:
    """Normalized macro outputs should align with DatasetContract schema definitions."""
    con = fresh_gateway.con
    failures: list[str] = []
    for table_key, macro in sorted(NORMALIZED_MACROS.items()):
        contract = get_dataset_contracts_by_table_key().get(table_key)
        if contract is None or contract.schema is None:
            failures.append(f"{table_key}: no contract schema found")
            continue
        schema = contract.schema
        sql, params = safe_macro_call(
            macro, [table_key, 0, 0], allowed=set(NORMALIZED_MACROS.values())
        )
        rel = con.sql(sql, params=params)
        actual: dict[str, str] = {}
        for name, dtype in zip(rel.columns, rel.dtypes, strict=False):
            if name.endswith("_1"):
                continue
            actual[name] = _canonical_type(str(dtype))

        expected = {col.name: _canonical_type(col.type) for col in schema.columns}

        missing = expected.keys() - actual.keys()
        if missing:
            failures.append(f"{table_key}: missing columns {sorted(missing)}")
            continue
        for col_name, expected_type in expected.items():
            actual_type = actual[col_name]
            if expected_type in {"TIMESTAMP", "DATE"} and actual_type == "VARCHAR":
                continue
            if actual_type != expected_type:
                failures.append(f"{table_key}.{col_name}: {actual_type} != {expected_type}")
    if failures:
        pytest.fail("; ".join(failures))


# =============================================================================
# Performance Tests (merged from test_macro_performance.py)
# =============================================================================


@pytest.mark.parametrize("table_key", ["graph.call_graph_edges", "analytics.function_metrics"])
def test_normalized_macro_latency_smoke(fresh_gateway: StorageGateway, table_key: str) -> None:
    """
    Ensure representative normalized macros run quickly on empty data.

    The threshold is generous to avoid flakiness but still catches gross regressions.
    """
    con = fresh_gateway.con
    macro = NORMALIZED_MACROS[table_key]
    start = perf_counter()
    sql, params = safe_macro_call(macro, [table_key, 0, 0], allowed=set(NORMALIZED_MACROS.values()))
    _ = con.sql(sql, params=params).fetchall()
    duration = perf_counter() - start
    if duration > 1.0:
        pytest.fail(f"Macro {macro} exceeded latency threshold: {duration:.3f}s")
