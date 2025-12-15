"""Tests for the dataset contract single source of truth."""

from __future__ import annotations

import pytest

from codeintel.analytics.utilities.datasets import DELETE_SQL_BY_TABLE
from codeintel.build.schemas import (
    get_contract_provider,
    get_schema_provider,
    iter_contracts,
    iter_contracts_by_table_key,
)
from codeintel.config.datasets.rows.analytics import (
    FUNCTION_METRICS_COLUMNS,
    FUNCTION_TYPES_COLUMNS,
)
from codeintel.config.datasets.rows.profiles import (
    FILE_PROFILE_COLUMNS,
    FUNCTION_PROFILE_COLUMNS,
    GRAPH_METRICS_FUNCTIONS_COLUMNS,
    GRAPH_METRICS_FUNCTIONS_EXT_COLUMNS,
    GRAPH_METRICS_MODULES_COLUMNS,
    GRAPH_METRICS_MODULES_EXT_COLUMNS,
    MODULE_PROFILE_COLUMNS,
)
from codeintel.config.datasets.rows.test import (
    BEHAVIORAL_COVERAGE_COLUMNS,
    SUBSYSTEM_COVERAGE_COLUMNS,
    SUBSYSTEM_PROFILE_COLUMNS,
    TEST_COVERAGE_EDGE_COLUMNS,
    TEST_PROFILE_COLUMNS,
)


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def test_all_tables_have_contracts() -> None:
    """Every non-temporary table should have a DatasetContract entry."""
    contracts_by_key = dict(iter_contracts_by_table_key())
    schema_provider = get_schema_provider()
    missing = [
        schema.table_key
        for schema in schema_provider.iter_table_schemas()
        if not schema.table_key.startswith("tmp_") and schema.table_key not in contracts_by_key
    ]
    _require(condition=not missing, message=f"Missing contracts for: {missing}")


def test_json_schema_map_matches_contracts() -> None:
    """Derived JSON Schema mapping should mirror contract definitions."""
    expected = {
        contract.name: contract.json_schema_id
        for contract in iter_contracts()
        if contract.json_schema_id is not None
    }
    json_schema_map = get_contract_provider().json_schema_by_dataset_name
    _require(
        condition=expected == json_schema_map,
        message="JSON Schema mapping diverged from contracts",
    )


def test_capabilities_shape() -> None:
    """Capability flags should include read-only and view indicators."""
    contracts = {c.name: c for c in iter_contracts()}
    contract = contracts.get("function_profile")
    _require(condition=contract is not None, message="function_profile contract missing")
    if contract is None:
        return
    caps = contract.capabilities()
    expected_keys = {
        "can_validate",
        "can_export_jsonl",
        "can_export_parquet",
        "has_row_binding",
        "is_view",
        "docs_view",
        "read_only",
    }
    _require(condition=expected_keys.issubset(set(caps)), message="Capability keys missing")
    _require(condition=caps["is_view"] is False, message="function_profile marked as view")
    _require(condition=caps["read_only"] is False, message="function_profile marked read-only")


def test_column_names_method() -> None:
    """Column names method should return schema columns in order."""
    contracts = {c.name: c for c in iter_contracts()}
    contract = contracts.get("function_profile")
    _require(condition=contract is not None, message="function_profile contract missing")
    if contract is None:
        return
    columns = contract.column_names()
    _require(condition=len(columns) > 0, message="function_profile has no columns")
    _require(condition=columns[0] == "function_goid_h128", message="First column mismatch")
    _require(condition="repo" in columns, message="repo column missing")
    _require(condition="commit" in columns, message="commit column missing")


def test_contract_derived_columns_match_schemas() -> None:
    """Contract-derived columns should match original TABLE_SCHEMAS definitions."""
    schema_provider = get_schema_provider()
    table_schemas = {s.table_key: s for s in schema_provider.iter_table_schemas()}
    mismatches: list[str] = []
    for table_key, contract in iter_contracts_by_table_key():
        if contract.schema is None:
            continue
        if table_key not in table_schemas:
            continue
        original_schema = table_schemas[table_key]
        contract_cols = contract.column_names()
        original_cols = tuple(original_schema.column_names())
        if contract_cols != original_cols:
            mismatches.append(
                f"{table_key}: contract={contract_cols[:3]}... vs original={original_cols[:3]}..."
            )
    _require(
        condition=not mismatches,
        message=f"Column order mismatches: {', '.join(mismatches[:5])}",
    )


def test_delete_sql_covers_repo_commit_tables() -> None:
    """DELETE_SQL_BY_TABLE should cover all datasets with repo+commit columns."""
    missing: list[str] = []
    for table_key, contract in iter_contracts_by_table_key():
        if contract.schema is None or contract.is_view:
            continue
        col_names = contract.schema.column_names()
        has_repo_commit = "repo" in col_names and "commit" in col_names
        if has_repo_commit and table_key not in DELETE_SQL_BY_TABLE:
            missing.append(table_key)
    _require(
        condition=not missing,
        message=f"Tables with repo+commit missing from DELETE_SQL_BY_TABLE: {missing[:5]}",
    )


def test_row_models_column_constants_match_contracts() -> None:
    """Column constants in dataset_contract.py should match contract-derived columns."""
    column_mappings = [
        ("analytics.function_metrics", FUNCTION_METRICS_COLUMNS),
        ("analytics.function_types", FUNCTION_TYPES_COLUMNS),
        ("analytics.graph_metrics_functions", GRAPH_METRICS_FUNCTIONS_COLUMNS),
        ("analytics.graph_metrics_modules", GRAPH_METRICS_MODULES_COLUMNS),
        ("analytics.graph_metrics_functions_ext", GRAPH_METRICS_FUNCTIONS_EXT_COLUMNS),
        ("analytics.graph_metrics_modules_ext", GRAPH_METRICS_MODULES_EXT_COLUMNS),
        ("analytics.test_coverage_edges", TEST_COVERAGE_EDGE_COLUMNS),
        ("analytics.function_profile", FUNCTION_PROFILE_COLUMNS),
        ("analytics.file_profile", FILE_PROFILE_COLUMNS),
        ("analytics.module_profile", MODULE_PROFILE_COLUMNS),
        ("analytics.test_profile", TEST_PROFILE_COLUMNS),
        ("analytics.behavioral_coverage", BEHAVIORAL_COVERAGE_COLUMNS),
        ("analytics.subsystem_profile_cache", SUBSYSTEM_PROFILE_COLUMNS),
        ("analytics.subsystem_coverage_cache", SUBSYSTEM_COVERAGE_COLUMNS),
    ]
    contracts_by_key = dict(iter_contracts_by_table_key())
    mismatches: list[str] = []
    for table_key, constant in column_mappings:
        contract = contracts_by_key.get(table_key)
        if contract is None or contract.schema is None:
            mismatches.append(f"{table_key}: no contract or schema")
            continue
        expected = contract.column_names()
        if tuple(constant) != expected:
            mismatches.append(
                f"{table_key}: constant={tuple(constant)[:3]}... vs contract={expected[:3]}..."
            )
    _require(
        condition=not mismatches,
        message=f"Column constant mismatches: {', '.join(mismatches[:5])}",
    )
