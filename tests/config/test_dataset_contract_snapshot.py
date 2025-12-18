"""Snapshot test to guard against accidental dataset/table loss during refactoring."""

from __future__ import annotations

import pytest

from codeintel.build.schemas import (
    get_schema_provider,
    iter_contracts,
    iter_contracts_by_table_key,
    iter_row_bindings,
)

EXPECTED_DATASET_CONTRACTS_COUNT = 126
EXPECTED_TABLE_SCHEMAS_COUNT = 96
# Row bindings are now generated for ALL table schemas (not just hand-maintained ones)
EXPECTED_ROW_BINDINGS_COUNT = 96


def test_dataset_contracts_count_snapshot() -> None:
    """Lock in the current DATASET_CONTRACTS count to detect accidental removal."""
    actual = len(list(iter_contracts()))
    if actual != EXPECTED_DATASET_CONTRACTS_COUNT:
        pytest.fail(f"Expected {EXPECTED_DATASET_CONTRACTS_COUNT} DATASET_CONTRACTS, got {actual}")


def test_dataset_contracts_by_table_key_count_snapshot() -> None:
    """Lock in the current DATASET_CONTRACTS_BY_TABLE_KEY count."""
    actual = len(list(iter_contracts_by_table_key()))
    if actual != EXPECTED_DATASET_CONTRACTS_COUNT:
        pytest.fail(
            f"Expected {EXPECTED_DATASET_CONTRACTS_COUNT} entries in "
            f"DATASET_CONTRACTS_BY_TABLE_KEY, got {actual}"
        )


def test_table_schemas_count_snapshot() -> None:
    """Lock in the current TABLE_SCHEMAS count to detect accidental removal."""
    schema_provider = get_schema_provider()
    actual = len(list(schema_provider.iter_table_schemas()))
    if actual != EXPECTED_TABLE_SCHEMAS_COUNT:
        pytest.fail(f"Expected {EXPECTED_TABLE_SCHEMAS_COUNT} TABLE_SCHEMAS, got {actual}")


def test_row_bindings_count_snapshot() -> None:
    """Lock in the current ROW_BINDINGS_BY_TABLE_KEY count."""
    actual = len(list(iter_row_bindings()))
    if actual != EXPECTED_ROW_BINDINGS_COUNT:
        pytest.fail(
            f"Expected {EXPECTED_ROW_BINDINGS_COUNT} ROW_BINDINGS_BY_TABLE_KEY, got {actual}"
        )


def test_all_tables_have_schemas() -> None:
    """Every table_key in DATASET_CONTRACTS should have a schema (unless it's a view)."""
    schema_provider = get_schema_provider()
    table_schemas = {s.table_key: s for s in schema_provider.iter_table_schemas()}
    missing = [
        contract.table_key
        for contract in iter_contracts()
        if not contract.is_view and contract.table_key not in table_schemas
    ]
    if missing:
        pytest.fail(f"Missing TABLE_SCHEMAS for non-view contracts: {missing}")


def test_row_bindings_have_table_schemas() -> None:
    """Every row binding should reference a valid table_key in TABLE_SCHEMAS."""
    schema_provider = get_schema_provider()
    table_schemas = {s.table_key: s for s in schema_provider.iter_table_schemas()}
    missing = [
        binding.table_key
        for binding in iter_row_bindings()
        if binding.table_key not in table_schemas and not binding.table_key.startswith("docs.")
    ]
    if missing:
        pytest.fail(f"ROW_BINDINGS_BY_TABLE_KEY references unknown TABLE_SCHEMAS: {missing}")
