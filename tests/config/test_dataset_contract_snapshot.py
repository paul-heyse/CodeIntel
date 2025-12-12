"""Snapshot test to guard against accidental dataset/table loss during refactoring."""

from __future__ import annotations

import pytest

from codeintel.config.datasets import (
    get_dataset_contracts,
    get_dataset_contracts_by_table_key,
    get_row_bindings,
    get_table_schemas,
)

# Expected counts for snapshot verification (updated after TargetPlugin migration)
EXPECTED_DATASET_CONTRACTS_COUNT = 115
EXPECTED_TABLE_SCHEMAS_COUNT = 86
EXPECTED_ROW_BINDINGS_COUNT = 38


def test_dataset_contracts_count_snapshot() -> None:
    """Lock in the current DATASET_CONTRACTS count to detect accidental removal."""
    actual = len(get_dataset_contracts())
    if actual != EXPECTED_DATASET_CONTRACTS_COUNT:
        pytest.fail(f"Expected {EXPECTED_DATASET_CONTRACTS_COUNT} DATASET_CONTRACTS, got {actual}")


def test_dataset_contracts_by_table_key_count_snapshot() -> None:
    """Lock in the current DATASET_CONTRACTS_BY_TABLE_KEY count."""
    actual = len(get_dataset_contracts_by_table_key())
    if actual != EXPECTED_DATASET_CONTRACTS_COUNT:
        pytest.fail(
            f"Expected {EXPECTED_DATASET_CONTRACTS_COUNT} entries in "
            f"DATASET_CONTRACTS_BY_TABLE_KEY, got {actual}"
        )


def test_table_schemas_count_snapshot() -> None:
    """Lock in the current TABLE_SCHEMAS count to detect accidental removal."""
    actual = len(get_table_schemas())
    if actual != EXPECTED_TABLE_SCHEMAS_COUNT:
        pytest.fail(f"Expected {EXPECTED_TABLE_SCHEMAS_COUNT} TABLE_SCHEMAS, got {actual}")


def test_row_bindings_count_snapshot() -> None:
    """Lock in the current ROW_BINDINGS_BY_TABLE_KEY count."""
    actual = len(get_row_bindings())
    if actual != EXPECTED_ROW_BINDINGS_COUNT:
        pytest.fail(
            f"Expected {EXPECTED_ROW_BINDINGS_COUNT} ROW_BINDINGS_BY_TABLE_KEY, got {actual}"
        )


def test_all_tables_have_schemas() -> None:
    """Every table_key in DATASET_CONTRACTS should have a schema (unless it's a view)."""
    table_schemas = get_table_schemas()
    missing = [
        contract.table_key
        for contract in get_dataset_contracts().values()
        if not contract.is_view and contract.table_key not in table_schemas
    ]
    if missing:
        pytest.fail(f"Missing TABLE_SCHEMAS for non-view contracts: {missing}")


def test_row_bindings_have_table_schemas() -> None:
    """Every row binding should reference a valid table_key in TABLE_SCHEMAS."""
    table_schemas = get_table_schemas()
    missing = [
        table_key
        for table_key in get_row_bindings()
        if table_key not in table_schemas and not table_key.startswith("docs.")
    ]
    if missing:
        pytest.fail(f"ROW_BINDINGS_BY_TABLE_KEY references unknown TABLE_SCHEMAS: {missing}")
