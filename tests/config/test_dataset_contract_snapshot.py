"""Snapshot test to guard against accidental dataset/table loss during refactoring."""

from __future__ import annotations

import pytest

from codeintel.config.datasets import (
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
    ROW_BINDINGS_BY_TABLE_KEY,
    TABLE_SCHEMAS,
)

# Expected counts for snapshot verification (captured before refactor)
EXPECTED_DATASET_CONTRACTS_COUNT = 108
EXPECTED_TABLE_SCHEMAS_COUNT = 79
EXPECTED_ROW_BINDINGS_COUNT = 36


def test_dataset_contracts_count_snapshot() -> None:
    """Lock in the current DATASET_CONTRACTS count to detect accidental removal."""
    actual = len(DATASET_CONTRACTS)
    if actual != EXPECTED_DATASET_CONTRACTS_COUNT:
        pytest.fail(f"Expected {EXPECTED_DATASET_CONTRACTS_COUNT} DATASET_CONTRACTS, got {actual}")


def test_dataset_contracts_by_table_key_count_snapshot() -> None:
    """Lock in the current DATASET_CONTRACTS_BY_TABLE_KEY count."""
    actual = len(DATASET_CONTRACTS_BY_TABLE_KEY)
    if actual != EXPECTED_DATASET_CONTRACTS_COUNT:
        pytest.fail(
            f"Expected {EXPECTED_DATASET_CONTRACTS_COUNT} entries in "
            f"DATASET_CONTRACTS_BY_TABLE_KEY, got {actual}"
        )


def test_table_schemas_count_snapshot() -> None:
    """Lock in the current TABLE_SCHEMAS count to detect accidental removal."""
    actual = len(TABLE_SCHEMAS)
    if actual != EXPECTED_TABLE_SCHEMAS_COUNT:
        pytest.fail(f"Expected {EXPECTED_TABLE_SCHEMAS_COUNT} TABLE_SCHEMAS, got {actual}")


def test_row_bindings_count_snapshot() -> None:
    """Lock in the current ROW_BINDINGS_BY_TABLE_KEY count."""
    actual = len(ROW_BINDINGS_BY_TABLE_KEY)
    if actual != EXPECTED_ROW_BINDINGS_COUNT:
        pytest.fail(
            f"Expected {EXPECTED_ROW_BINDINGS_COUNT} ROW_BINDINGS_BY_TABLE_KEY, got {actual}"
        )


def test_all_tables_have_schemas() -> None:
    """Every table_key in DATASET_CONTRACTS should have a schema (unless it's a view)."""
    missing = [
        contract.table_key
        for contract in DATASET_CONTRACTS.values()
        if not contract.is_view and contract.table_key not in TABLE_SCHEMAS
    ]
    if missing:
        pytest.fail(f"Missing TABLE_SCHEMAS for non-view contracts: {missing}")


def test_row_bindings_have_table_schemas() -> None:
    """Every row binding should reference a valid table_key in TABLE_SCHEMAS."""
    missing = [
        table_key
        for table_key in ROW_BINDINGS_BY_TABLE_KEY
        if table_key not in TABLE_SCHEMAS and not table_key.startswith("docs.")
    ]
    if missing:
        pytest.fail(f"ROW_BINDINGS_BY_TABLE_KEY references unknown TABLE_SCHEMAS: {missing}")
