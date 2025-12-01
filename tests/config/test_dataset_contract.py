"""Tests for the dataset contract single source of truth."""

from __future__ import annotations

import pytest

from codeintel.config.dataset_contract import (
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
    JSON_SCHEMA_BY_DATASET_NAME,
)
from codeintel.config.schemas.tables import TABLE_SCHEMAS


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def test_all_tables_have_contracts() -> None:
    """Every non-temporary table should have a DatasetContract entry."""
    missing = [
        table_key
        for table_key in TABLE_SCHEMAS
        if not table_key.startswith("tmp_") and table_key not in DATASET_CONTRACTS_BY_TABLE_KEY
    ]
    _require(condition=not missing, message=f"Missing contracts for: {missing}")


def test_json_schema_map_matches_contracts() -> None:
    """Derived JSON Schema mapping should mirror contract definitions."""
    expected = {
        name: contract.json_schema_id
        for name, contract in DATASET_CONTRACTS.items()
        if contract.json_schema_id is not None
    }
    _require(
        condition=expected == JSON_SCHEMA_BY_DATASET_NAME,
        message="JSON Schema mapping diverged from contracts",
    )


def test_capabilities_shape() -> None:
    """Capability flags should include read-only and view indicators."""
    contract = DATASET_CONTRACTS.get("function_profile")
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
