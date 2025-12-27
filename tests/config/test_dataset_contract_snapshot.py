"""Snapshot test to guard against accidental dataset/table loss during refactoring."""

from __future__ import annotations

import pytest

from codeintel.build.schemas import (
    configure_schema_service,
    get_schema_provider,
    iter_contracts,
    iter_contracts_by_table_key,
    iter_row_bindings,
)
from codeintel.build.schemas.provider_unified import non_inferable_schema_provider
from codeintel.core.schemas.primitives import TableSchema
from codeintel.runtime.runtime_bundle import RuntimeBundle

EXPECTED_DATASET_CONTRACTS_COUNT = 99
EXPECTED_TABLE_SCHEMAS_COUNT = 101
EXPECTED_ROW_BINDINGS_COUNT = 99


@pytest.fixture(autouse=True)
def _configure_schema_provider(hamilton_runtime: RuntimeBundle) -> None:
    configure_schema_service(runtime=hamilton_runtime)


@pytest.fixture(scope="module")
def non_inferable_table_schemas(
    hamilton_runtime: RuntimeBundle,
) -> tuple[TableSchema, ...]:
    """Provide table schemas from the non-inferable provider.

    Returns
    -------
    tuple[TableSchema, ...]
        Non-inferable table schemas for snapshot checks.
    """
    provider = non_inferable_schema_provider(runtime=hamilton_runtime)
    return tuple(provider.iter_table_schemas())


@pytest.fixture(scope="module")
def inferable_table_keys(hamilton_runtime: RuntimeBundle) -> frozenset[str]:
    """Provide inferable table keys from the schema provider.

    Returns
    -------
    frozenset[str]
        Table keys that can be inferred from the DAG.
    """
    configure_schema_service(runtime=hamilton_runtime)
    provider = get_schema_provider()
    return getattr(provider, "inferable_table_keys", frozenset())


def test_dataset_contracts_count_snapshot() -> None:
    """Lock in the current DATASET_CONTRACTS count to detect accidental removal."""
    actual = len(list(iter_contracts()))
    if actual < EXPECTED_DATASET_CONTRACTS_COUNT:
        pytest.fail(
            f"Expected at least {EXPECTED_DATASET_CONTRACTS_COUNT} DATASET_CONTRACTS, got {actual}"
        )


def test_dataset_contracts_by_table_key_count_snapshot() -> None:
    """Lock in the current DATASET_CONTRACTS_BY_TABLE_KEY count."""
    actual = len(list(iter_contracts_by_table_key()))
    if actual < EXPECTED_DATASET_CONTRACTS_COUNT:
        pytest.fail(
            f"Expected at least {EXPECTED_DATASET_CONTRACTS_COUNT} entries in "
            f"DATASET_CONTRACTS_BY_TABLE_KEY, got {actual}"
        )


def test_table_schemas_count_snapshot(non_inferable_table_schemas: tuple[object, ...]) -> None:
    """Lock in the current TABLE_SCHEMAS count to detect accidental removal."""
    actual = len(non_inferable_table_schemas)
    if actual != EXPECTED_TABLE_SCHEMAS_COUNT:
        pytest.fail(f"Expected {EXPECTED_TABLE_SCHEMAS_COUNT} TABLE_SCHEMAS, got {actual}")


def test_row_bindings_count_snapshot() -> None:
    """Lock in the current ROW_BINDINGS_BY_TABLE_KEY count."""
    actual = len(list(iter_row_bindings()))
    if actual < EXPECTED_ROW_BINDINGS_COUNT:
        pytest.fail(
            f"Expected {EXPECTED_ROW_BINDINGS_COUNT} ROW_BINDINGS_BY_TABLE_KEY, got {actual}"
        )


def test_all_tables_have_schemas(inferable_table_keys: frozenset[str]) -> None:
    """Every table_key in DATASET_CONTRACTS should have a schema (unless it's a view)."""
    schema_provider = get_schema_provider()
    table_schemas = {s.table_key: s for s in schema_provider.iter_table_schemas()}
    missing = [
        contract.table_key
        for contract in iter_contracts()
        if not contract.is_view
        and contract.table_key not in table_schemas
        and contract.table_key not in inferable_table_keys
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
