"""Tests for codeintel.config.datasets.contracts module."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from codeintel.config.datasets.contracts import (
    DatasetContract,
    RowBinding,
    get_composite_schemas,
    get_dataset_contracts,
    get_dataset_contracts_by_table_key,
    get_row_bindings,
    get_table_schemas,
)


def require(*, condition: bool, message: str) -> None:
    """Fail the current test with a descriptive message."""
    if not condition:
        pytest.fail(message)


def test_get_table_schemas_returns_dict() -> None:
    """Verify get_table_schemas returns a dictionary."""
    schemas = get_table_schemas()
    require(
        condition=isinstance(schemas, dict),
        message="get_table_schemas should return a dictionary",
    )
    require(condition=len(schemas) > 0, message="table schemas should not be empty")


def test_get_table_schemas_contains_expected_keys() -> None:
    """Verify TABLE_SCHEMAS contains known table keys."""
    schemas = get_table_schemas()
    # Check for some known table keys
    require(condition="core.goids" in schemas, message="core.goids table schema missing")
    require(
        condition="analytics.function_metrics" in schemas,
        message="analytics.function_metrics table schema missing",
    )
    require(
        condition="graph.call_graph_edges" in schemas,
        message="graph.call_graph_edges schema missing",
    )


def test_get_table_schemas_values_are_table_schema() -> None:
    """Verify TABLE_SCHEMAS values are TableSchema instances."""
    schemas = get_table_schemas()
    for key, value in schemas.items():
        # Check by class name since TableSchema may come from legacy or new module
        require(
            condition=value.__class__.__name__ == "TableSchema",
            message=f"{key} is not a TableSchema",
        )
        # Verify it has the expected attributes
        require(
            condition=hasattr(value, "schema"),
            message=f"{key} TableSchema missing schema attribute",
        )
        require(
            condition=hasattr(value, "name"),
            message=f"{key} TableSchema missing name attribute",
        )
        require(
            condition=hasattr(value, "columns"),
            message=f"{key} TableSchema missing columns attribute",
        )
        require(
            condition=hasattr(value, "fq_name"),
            message=f"{key} TableSchema missing fq_name attribute",
        )


def test_get_composite_schemas_returns_dict() -> None:
    """Verify get_composite_schemas returns a dictionary."""
    composites = get_composite_schemas()
    require(
        condition=isinstance(composites, dict),
        message="get_composite_schemas should return a dict",
    )


def test_get_dataset_contracts_returns_dict() -> None:
    """Verify get_dataset_contracts returns a dictionary."""
    contracts = get_dataset_contracts()
    require(
        condition=isinstance(contracts, dict),
        message="get_dataset_contracts should return a dict",
    )
    require(condition=len(contracts) > 0, message="dataset contracts should not be empty")


def test_get_dataset_contracts_by_table_key_returns_dict() -> None:
    """Verify get_dataset_contracts_by_table_key returns a dictionary."""
    contracts = get_dataset_contracts_by_table_key()
    require(
        condition=isinstance(contracts, dict),
        message="get_dataset_contracts_by_table_key should return a dict",
    )
    require(
        condition=len(contracts) > 0,
        message="dataset contracts by table key should not be empty",
    )


def test_dataset_contract_counts_match() -> None:
    """Verify DATASET_CONTRACTS and DATASET_CONTRACTS_BY_TABLE_KEY have same count."""
    by_name = get_dataset_contracts()
    by_key = get_dataset_contracts_by_table_key()
    require(
        condition=len(by_name) == len(by_key),
        message="contract counts by name and by table key should match",
    )


def test_row_binding_dataclass() -> None:
    """Verify RowBinding dataclass behaves correctly."""

    def dummy_serializer(_row: Mapping[str, object]) -> tuple[object, ...]:
        return ()

    binding = RowBinding(row_type=dict, to_tuple=dummy_serializer)
    require(condition=binding.row_type is dict, message="row_type should store provided type")
    require(
        condition=binding.to_tuple is dummy_serializer,
        message="to_tuple should store the serializer",
    )


def test_dataset_contract_capabilities() -> None:
    """Verify DatasetContract.capabilities method returns expected flags."""
    contracts = get_dataset_contracts()
    # Find a contract with a schema to test
    test_contract = None
    for contract in contracts.values():
        if contract.schema is not None and contract.jsonl_filename is not None:
            test_contract = contract
            break

    if test_contract is None:
        pytest.fail("expected at least one contract with schema and jsonl filename")

    caps = test_contract.capabilities()
    require(
        condition="can_export_jsonl" in caps,
        message="capabilities should include can_export_jsonl",
    )
    require(
        condition="can_export_parquet" in caps,
        message="capabilities should include can_export_parquet",
    )
    require(
        condition="has_row_binding" in caps,
        message="capabilities should include has_row_binding",
    )
    require(condition="is_view" in caps, message="capabilities should include is_view")


def test_dataset_contract_column_names() -> None:
    """Verify DatasetContract.column_names method works."""
    contracts = get_dataset_contracts()
    # Find a contract with a schema
    for contract in contracts.values():
        if contract.schema is not None:
            cols = contract.column_names()
            require(
                condition=isinstance(cols, tuple),
                message="column_names should return a tuple",
            )
            require(condition=len(cols) > 0, message="column_names should not be empty")
            # Column names should be strings
            for col in cols:
                require(condition=isinstance(col, str), message="column names must be strings")
            break
    else:
        pytest.fail("expected at least one contract with a schema")


def test_dataset_contract_has_row_binding() -> None:
    """Verify DatasetContract.has_row_binding method."""
    contracts = get_dataset_contracts()
    bindings = get_row_bindings()

    # Find a contract with a row binding
    for contract in contracts.values():
        if contract.table_key in bindings:
            require(
                condition=contract.has_row_binding(),
                message="has_row_binding should be True",
            )
            binding = contract.require_row_binding()
            require(
                condition=binding is not None,
                message="require_row_binding should return binding",
            )
            break
    else:
        pytest.fail("expected at least one contract with a row binding")


def test_dataset_contract_without_row_binding_raises() -> None:
    """Verify require_row_binding raises for contracts without bindings."""
    # Create a contract without row_binding
    contract = DatasetContract(
        table_key="test.table",
        name="test_table",
        schema=None,
        row_binding=None,
    )
    with pytest.raises(KeyError, match="has no row binding"):
        contract.require_row_binding()


def test_dataset_contract_deprecation_fields_exist() -> None:
    """Verify DatasetContract has deprecation fields."""
    # Check that the fields exist in the dataclass
    contract = DatasetContract(
        table_key="test.table",
        name="test_table",
        schema=None,
        deprecated=True,
        deprecation_message="Use new_table instead",
    )
    require(condition=contract.deprecated is True, message="deprecated flag should be True")
    require(
        condition=contract.deprecation_message == "Use new_table instead",
        message="deprecation_message should match provided text",
    )


def test_dataset_contract_deprecation_defaults() -> None:
    """Verify DatasetContract deprecation fields have correct defaults."""
    contract = DatasetContract(
        table_key="test.table",
        name="test_table",
        schema=None,
    )
    require(condition=contract.deprecated is False, message="deprecated should default to False")
    require(
        condition=contract.deprecation_message is None,
        message="deprecation_message should default to None",
    )
