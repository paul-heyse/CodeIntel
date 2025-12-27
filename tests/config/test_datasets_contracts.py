"""Tests for codeintel.config.datasets.contracts module."""

from __future__ import annotations

import pytest

from codeintel.build.schemas import (
    configure_schema_service,
    get_composite_schemas,
    get_schema_provider,
    iter_contracts,
    iter_contracts_by_table_key,
    iter_row_bindings,
)
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.row_models import GeneratedRowBinding, row_binding_for_table_schema
from codeintel.runtime.runtime_bundle import RuntimeBundle

_SHA256_HEX_LEN: int = 64


@pytest.fixture(autouse=True)
def _configure_schema_provider(hamilton_runtime: RuntimeBundle) -> None:
    configure_schema_service(runtime=hamilton_runtime)


def require(*, condition: bool, message: str) -> None:
    """Fail the current test with a descriptive message."""
    if not condition:
        pytest.fail(message)


def test_get_table_schemas_returns_dict() -> None:
    """Verify get_schema_provider returns table schemas."""
    schemas = {s.table_key: s for s in get_schema_provider().iter_table_schemas()}
    require(
        condition=isinstance(schemas, dict),
        message="get_schema_provider should return an iterable of table schemas",
    )
    require(condition=len(schemas) > 0, message="table schemas should not be empty")


def test_get_table_schemas_contains_expected_keys() -> None:
    """Verify schema provider contains known table keys."""
    provider = get_schema_provider()
    schemas = {s.table_key: s for s in provider.iter_table_schemas()}
    inferable = getattr(provider, "inferable_table_keys", frozenset())

    require(condition="core.goids" in schemas, message="core.goids table schema missing")
    require(
        condition="analytics.function_metrics" in schemas,
        message="analytics.function_metrics table schema missing",
    )
    require(
        condition="graph.call_graph_edges" in schemas or "graph.call_graph_edges" in inferable,
        message="graph.call_graph_edges should be inferable or declared",
    )


def test_get_table_schemas_values_are_table_schema() -> None:
    """Verify TABLE_SCHEMAS values are TableSchema instances."""
    schemas = {s.table_key: s for s in get_schema_provider().iter_table_schemas()}
    for key, value in schemas.items():
        require(
            condition=value.__class__.__name__ == "TableSchema",
            message=f"{key} is not a TableSchema",
        )

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
    """Verify iter_contracts returns an iterable of contracts."""
    contracts = {c.name: c for c in iter_contracts()}
    require(
        condition=isinstance(contracts, dict),
        message="iter_contracts should yield DatasetContracts",
    )
    require(condition=len(contracts) > 0, message="dataset contracts should not be empty")


def test_get_dataset_contracts_by_table_key_returns_dict() -> None:
    """Verify iter_contracts_by_table_key returns an iterable of key-contract pairs."""
    contracts = dict(iter_contracts_by_table_key())
    require(
        condition=isinstance(contracts, dict),
        message="iter_contracts_by_table_key should yield key-contract tuples",
    )
    require(
        condition=len(contracts) > 0,
        message="dataset contracts by table key should not be empty",
    )


def test_dataset_contract_counts_match() -> None:
    """Verify contracts by name and by table key have same count."""
    by_name = {c.name: c for c in iter_contracts()}
    by_key = dict(iter_contracts_by_table_key())
    require(
        condition=len(by_name) == len(by_key),
        message="contract counts by name and by table key should match",
    )


def test_generated_row_binding_has_provenance() -> None:
    """Verify GeneratedRowBinding includes provenance metadata."""
    schema = TableSchema(
        schema="test",
        name="example",
        columns=[Column(name="id", type="INTEGER", nullable=False)],
    )
    binding = row_binding_for_table_schema(table_schema=schema)
    require(
        condition=isinstance(binding, GeneratedRowBinding),
        message="row_binding_for_table_schema should return GeneratedRowBinding",
    )
    require(
        condition=binding.table_key == "test.example",
        message="table_key should match schema table key",
    )
    require(
        condition=len(binding.schema_hash) == _SHA256_HEX_LEN,
        message="schema_hash should be a SHA-256 hex string",
    )
    require(condition=callable(binding.serializer), message="serializer should be callable")
    require(condition=isinstance(binding.row_model, type), message="row_model should be a type")


def test_dataset_contract_capabilities() -> None:
    """Verify DatasetContract.capabilities method returns expected flags."""
    contracts = {c.name: c for c in iter_contracts()}

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
    contracts = {c.name: c for c in iter_contracts()}

    for contract in contracts.values():
        if contract.schema is not None:
            cols = contract.column_names()
            require(
                condition=isinstance(cols, tuple),
                message="column_names should return a tuple",
            )
            require(condition=len(cols) > 0, message="column_names should not be empty")

            for col in cols:
                require(condition=isinstance(col, str), message="column names must be strings")
            break
    else:
        pytest.fail("expected at least one contract with a schema")


def test_dataset_contract_has_row_binding() -> None:
    """Verify DatasetContract.has_row_binding method."""
    contracts = {c.name: c for c in iter_contracts()}
    bindings = {b.table_key: b for b in iter_row_bindings()}

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
    contract = DatasetContract(
        table_key="test.table",
        name="test_table",
        schema=None,
        row_binding=None,
    )
    with pytest.raises(KeyError, match="has no row binding"):
        contract.require_row_binding()
