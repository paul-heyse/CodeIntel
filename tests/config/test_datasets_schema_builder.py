"""Tests for codeintel.config.datasets.schema_builder module."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import pytest
from pandera import Column, DataFrameSchema

from codeintel.config.datasets.contracts import DatasetContract, RowBinding
from codeintel.config.datasets.primitives import Column as DuckDBColumn
from codeintel.config.datasets.primitives import CompositeSchema, TableSchema
from codeintel.config.datasets.schema import DatasetSchema
from codeintel.config.datasets.schema_builder import (
    build_all_schemas,
    build_dataset_schema,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


def test_build_creates_dataset_schema() -> None:
    """Create DatasetSchema from contract and Pandera schema."""
    pandera_schema = DataFrameSchema(
        {
            "repo": Column(str),
            "commit": Column(str),
        }
    )

    contract = DatasetContract(
        table_key="test.example",
        name="example",
        schema=None,
        description="Test dataset",
        owner="analytics",
        family="test",
        freshness_sla="daily",
        retention_policy="90d",
        upstream_dependencies=("core.goids",),
        tags=frozenset({"test"}),
    )

    result = build_dataset_schema(contract, pandera_schema)

    _require(condition=isinstance(result, DatasetSchema), message="should return DatasetSchema")
    _require(condition=result.name == "test.example", message="name mismatch")
    _require(condition=result.pandera_schema is pandera_schema, message="pandera_schema mismatch")


def test_build_copies_metadata_from_contract() -> None:
    """Metadata is copied from contract to DatasetSchema."""
    pandera_schema = DataFrameSchema({"col": Column(str)})

    contract = DatasetContract(
        table_key="test.example",
        name="example",
        schema=None,
        description="My description",
        owner="my_team",
        family="analytics",
        freshness_sla="hourly",
        retention_policy="30d",
        upstream_dependencies=("dep1", "dep2"),
        tags=frozenset({"production", "critical"}),
        deprecated=True,
        deprecation_message="Use new_table instead",
    )

    result = build_dataset_schema(contract, pandera_schema)

    _require(
        condition=result.metadata.description == "My description", message="description mismatch"
    )
    _require(condition=result.metadata.owner == "my_team", message="owner mismatch")
    _require(condition=result.metadata.family == "analytics", message="family mismatch")
    _require(condition=result.metadata.freshness_sla == "hourly", message="freshness_sla mismatch")
    _require(
        condition=result.metadata.retention_policy == "30d", message="retention_policy mismatch"
    )
    _require(
        condition=result.metadata.upstream_dependencies == ("dep1", "dep2"),
        message="upstream_dependencies mismatch",
    )
    _require(
        condition=result.metadata.tags == frozenset({"production", "critical"}),
        message="tags mismatch",
    )
    _require(condition=result.metadata.deprecated is True, message="deprecated mismatch")
    _require(
        condition=result.metadata.deprecation_message == "Use new_table instead",
        message="deprecation_message mismatch",
    )


def test_build_includes_ddl_schema_if_available() -> None:
    """DDL schema is included if contract has one."""
    pandera_schema = DataFrameSchema({"col": Column(str)})

    ddl_schema = TableSchema(
        schema="test",
        name="example",
        columns=[DuckDBColumn(name="col", type="VARCHAR")],
    )

    contract = DatasetContract(
        table_key="test.example",
        name="example",
        schema=ddl_schema,
    )

    result = build_dataset_schema(contract, pandera_schema)

    _require(condition=result.ddl_schema is ddl_schema, message="ddl_schema should be included")


def test_build_includes_row_model_from_binding() -> None:
    """Row model is extracted from row binding if available."""

    class MyRow(TypedDict):
        col: str

    def to_tuple(row: Mapping[str, object]) -> tuple[object, ...]:
        return (row["col"],)

    pandera_schema = DataFrameSchema({"col": Column(str)})
    row_binding = RowBinding(row_type=MyRow, to_tuple=to_tuple)

    contract = DatasetContract(
        table_key="test.example",
        name="example",
        schema=None,
        row_binding=row_binding,
    )

    result = build_dataset_schema(contract, pandera_schema)

    _require(condition=result.row_model is MyRow, message="row_model should be included")


def test_build_includes_composition_if_available() -> None:
    """Composition metadata is included if contract has it."""
    pandera_schema = DataFrameSchema({"col": Column(str)})

    composition = CompositeSchema(
        composed_of=("source1", "source2"),
        shared_fragments=(),
        additional_columns=(),
        column_mappings={},
        excluded_columns=frozenset(),
    )

    contract = DatasetContract(
        table_key="test.example",
        name="example",
        schema=None,
        composition=composition,
    )

    result = build_dataset_schema(contract, pandera_schema)

    _require(condition=result.composition is composition, message="composition should be included")


def test_build_all_returns_dict_of_schemas() -> None:
    """Returns a dictionary of DatasetSchema instances."""
    result = build_all_schemas()

    _require(condition=isinstance(result, dict), message="should return dict")
    for key, schema in result.items():
        _require(condition=isinstance(key, str), message=f"key {key} should be string")
        _require(
            condition=isinstance(schema, DatasetSchema),
            message=f"value for {key} should be DatasetSchema",
        )


def test_build_all_keys_are_table_keys() -> None:
    """All keys are fully qualified table names."""
    result = build_all_schemas()

    for key in result:
        _require(condition="." in key, message=f"Key {key} is not a fully qualified table name")


def test_build_all_schema_names_match_keys() -> None:
    """Each schema's name matches its registry key."""
    result = build_all_schemas()

    for key, schema in result.items():
        _require(condition=schema.name == key, message=f"schema name {schema.name} != key {key}")


def test_build_all_includes_known_datasets() -> None:
    """Known datasets with Pandera schemas are included."""
    result = build_all_schemas()

    if "analytics.function_metrics" in result:
        schema = result["analytics.function_metrics"]
        _require(condition=schema.name == "analytics.function_metrics", message="name mismatch")
        _require(condition=len(schema.column_names()) > 0, message="should have columns")


def test_build_all_only_includes_valid_schemas() -> None:
    """All returned schemas are valid DatasetSchema instances."""
    result = build_all_schemas()

    for table_key, schema in result.items():
        _require(
            condition=isinstance(schema, DatasetSchema),
            message=f"{table_key} should be a DatasetSchema",
        )
        _require(
            condition=schema.name == table_key,
            message=f"schema name {schema.name} should match key {table_key}",
        )

        _require(
            condition=len(schema.column_names()) > 0,
            message=f"{table_key} should have columns",
        )


def test_build_all_schemas_have_valid_structure() -> None:
    """All schemas have valid structure."""
    result = build_all_schemas()

    for schema in result.values():
        _require(condition=bool(schema.name), message="schema should have a name")

        columns = schema.column_names()
        _require(condition=isinstance(columns, tuple), message="column_names should return tuple")

        _require(condition=schema.metadata is not None, message="schema should have metadata")
