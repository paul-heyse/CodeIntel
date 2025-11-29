"""Tests for the dataset contract and registry wiring."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from codeintel.storage.datasets import (
    JSON_SCHEMA_BY_DATASET_NAME,
    Dataset,
    RowBinding,
    describe_dataset,
    load_dataset_registry,
)
from codeintel.storage.gateway import open_memory_gateway


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def _stub_to_tuple(row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(row.values())


def test_json_schema_ids_attached_to_datasets() -> None:
    """Datasets loaded from DuckDB should include JSON Schema identifiers when present."""
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=False)
    try:
        registry = load_dataset_registry(gateway.con)
        names_with_schema = set(registry.datasets_with_json_schema())
        expected = set(JSON_SCHEMA_BY_DATASET_NAME.keys())
        _require(
            condition=names_with_schema == expected,
            message=f"datasets_with_json_schema mismatch: {names_with_schema} != {expected}",
        )
    finally:
        gateway.close()


def test_require_row_binding_behavior() -> None:
    """Row binding helpers should expose deterministic behavior."""
    binding = RowBinding(row_type=dict, to_tuple=_stub_to_tuple)
    dataset_with_binding = Dataset(
        table_key="dummy.table",
        name="dummy",
        schema=None,
        row_binding=binding,
    )
    _require(
        condition=dataset_with_binding.has_row_binding() is True,
        message="Expected binding presence",
    )
    _require(
        condition=dataset_with_binding.require_row_binding() is binding,
        message="require_row_binding did not return configured binding",
    )

    dataset_without_binding = Dataset(
        table_key="dummy2.table",
        name="dummy2",
        schema=None,
    )
    _require(
        condition=dataset_without_binding.has_row_binding() is False,
        message="Unexpected binding presence on dataset_without_binding",
    )
    with pytest.raises(KeyError):
        dataset_without_binding.require_row_binding()


def test_describe_dataset_shape() -> None:
    """describe_dataset should emit a JSON-friendly summary."""
    dataset = Dataset(
        table_key="analytics.function_profile",
        name="function_profile",
        schema=None,
        jsonl_filename="function_profile.jsonl",
        parquet_filename="function_profile.parquet",
        json_schema_id="function_profile",
        description="Function-level profile dataset.",
    )
    description = describe_dataset(dataset)
    _require(
        condition=description["name"] == "function_profile",
        message="Name mismatch in description",
    )
    _require(
        condition=description["table_key"] == "analytics.function_profile",
        message="Table key mismatch in description",
    )
    _require(
        condition=description["json_schema_id"] == "function_profile",
        message="json_schema_id mismatch in description",
    )
    _require(
        condition=description["has_row_binding"] is False,
        message="Unexpected binding flag in description",
    )
    _require(
        condition=description["schema_columns"] == [],
        message="Expected empty schema_columns",
    )


def test_json_schema_datasets_have_row_bindings() -> None:
    """Datasets with JSON Schemas should expose row bindings where supported."""
    allow_missing = {"data_model_fields", "data_model_relationships"}
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=False)
    try:
        registry = load_dataset_registry(gateway.con)
        for dataset_name in JSON_SCHEMA_BY_DATASET_NAME:
            if dataset_name in allow_missing:
                continue
            dataset = registry.by_name.get(dataset_name)
            _require(
                condition=dataset is not None,
                message=f"Dataset missing from registry: {dataset_name}",
            )
            if dataset is None:
                continue
            _require(
                condition=dataset.row_binding is not None,
                message=f"Row binding missing for dataset: {dataset_name}",
            )
    finally:
        gateway.close()
