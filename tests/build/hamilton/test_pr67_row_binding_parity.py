"""Tests for PR-67: Row binding migration to schema-generated models.

This module validates that schema-generated row bindings are compatible
with declared TableSchema definitions and provide stable serialization order.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.build.schemas import (
    clear_row_binding_cache,
    clear_schema_provider_cache,
    configure_schema_service,
    get_row_binding,
    get_schema_provider,
    iter_row_bindings,
)
from codeintel.core.schemas import schema_hash
from codeintel.core.schemas.row_models import GeneratedRowBinding
from codeintel.runtime.runtime_bundle import RuntimeBundle

if TYPE_CHECKING:
    from collections.abc import Mapping


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


def _expect_equal(actual: object, expected: object, label: str) -> None:
    """Check equality with clear failure message."""
    if actual != expected:
        pytest.fail(f"{label}: expected {expected!r}, got {actual!r}")


@pytest.fixture(autouse=True)
def _configure_schema_provider(hamilton_runtime: RuntimeBundle) -> None:
    configure_schema_service(runtime=hamilton_runtime)


# ---------------------------------------------------------------------------
# Basic functionality tests
# ---------------------------------------------------------------------------


def test_get_row_binding_returns_generated_binding() -> None:
    """Verify get_row_binding returns a GeneratedRowBinding."""
    clear_row_binding_cache()
    binding = get_row_binding("analytics.function_metrics")

    _require(
        condition=isinstance(binding, GeneratedRowBinding),
        message="get_row_binding should return GeneratedRowBinding",
    )
    _expect_equal(binding.table_key, "analytics.function_metrics", "table_key")


def test_get_row_binding_has_schema_hash() -> None:
    """Verify generated bindings include schema hash for cache invalidation."""
    binding = get_row_binding("core.modules")
    sha256_hex_length = 64

    _require(
        condition=len(binding.schema_hash) == sha256_hex_length,
        message=f"schema_hash should be {sha256_hex_length} chars, got {len(binding.schema_hash)}",
    )


def test_get_row_binding_is_cached() -> None:
    """Verify get_row_binding returns cached instances."""
    clear_row_binding_cache()
    binding1 = get_row_binding("core.modules")
    binding2 = get_row_binding("core.modules")

    _require(
        condition=binding1 is binding2,
        message="get_row_binding should return cached instance",
    )


def test_clear_row_binding_cache_works() -> None:
    """Verify cache clearing regenerates bindings."""
    binding1 = get_row_binding("core.modules")
    clear_row_binding_cache()
    binding2 = get_row_binding("core.modules")

    # After cache clear, we get a new (but equivalent) binding
    _require(
        condition=binding1 is not binding2,
        message="After cache clear, should get new binding instance",
    )
    _expect_equal(binding1.table_key, binding2.table_key, "table_key after cache clear")


def test_get_row_binding_raises_for_unknown_key() -> None:
    """Verify get_row_binding raises KeyError for unknown keys."""
    with pytest.raises(KeyError, match=r"nonexistent\.table"):
        get_row_binding("nonexistent.table")


def test_iter_row_bindings_returns_all() -> None:
    """Verify iter_row_bindings yields bindings for all schemas."""
    provider = get_schema_provider()
    schema_count = len(list(provider.iter_table_schemas()))

    bindings = list(iter_row_bindings())

    _expect_equal(len(bindings), schema_count, "binding count")
    for binding in bindings:
        _require(
            condition=isinstance(binding, GeneratedRowBinding),
            message=f"Expected GeneratedRowBinding, got {type(binding)}",
        )


# ---------------------------------------------------------------------------
# Schema alignment tests
# ---------------------------------------------------------------------------


def test_generated_binding_has_canonical_properties() -> None:
    """Verify GeneratedRowBinding has row_model and serializer properties."""
    binding = get_row_binding("analytics.function_metrics")

    _require(
        condition=hasattr(binding, "row_model"),
        message="GeneratedRowBinding should have row_model property",
    )
    _require(
        condition=hasattr(binding, "serializer"),
        message="GeneratedRowBinding should have serializer property",
    )
    _require(
        condition=callable(binding.serializer),
        message="serializer should be callable",
    )
    _require(
        condition=isinstance(binding.row_model, type),
        message="row_model should be a type",
    )


def test_row_model_fields_match_schema_order() -> None:
    """Verify generated row model field order matches the TableSchema column order."""
    provider = get_schema_provider()
    mismatches: list[str] = []

    for schema in provider.iter_table_schemas():
        table_key = schema.table_key
        try:
            binding = get_row_binding(table_key)
        except KeyError:
            mismatches.append(table_key)
            continue

        schema_order = list(schema.column_names())
        model_order = list(getattr(binding.row_model, "__annotations__", {}).keys())

        if schema_order != model_order:
            mismatches.append(table_key)

    if mismatches:
        pytest.fail(f"Row model field order mismatched schema for: {mismatches[:10]}")


def test_serializer_column_order_matches_schema() -> None:
    """Verify generated serializer follows schema column ordering."""
    provider = get_schema_provider()
    mismatches: list[str] = []

    for schema in provider.iter_table_schemas():
        table_key = schema.table_key
        binding = get_row_binding(table_key)
        column_names = list(schema.column_names())
        row = {name: f"value_{idx}_{name}" for idx, name in enumerate(column_names)}
        result = binding.serializer(row)
        expected = tuple(row[name] for name in column_names)
        if result != expected:
            mismatches.append(table_key)

    if mismatches:
        pytest.fail(f"Row serializer order mismatched schema for: {mismatches[:10]}")


# ---------------------------------------------------------------------------
# Serialization roundtrip tests
# ---------------------------------------------------------------------------


def _create_test_row(table_key: str) -> Mapping[str, object]:
    """Create a minimal test row for serialization testing.

    Parameters
    ----------
    table_key
        Table key to create test row for.

    Returns
    -------
    Mapping[str, object]
        Test row with default values for all columns.
    """
    provider = get_schema_provider()
    schema = provider.require_table_schema(table_key)

    row: dict[str, object] = {}
    for col in schema.columns:
        if col.type in {"INTEGER", "BIGINT", "DECIMAL(38,0)"}:
            row[col.name] = 42
        elif col.type in {"DOUBLE", "DECIMAL"}:
            row[col.name] = 3.14
        elif col.type == "BOOLEAN":
            row[col.name] = True
        elif col.type in {"VARCHAR", "JSON"}:
            row[col.name] = f"test_{col.name}"
        elif col.type in {"TIMESTAMP", "TIMESTAMPTZ"}:
            row[col.name] = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        else:
            row[col.name] = None

    return row


def test_generated_serializer_produces_tuples() -> None:
    """Verify generated serializer produces tuples of correct length."""
    provider = get_schema_provider()
    test_keys = ["analytics.function_metrics", "core.modules", "core.goids"]

    for table_key in test_keys:
        schema = provider.get_table_schema(table_key)
        if schema is None:
            continue

        binding = get_row_binding(table_key)
        test_row = _create_test_row(table_key)
        result = binding.serializer(test_row)

        _require(
            condition=isinstance(result, tuple),
            message=f"{table_key}: serializer should produce tuple",
        )
        _expect_equal(
            len(result),
            len(schema.columns),
            f"{table_key} tuple length",
        )


def test_generated_serializer_matches_legacy_output() -> None:
    """Verify serializer output follows schema order for a representative subset."""
    provider = get_schema_provider()
    test_keys = [
        "analytics.function_metrics",
        "analytics.coverage_lines",
        "core.goids",
        "graph.call_graph_edges",
    ]

    for table_key in test_keys:
        schema = provider.get_table_schema(table_key)
        if schema is None:
            continue
        binding = get_row_binding(table_key)
        test_row = _create_test_row(table_key)
        expected = tuple(test_row[name] for name in schema.column_names())
        _expect_equal(binding.serializer(test_row), expected, f"{table_key} serializer order")


# ---------------------------------------------------------------------------
# Schema provider integration tests
# ---------------------------------------------------------------------------


def test_row_binding_uses_schema_provider(hamilton_runtime: RuntimeBundle) -> None:
    """Verify row binding generation uses the schema provider."""
    clear_schema_provider_cache()
    clear_row_binding_cache()
    configure_schema_service(runtime=hamilton_runtime)

    provider = get_schema_provider()
    schema = provider.require_table_schema("analytics.function_metrics")
    binding = get_row_binding("analytics.function_metrics")

    # Binding should have same column count as schema
    schema_cols = len(schema.columns)
    binding_fields = len(getattr(binding.row_model, "__annotations__", {}))

    _expect_equal(binding_fields, schema_cols, "field count")


def test_row_binding_cache_uses_schema_hash() -> None:
    """Verify row binding includes schema hash for cache validation."""
    binding = get_row_binding("analytics.function_metrics")
    provider = get_schema_provider()
    schema = provider.require_table_schema("analytics.function_metrics")

    expected_hash = schema_hash(schema)

    _expect_equal(binding.schema_hash, expected_hash, "schema_hash")


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_generated_binding_handles_nullable_columns() -> None:
    """Verify generated bindings handle nullable columns correctly."""
    # analytics.function_metrics has nullable columns
    binding = get_row_binding("analytics.function_metrics")
    annotations = getattr(binding.row_model, "__annotations__", {})

    # All generated columns should be optional (type | None)
    for field_name, field_type in annotations.items():
        type_str = str(field_type)
        _require(
            condition="None" in type_str or "NoneType" in type_str,
            message=f"Field {field_name} should be nullable, got {field_type}",
        )


def test_generated_binding_handles_all_column_types() -> None:
    """Verify generated bindings handle all column types."""
    # Get bindings for tables with diverse column types
    test_keys = ["analytics.function_metrics", "core.modules", "graph.call_graph_edges"]

    for table_key in test_keys:
        try:
            binding = get_row_binding(table_key)
        except KeyError:
            continue

        _require(
            condition=len(getattr(binding.row_model, "__annotations__", {})) > 0,
            message=f"{table_key}: should have annotations",
        )
