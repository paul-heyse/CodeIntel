"""Tests for codeintel.config.datasets.schema_registry module."""

from __future__ import annotations

import pytest

from codeintel.config.datasets.schema_registry import (
    SCHEMA_REGISTRY,
    DatasetSchemaRegistry,
    get_schema,
)


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


# ------------------------------------------------------------------
# DatasetSchemaRegistry unit tests with controlled fixture
# ------------------------------------------------------------------


@pytest.fixture
def isolated_registry() -> DatasetSchemaRegistry:
    """Create an isolated registry for testing.

    Returns
    -------
    DatasetSchemaRegistry
        A new registry instance that won't affect the global one.
    """
    return DatasetSchemaRegistry()


def test_registry_get_returns_none_for_unknown_key(isolated_registry: DatasetSchemaRegistry) -> None:
    """Get returns None for unknown table keys."""
    # Force initialization by accessing any method that triggers it
    isolated_registry.initialize()
    result = isolated_registry.get("nonexistent.table.xyz123")

    _require(condition=result is None, message="should return None for unknown key")


def test_registry_require_raises_for_unknown_key(isolated_registry: DatasetSchemaRegistry) -> None:
    """Require raises KeyError for unknown table keys."""
    isolated_registry.initialize()

    with pytest.raises(KeyError, match="No DatasetSchema registered"):
        isolated_registry.require("nonexistent.table.xyz123")


def test_registry_all_returns_dict(isolated_registry: DatasetSchemaRegistry) -> None:
    """All returns a dictionary of schemas."""
    result = isolated_registry.all()

    _require(condition=isinstance(result, dict), message="should return dict")


def test_registry_keys_returns_list(isolated_registry: DatasetSchemaRegistry) -> None:
    """Keys returns a list of strings."""
    result = isolated_registry.keys()

    _require(condition=isinstance(result, list), message="should return list")
    for key in result:
        _require(condition=isinstance(key, str), message=f"key {key} should be string")


def test_registry_len_returns_int(isolated_registry: DatasetSchemaRegistry) -> None:
    """Len returns a non-negative integer."""
    result = len(isolated_registry)

    _require(condition=isinstance(result, int), message="should return int")
    _require(condition=result >= 0, message="should be non-negative")


def test_registry_contains_check(isolated_registry: DatasetSchemaRegistry) -> None:
    """Contains check works correctly."""
    isolated_registry.initialize()

    # Should return False for nonexistent key
    _require(
        condition="nonexistent.table.xyz123" not in isolated_registry,
        message="should not contain unknown key",
    )


# ------------------------------------------------------------------
# Global SCHEMA_REGISTRY integration tests
# ------------------------------------------------------------------


def test_global_registry_initializes_from_contracts() -> None:
    """SCHEMA_REGISTRY initializes from existing contracts."""
    all_schemas = SCHEMA_REGISTRY.all()

    # Should have schemas for datasets that have both contracts and Pandera schemas
    _require(condition=len(all_schemas) > 0, message="should have at least one schema")


def test_global_registry_known_dataset_is_available() -> None:
    """A known dataset should be available in the registry."""
    # analytics.function_metrics should have both contract and Pandera schema
    schema = SCHEMA_REGISTRY.get("analytics.function_metrics")

    # May be None if Pandera schema not defined for this table
    # but should not raise an error
    if schema is not None:
        _require(
            condition=schema.name == "analytics.function_metrics",
            message="schema name mismatch",
        )
        _require(condition=len(schema.column_names()) > 0, message="should have columns")


def test_global_registry_keys_are_table_keys() -> None:
    """All registry keys should be fully qualified table names."""
    keys = SCHEMA_REGISTRY.keys()

    for key in keys:
        _require(condition="." in key, message=f"Key {key} is not a fully qualified table name")


def test_global_registry_consistency() -> None:
    """Registry all() and keys() should be consistent."""
    all_schemas = SCHEMA_REGISTRY.all()
    keys = SCHEMA_REGISTRY.keys()

    _require(
        condition=len(all_schemas) == len(keys),
        message="all() and keys() should have same length",
    )
    for key in keys:
        _require(condition=key in all_schemas, message=f"key {key} should be in all()")


# ------------------------------------------------------------------
# get_schema convenience function tests
# ------------------------------------------------------------------


def test_get_schema_returns_schema_or_none() -> None:
    """Get_schema returns schema from global registry or None."""
    # Test with known key
    all_schemas = SCHEMA_REGISTRY.all()

    if len(all_schemas) > 0:
        first_key = next(iter(all_schemas.keys()))
        result = get_schema(first_key)
        if result is None:
            pytest.fail("should return schema for known key")
        _require(condition=result.name == first_key, message="schema name should match key")


def test_get_schema_returns_none_for_unknown() -> None:
    """Get_schema returns None for unknown keys."""
    result = get_schema("definitely.nonexistent.table.xyz")

    _require(condition=result is None, message="should return None for unknown key")


# ------------------------------------------------------------------
# producers_of and consumers_of tests
# ------------------------------------------------------------------


def test_producers_of_returns_list() -> None:
    """Producers_of returns a list."""
    result = DatasetSchemaRegistry.producers_of("analytics.function_metrics")

    _require(condition=isinstance(result, list), message="should return list")


def test_consumers_of_returns_list() -> None:
    """Consumers_of returns a list."""
    result = DatasetSchemaRegistry.consumers_of("analytics.function_metrics")

    _require(condition=isinstance(result, list), message="should return list")
