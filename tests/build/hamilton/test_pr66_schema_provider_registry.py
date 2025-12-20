"""Tests for PR-66: Schema provider registry migration.

This module validates that the new schema provider registry correctly
exposes all declared schemas through the SchemaProvider interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.build.schemas import (
    clear_schema_provider_cache,
    declared_schema_provider,
    get_schema_provider,
    iter_table_schemas,
    require_table_schema,
)
from codeintel.build.target_metadata import get_target_metadata_service
from codeintel.core.schemas import schema_hash
from codeintel.core.schemas.declared import iter_declared_schemas

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import TableSchema


def _get_full_declared_schemas() -> dict[str, TableSchema]:
    """Return the full declared schema registry.

    Returns
    -------
    dict[str, TableSchema]
        Mapping from table_key to declared schema (sources + outputs).
    """
    return {schema.table_key: schema for schema in iter_declared_schemas()}


def _source_only_declared_schemas() -> dict[str, TableSchema]:
    """Return source-only declared schemas from the build provider.

    Returns
    -------
    dict[str, TableSchema]
        Mapping from table_key to source-only declared schema.
    """
    return {
        schema.table_key: schema
        for schema in declared_schema_provider().iter_table_schemas()
    }

def test_get_schema_provider_returns_valid_provider() -> None:
    """Verify get_schema_provider returns a valid SchemaProvider."""
    provider = get_schema_provider()
    # Verify it has the SchemaProvider interface methods
    if not hasattr(provider, "get_table_schema"):
        pytest.fail("Provider missing get_table_schema method")
    if not hasattr(provider, "require_table_schema"):
        pytest.fail("Provider missing require_table_schema method")
    if not hasattr(provider, "iter_table_schemas"):
        pytest.fail("Provider missing iter_table_schemas method")
    if not callable(provider.get_table_schema):
        pytest.fail("get_table_schema is not callable")
    if not callable(provider.require_table_schema):
        pytest.fail("require_table_schema is not callable")
    if not callable(provider.iter_table_schemas):
        pytest.fail("iter_table_schemas is not callable")


def test_get_schema_provider_is_cached() -> None:
    """Verify get_schema_provider returns the same instance on repeated calls."""
    clear_schema_provider_cache()
    provider1 = get_schema_provider()
    provider2 = get_schema_provider()
    if provider1 is not provider2:
        pytest.fail("get_schema_provider did not return cached provider")


def test_require_table_schema_for_known_key() -> None:
    """Verify require_table_schema returns a schema for known keys."""
    schema = require_table_schema("analytics.function_metrics")
    if schema is None:
        pytest.fail("Expected non-None schema for analytics.function_metrics")
    if schema.table_key != "analytics.function_metrics":
        pytest.fail(f"Expected table_key 'analytics.function_metrics', got '{schema.table_key}'")
    if schema.schema != "analytics":
        pytest.fail(f"Expected schema 'analytics', got '{schema.schema}'")
    if schema.name != "function_metrics":
        pytest.fail(f"Expected name 'function_metrics', got '{schema.name}'")


def test_require_table_schema_raises_for_unknown_key() -> None:
    """Verify require_table_schema raises KeyError for unknown keys."""
    with pytest.raises(KeyError, match=r"nonexistent\.table"):
        require_table_schema("nonexistent.table")


def test_iter_table_schemas_returns_all_schemas() -> None:
    """Verify iter_table_schemas returns all known schemas."""
    schemas = list(iter_table_schemas())
    if len(schemas) == 0:
        pytest.fail("iter_table_schemas returned empty list")
    # Every schema should have a valid table_key
    for schema in schemas:
        if not schema.table_key:
            pytest.fail("Found schema with empty table_key")
        if "." not in schema.table_key:
            pytest.fail(f"Invalid table_key format: {schema.table_key}")


def test_iter_table_schemas_contains_expected_keys() -> None:
    """Verify iter_table_schemas contains known critical schemas."""
    schema_keys = {schema.table_key for schema in iter_table_schemas()}
    expected_keys = {
        "analytics.function_metrics",
        "core.modules",
        "core.goids",
        "graph.call_graph_nodes",
        "graph.call_graph_edges",
    }
    missing = expected_keys - schema_keys
    if missing:
        pytest.fail(f"Missing expected schema keys: {missing}")


def test_declared_schema_provider_is_source_only() -> None:
    """Verify declared_schema_provider excludes DAG outputs."""
    provider_keys = set(_source_only_declared_schemas())
    outputs = set(get_target_metadata_service().system.all_table_keys)
    overlap = sorted(provider_keys & outputs)
    if overlap:
        pytest.fail(f"Declared provider leaked output keys: {overlap[:10]}")


def test_every_declared_key_resolves_via_provider() -> None:
    """Verify all full declared schema keys resolve through the provider."""
    provider = get_schema_provider()
    declared_schemas = _get_full_declared_schemas()
    missing_keys: list[str] = []

    for key in declared_schemas:
        schema = provider.get_table_schema(key)
        if schema is None:
            missing_keys.append(key)

    if missing_keys:
        pytest.fail(f"Missing keys in provider: {missing_keys}")


def test_provider_schemas_match_declared_schemas() -> None:
    """Verify source-only schemas match declared definitions."""
    provider = get_schema_provider()
    declared_schemas = _source_only_declared_schemas()
    mismatches: list[str] = []

    for key, declared_schema in declared_schemas.items():
        provider_schema = provider.get_table_schema(key)
        if provider_schema is None:
            mismatches.append(f"{key}: not found in provider")
        elif provider_schema != declared_schema:
            mismatches.append(f"{key}: schema mismatch")

    if mismatches:
        pytest.fail("Schema mismatches:\n" + "\n".join(mismatches))


def test_provider_schema_count_at_least_declared() -> None:
    """Verify provider has at least as many schemas as source-only declared."""
    provider_count = len(list(iter_table_schemas()))
    declared_count = len(_source_only_declared_schemas())
    # Provider may have more schemas from Hamilton inference
    if provider_count < declared_count:
        pytest.fail(f"Provider has {provider_count} schemas, declared has {declared_count}")


def test_schema_hash_is_consistent() -> None:
    """Verify schema_hash produces consistent results."""
    schema = require_table_schema("analytics.function_metrics")
    hash1 = schema_hash(schema)
    hash2 = schema_hash(schema)
    if hash1 != hash2:
        pytest.fail("schema_hash produced inconsistent results")


def test_schema_hash_is_deterministic_across_provider_calls() -> None:
    """Verify schema hashes are deterministic regardless of how schema is obtained."""
    provider = get_schema_provider()
    schema_from_require = require_table_schema("core.modules")
    schema_from_provider = provider.require_table_schema("core.modules")

    hash_from_require = schema_hash(schema_from_require)
    hash_from_provider = schema_hash(schema_from_provider)

    if hash_from_require != hash_from_provider:
        pytest.fail("Schema hashes differ depending on how schema was obtained")


def test_all_provider_schemas_are_hashable() -> None:
    """Verify all schemas from the provider can be hashed."""
    sha256_hex_length = 64

    for schema in iter_table_schemas():
        hash_val = schema_hash(schema)
        if not hash_val:
            pytest.fail(f"Empty hash for schema {schema.table_key}")
        if len(hash_val) != sha256_hex_length:
            pytest.fail(
                f"Invalid hash length for {schema.table_key}: "
                f"expected {sha256_hex_length}, got {len(hash_val)}"
            )


def test_clear_cache_allows_fresh_provider() -> None:
    """Verify cache clearing works correctly."""
    provider1 = get_schema_provider()
    clear_schema_provider_cache()
    provider2 = get_schema_provider()
    # After cache clear, we get a new (but equivalent) provider
    # The providers should be functionally equivalent
    schema1 = provider1.get_table_schema("core.modules")
    schema2 = provider2.get_table_schema("core.modules")
    if schema1 != schema2:
        pytest.fail("Providers return different schemas after cache clear")
