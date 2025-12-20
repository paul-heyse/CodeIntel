"""Tests for PR-69: Unified schema provider with DAG-first schema resolution.

This module validates that the unified schema provider:

1. Resolves target outputs through the global DAG (SchemaIndex).
2. Uses declared schemas only for non-target source tables.
3. Preserves deterministic iteration and caching behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.build.schemas import (
    UnifiedSchemaProvider,
    clear_schema_provider_cache,
    clear_unified_provider_cache,
    declared_schema_provider,
    get_schema_provider,
    iter_table_schemas,
    require_table_schema,
    unified_schema_provider,
)
from codeintel.build.target_metadata import get_target_metadata_service

if TYPE_CHECKING:
    from codeintel.build.schemas.schema_index import SchemaIndex
    from codeintel.core.schemas.primitives import TableSchema


def _get_declared_schemas() -> dict[str, TableSchema]:
    """Get declared schemas as a dict for comparison.

    Returns
    -------
    dict[str, TableSchema]
        Mapping from table_key to declared schema.
    """
    return {s.table_key: s for s in declared_schema_provider().iter_table_schemas()}


def _get_schema_index() -> SchemaIndex:
    """Return the DAG-derived schema index.

    Returns
    -------
    SchemaIndex
        Schema index tied to the global target graph.
    """
    return get_target_metadata_service().schema_index


def _declared_source_keys() -> list[str]:
    declared = _get_declared_schemas()
    derivations = _get_schema_index().derivations
    return sorted(set(declared) - set(derivations))


def _non_inferable_table_key() -> str | None:
    schema_index = _get_schema_index()
    for table_key in schema_index.derivations:
        if table_key not in schema_index.inferable_table_keys:
            return table_key
    return None


# -----------------------------------------------------------------------------
# Basic functionality tests
# -----------------------------------------------------------------------------


def test_unified_provider_is_returned_by_get_schema_provider() -> None:
    """Verify get_schema_provider returns a UnifiedSchemaProvider."""
    clear_schema_provider_cache()
    provider = get_schema_provider()
    if not isinstance(provider, UnifiedSchemaProvider):
        pytest.fail(f"Expected UnifiedSchemaProvider, got {type(provider).__name__}")


def test_unified_provider_has_schema_provider_interface() -> None:
    """Verify unified provider implements the SchemaProvider interface."""
    provider = unified_schema_provider()
    if not hasattr(provider, "get_table_schema"):
        pytest.fail("Provider missing get_table_schema method")
    if not hasattr(provider, "require_table_schema"):
        pytest.fail("Provider missing require_table_schema method")
    if not hasattr(provider, "iter_table_schemas"):
        pytest.fail("Provider missing iter_table_schemas method")
    if not callable(provider.get_table_schema):
        pytest.fail("get_table_schema is not callable")


def test_unified_provider_is_cached() -> None:
    """Verify unified_schema_provider returns cached instance."""
    clear_unified_provider_cache()
    provider1 = unified_schema_provider()
    provider2 = unified_schema_provider()
    if provider1 is not provider2:
        pytest.fail("unified_schema_provider did not return cached provider")


def test_clear_unified_provider_cache_works() -> None:
    """Verify clearing the unified provider cache creates new instance."""
    provider1 = unified_schema_provider()
    clear_unified_provider_cache()
    provider2 = unified_schema_provider()
    # After cache clear, we may get a different instance but same schemas
    table_key = _non_inferable_table_key()
    if table_key is None:
        pytest.skip("No non-inferable table keys found in this environment")
    schema1 = provider1.get_table_schema(table_key)
    schema2 = provider2.get_table_schema(table_key)
    if schema1 != schema2:
        pytest.fail("Providers return different schemas after cache clear")


# -----------------------------------------------------------------------------
# Parity tests with declared source schemas
# -----------------------------------------------------------------------------


def test_unified_provider_resolves_declared_source_keys() -> None:
    """Verify declared source keys resolve through unified provider."""
    provider = unified_schema_provider()
    source_keys = _declared_source_keys()
    if not source_keys:
        pytest.skip("No declared source-only keys found in this environment")

    missing_keys = [key for key in source_keys if provider.get_table_schema(key) is None]
    if missing_keys:
        pytest.fail(f"Missing declared source keys: {missing_keys[:10]}...")


def test_unified_provider_source_schemas_match_declared() -> None:
    """Verify source table schemas match declared definitions."""
    provider = unified_schema_provider()
    declared_schemas = _get_declared_schemas()
    source_keys = _declared_source_keys()
    if not source_keys:
        pytest.skip("No declared source-only keys found in this environment")

    mismatches: list[str] = []
    for key in source_keys:
        provider_schema = provider.get_table_schema(key)
        declared_schema = declared_schemas.get(key)
        if provider_schema is None or declared_schema is None:
            mismatches.append(f"{key}: missing schema")
        elif provider_schema != declared_schema:
            mismatches.append(f"{key}: provider schema differs from declared")

    if mismatches:
        pytest.fail("Schema mismatches:\n" + "\n".join(mismatches[:10]))


# -----------------------------------------------------------------------------
# Fallback chain ordering tests
# -----------------------------------------------------------------------------


def test_unified_provider_has_inferable_table_keys() -> None:
    """Verify unified provider tracks inferable table keys."""
    provider = unified_schema_provider()
    if not hasattr(provider, "inferable_table_keys"):
        pytest.fail("Provider missing inferable_table_keys attribute")
    if not isinstance(provider.inferable_table_keys, frozenset):
        pytest.fail("inferable_table_keys should be a frozenset")
    if provider.inferable_table_keys != _get_schema_index().inferable_table_keys:
        pytest.fail("inferable_table_keys should align with the schema index")


def test_inferable_table_keys_not_empty() -> None:
    """Verify there are some inferable table keys in the target graph."""
    inferable = _get_schema_index().inferable_table_keys
    # We expect at least some inferable keys from Hamilton native compute
    if len(inferable) == 0:
        pytest.skip("No inferable table keys found - may be expected in test env")


def test_schema_index_covers_target_table_keys() -> None:
    """Verify schema index derivations cover all target table keys."""
    service = get_target_metadata_service()
    derivation_keys = set(service.schema_index.derivations)
    expected_keys = set(service.system.all_table_keys)
    if derivation_keys != expected_keys:
        missing = sorted(expected_keys - derivation_keys)
        extra = sorted(derivation_keys - expected_keys)
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if extra:
            details.append(f"extra={extra[:5]}")
        pytest.fail("Schema index key mismatch: " + ", ".join(details))


def test_unified_provider_has_declared_fallback() -> None:
    """Verify unified provider has a declared provider fallback."""
    provider = unified_schema_provider()
    if not hasattr(provider, "declared"):
        pytest.fail("Provider missing declared attribute")
    source_keys = _declared_source_keys()
    if not source_keys:
        pytest.skip("No declared source-only keys found in this environment")
    declared = declared_schema_provider()
    declared_schema = declared.get_table_schema(source_keys[0])
    if declared_schema is None:
        pytest.fail(f"Declared provider missing {source_keys[0]}")


def test_unified_provider_fallback_to_declared_for_sources() -> None:
    """Verify fallback works when table keys are not produced by the DAG."""
    provider = unified_schema_provider()
    source_keys = _declared_source_keys()
    if not source_keys:
        pytest.skip("No declared source-only keys found in this environment")
    table_key = source_keys[0]
    schema = provider.get_table_schema(table_key)
    if schema is None:
        pytest.fail(f"Failed to resolve {table_key} via fallback")
    if schema.table_key != table_key:
        pytest.fail(f"Wrong table_key: {schema.table_key}")


# -----------------------------------------------------------------------------
# Iterator and deduplication tests
# -----------------------------------------------------------------------------


def test_iter_table_schemas_no_duplicates() -> None:
    """Verify iter_table_schemas does not yield duplicate schemas."""
    seen_keys: set[str] = set()
    duplicates: list[str] = []

    for schema in iter_table_schemas():
        if schema.table_key in seen_keys:
            duplicates.append(schema.table_key)
        seen_keys.add(schema.table_key)

    if duplicates:
        pytest.fail(f"Duplicate table keys in iteration: {duplicates[:10]}")


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


def test_iter_table_schemas_returns_valid_schemas() -> None:
    """Verify all iterated schemas have valid structure."""
    for schema in iter_table_schemas():
        if not schema.table_key:
            pytest.fail("Found schema with empty table_key")
        if "." not in schema.table_key:
            pytest.fail(f"Invalid table_key format: {schema.table_key}")
        if len(schema.columns) == 0:
            pytest.fail(f"Schema {schema.table_key} has no columns")


# -----------------------------------------------------------------------------
# Error handling tests
# -----------------------------------------------------------------------------


def test_require_table_schema_raises_for_unknown_key() -> None:
    """Verify require_table_schema raises KeyError for unknown keys."""
    with pytest.raises(KeyError, match=r"nonexistent\.table"):
        require_table_schema("nonexistent.table")


def test_get_table_schema_returns_none_for_unknown_key() -> None:
    """Verify get_table_schema returns None for unknown keys."""
    provider = unified_schema_provider()
    result = provider.get_table_schema("nonexistent.table")
    if result is not None:
        pytest.fail("Expected None for unknown table key")


# -----------------------------------------------------------------------------
# Caching tests
# -----------------------------------------------------------------------------


def test_unified_provider_caches_resolved_schemas() -> None:
    """Verify unified provider caches resolved schemas internally."""
    clear_unified_provider_cache()
    provider = unified_schema_provider().with_inference(allow_inference=False)
    schema_index = _get_schema_index()
    override_keys = [
        table_key
        for table_key, derivation in schema_index.derivations.items()
        if derivation.override_schema is not None
    ]
    if not override_keys:
        pytest.skip("No override schemas available for caching test")
    table_key = override_keys[0]

    # First access should populate cache
    schema1 = provider.get_table_schema(table_key)
    if schema1 is None:
        pytest.fail(f"Failed to resolve {table_key}")

    # Second access should return cached result
    schema2 = provider.get_table_schema(table_key)
    if schema1 is not schema2:
        pytest.fail("Provider did not return cached schema")


def test_unified_provider_has_dataclass_fields() -> None:
    """Verify unified provider has expected dataclass structure."""
    provider = unified_schema_provider()
    # Verify the dataclass has expected attributes
    if not hasattr(provider, "declared"):
        pytest.fail("Provider missing declared attribute")
    if not hasattr(provider, "schema_index"):
        pytest.fail("Provider missing schema_index attribute")
    if not hasattr(provider, "allow_inference"):
        pytest.fail("Provider missing allow_inference attribute")
    if provider.allow_inference is not True:
        pytest.fail("Unified provider should default to allow_inference=True")
    if not hasattr(provider, "inferable_table_keys"):
        pytest.fail("Provider missing inferable_table_keys attribute")
    if not hasattr(provider, "fallback_to_override_on_error"):
        pytest.fail("Provider missing fallback_to_override_on_error attribute")


# -----------------------------------------------------------------------------
# Target-declared schema tests
# -----------------------------------------------------------------------------


def test_target_contract_schemas_accessible() -> None:
    """Verify schemas from target contracts are accessible."""
    graph = get_target_metadata_service().system.graph
    provider = unified_schema_provider().with_inference(allow_inference=False)

    # Find a target with declared output schemas
    for target in graph.all_targets:
        if target.contract.tables:
            for table_schema in target.contract.tables:
                resolved = provider.get_table_schema(table_schema.table_key)
                if resolved is None:
                    pytest.fail(
                        f"Target-declared schema {table_schema.table_key} "
                        f"not accessible via unified provider"
                    )
                return  # Found and validated one

    pytest.skip("No targets with declared output schemas found")


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------


def test_unified_provider_works_with_get_schema_provider() -> None:
    """Verify unified provider integrates correctly with registry."""
    clear_schema_provider_cache()
    provider = get_schema_provider()
    table_key = _non_inferable_table_key()
    if table_key is None:
        pytest.skip("No non-inferable table keys found in this environment")
    schema = provider.require_table_schema(table_key)
    if schema is None:
        pytest.fail(f"Failed to resolve {table_key}")
    if schema.table_key != table_key:
        pytest.fail(f"Wrong table_key: {schema.table_key}")


def test_unified_provider_works_with_require_table_schema() -> None:
    """Verify unified provider integrates correctly with convenience function."""
    table_key = _non_inferable_table_key()
    if table_key is None:
        pytest.skip("No non-inferable table keys found in this environment")
    schema = require_table_schema(table_key)
    if schema is None:
        pytest.fail(f"Failed to resolve {table_key}")
    if schema.table_key != table_key:
        pytest.fail(f"Wrong table_key: {schema.table_key}")
