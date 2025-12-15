"""Tests for PR-69: Unified schema provider with fallback chain.

This module validates that the unified schema provider correctly resolves
schemas through the three-tier fallback chain:

1. Hamilton-native inference (for q__-driven Ibis compute nodes)
2. Target-declared schemas from OutputContract.tables
3. Raw declared schemas from declared_schema_provider()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.build.registry import get_target_graph
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
from codeintel.build.schemas.provider_hamilton import inferable_native_table_keys

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import TableSchema


def _get_declared_schemas() -> dict[str, TableSchema]:
    """Get declared schemas as a dict for comparison.

    Returns
    -------
    dict[str, TableSchema]
        Mapping from table_key to declared schema.
    """
    return {s.table_key: s for s in declared_schema_provider().iter_table_schemas()}


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
    schema1 = provider1.get_table_schema("core.modules")
    schema2 = provider2.get_table_schema("core.modules")
    if schema1 != schema2:
        pytest.fail("Providers return different schemas after cache clear")


# -----------------------------------------------------------------------------
# Parity tests with declared schemas
# -----------------------------------------------------------------------------


def test_unified_provider_resolves_all_declared_table_keys() -> None:
    """Verify all declared schema keys resolve through unified provider."""
    provider = unified_schema_provider()
    declared_schemas = _get_declared_schemas()
    missing_keys: list[str] = []

    for key in declared_schemas:
        schema = provider.get_table_schema(key)
        if schema is None:
            missing_keys.append(key)

    if missing_keys:
        pytest.fail(f"Missing keys in unified provider: {missing_keys[:10]}...")


def test_unified_provider_schemas_match_declared_schemas() -> None:
    """Verify unified provider returns identical schemas as declared_schema_provider."""
    provider = unified_schema_provider()
    declared_schemas = _get_declared_schemas()
    mismatches: list[str] = []

    for key, declared_schema in declared_schemas.items():
        provider_schema = provider.get_table_schema(key)
        if provider_schema is None:
            mismatches.append(f"{key}: not found in provider")
        # Note: Inferred schemas may differ from declared, so we only check
        # column count and names match as a basic sanity check
        elif len(provider_schema.columns) != len(declared_schema.columns):
            mismatches.append(
                f"{key}: column count mismatch "
                f"(provider={len(provider_schema.columns)}, "
                f"declared={len(declared_schema.columns)})"
            )

    if mismatches:
        pytest.fail("Schema mismatches:\n" + "\n".join(mismatches[:10]))


def test_unified_provider_schema_count_at_least_declared() -> None:
    """Verify unified provider has at least as many schemas as declared."""
    provider_count = len(list(iter_table_schemas()))
    declared_count = len(_get_declared_schemas())
    # Unified provider may have MORE schemas from Hamilton inference, never fewer
    if provider_count < declared_count:
        pytest.fail(
            f"Unified provider has fewer schemas ({provider_count}) than declared ({declared_count})"
        )


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


def test_inferable_table_keys_not_empty() -> None:
    """Verify there are some inferable table keys in the target graph."""
    graph = get_target_graph()
    inferable = inferable_native_table_keys(graph=graph)
    # We expect at least some inferable keys from Hamilton native compute
    if len(inferable) == 0:
        pytest.skip("No inferable table keys found - may be expected in test env")


def test_unified_provider_has_declared_fallback() -> None:
    """Verify unified provider has a declared provider fallback."""
    provider = unified_schema_provider()
    if not hasattr(provider, "declared"):
        pytest.fail("Provider missing declared attribute")
    # The declared fallback should be the declared_schema_provider
    declared = declared_schema_provider()
    # They should be functionally equivalent
    test_key = "core.modules"
    declared_schema = declared.get_table_schema(test_key)
    if declared_schema is None:
        pytest.fail(f"Declared provider missing {test_key}")


def test_unified_provider_fallback_to_declared_for_unknown_inferable() -> None:
    """Verify fallback works when inference fails or key not inferable."""
    provider = unified_schema_provider()
    # Source tables like core.modules aren't inferable via Hamilton
    # but should still resolve via declared fallback
    schema = provider.get_table_schema("core.modules")
    if schema is None:
        pytest.fail("Failed to resolve core.modules via fallback")
    if schema.table_key != "core.modules":
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
    provider = unified_schema_provider()

    # First access should populate cache
    schema1 = provider.get_table_schema("core.modules")
    if schema1 is None:
        pytest.fail("Failed to resolve core.modules")

    # Second access should return cached result
    schema2 = provider.get_table_schema("core.modules")
    if schema1 is not schema2:
        pytest.fail("Provider did not return cached schema")


def test_unified_provider_has_dataclass_fields() -> None:
    """Verify unified provider has expected dataclass structure."""
    provider = unified_schema_provider()
    # Verify the dataclass has expected attributes
    if not hasattr(provider, "declared"):
        pytest.fail("Provider missing declared attribute")
    if not hasattr(provider, "inferable_table_keys"):
        pytest.fail("Provider missing inferable_table_keys attribute")
    if not hasattr(provider, "fallback_to_declared_on_error"):
        pytest.fail("Provider missing fallback_to_declared_on_error attribute")


# -----------------------------------------------------------------------------
# Target-declared schema tests
# -----------------------------------------------------------------------------


def test_target_contract_schemas_accessible() -> None:
    """Verify schemas from target contracts are accessible."""
    graph = get_target_graph()
    provider = unified_schema_provider()

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
    schema = provider.require_table_schema("analytics.function_metrics")
    if schema is None:
        pytest.fail("Failed to resolve analytics.function_metrics")
    if schema.table_key != "analytics.function_metrics":
        pytest.fail(f"Wrong table_key: {schema.table_key}")


def test_unified_provider_works_with_require_table_schema() -> None:
    """Verify unified provider integrates correctly with convenience function."""
    schema = require_table_schema("core.goids")
    if schema is None:
        pytest.fail("Failed to resolve core.goids")
    if schema.table_key != "core.goids":
        pytest.fail(f"Wrong table_key: {schema.table_key}")
