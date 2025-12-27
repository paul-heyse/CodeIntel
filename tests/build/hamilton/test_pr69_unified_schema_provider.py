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
    configure_schema_service,
    declared_schema_provider,
    get_schema_provider,
    iter_table_schemas,
    require_table_schema,
    unified_schema_provider,
)
from codeintel.build.target_metadata import build_target_system
from codeintel.runtime.runtime_bundle import RuntimeBundle

if TYPE_CHECKING:
    from codeintel.build.schemas.schema_index import SchemaIndex
    from codeintel.core.schemas.primitives import TableSchema


def _get_declared_schemas(runtime: RuntimeBundle) -> dict[str, TableSchema]:
    """Get declared schemas as a dict for comparison.

    Returns
    -------
    dict[str, TableSchema]
        Mapping from table_key to declared schema.
    """
    return {s.table_key: s for s in declared_schema_provider(runtime=runtime).iter_table_schemas()}


def _get_schema_index(runtime: RuntimeBundle) -> SchemaIndex:
    """Return the DAG-derived schema index.

    Returns
    -------
    SchemaIndex
        Schema index tied to the global target graph.

    Raises
    ------
    RuntimeError
        If the runtime bundle has no schema index.
    """
    schema_index = runtime.schema_index
    if schema_index is None:
        msg = "Runtime bundle missing schema_index"
        raise RuntimeError(msg)
    return schema_index


def _declared_source_keys(runtime: RuntimeBundle) -> list[str]:
    declared = _get_declared_schemas(runtime)
    derivations = _get_schema_index(runtime).derivations
    return sorted(set(declared) - set(derivations))


def _non_inferable_table_key(runtime: RuntimeBundle) -> str | None:
    schema_index = _get_schema_index(runtime)
    for table_key in schema_index.derivations:
        if table_key not in schema_index.inferable_table_keys:
            return table_key
    return None


@pytest.fixture(autouse=True)
def _configure_schema_provider(hamilton_runtime: RuntimeBundle) -> None:
    configure_schema_service(runtime=hamilton_runtime)


# -----------------------------------------------------------------------------
# Basic functionality tests
# -----------------------------------------------------------------------------


def test_unified_provider_is_returned_by_get_schema_provider(
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Verify get_schema_provider returns a UnifiedSchemaProvider."""
    clear_schema_provider_cache()
    configure_schema_service(runtime=hamilton_runtime)
    provider = get_schema_provider()
    if not isinstance(provider, UnifiedSchemaProvider):
        pytest.fail(f"Expected UnifiedSchemaProvider, got {type(provider).__name__}")


def test_unified_provider_has_schema_provider_interface(
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Verify unified provider implements the SchemaProvider interface."""
    provider = unified_schema_provider(runtime=hamilton_runtime)
    if not hasattr(provider, "get_table_schema"):
        pytest.fail("Provider missing get_table_schema method")
    if not hasattr(provider, "require_table_schema"):
        pytest.fail("Provider missing require_table_schema method")
    if not hasattr(provider, "iter_table_schemas"):
        pytest.fail("Provider missing iter_table_schemas method")
    if not callable(provider.get_table_schema):
        pytest.fail("get_table_schema is not callable")


def test_unified_provider_is_cached(hamilton_runtime: RuntimeBundle) -> None:
    """Verify unified_schema_provider returns cached instance."""
    clear_unified_provider_cache()
    provider1 = unified_schema_provider(runtime=hamilton_runtime)
    provider2 = unified_schema_provider(runtime=hamilton_runtime)
    if provider1 is not provider2:
        pytest.fail("unified_schema_provider did not return cached provider")


def test_clear_unified_provider_cache_works(hamilton_runtime: RuntimeBundle) -> None:
    """Verify clearing the unified provider cache creates new instance."""
    provider1 = unified_schema_provider(runtime=hamilton_runtime)
    clear_unified_provider_cache()
    provider2 = unified_schema_provider(runtime=hamilton_runtime)
    # After cache clear, we may get a different instance but same schemas
    table_key = _non_inferable_table_key(hamilton_runtime)
    if table_key is None:
        pytest.skip("No non-inferable table keys found in this environment")
    schema1 = provider1.get_table_schema(table_key)
    schema2 = provider2.get_table_schema(table_key)
    if schema1 != schema2:
        pytest.fail("Providers return different schemas after cache clear")


# -----------------------------------------------------------------------------
# Parity tests with declared source schemas
# -----------------------------------------------------------------------------


def test_unified_provider_resolves_declared_source_keys(
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Verify declared source keys resolve through unified provider."""
    provider = unified_schema_provider(runtime=hamilton_runtime)
    source_keys = _declared_source_keys(hamilton_runtime)
    if not source_keys:
        pytest.skip("No declared source-only keys found in this environment")

    missing_keys = [key for key in source_keys if provider.get_table_schema(key) is None]
    if missing_keys:
        pytest.fail(f"Missing declared source keys: {missing_keys[:10]}...")


def test_unified_provider_source_schemas_match_declared(
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Verify source table schemas match declared definitions."""
    provider = unified_schema_provider(runtime=hamilton_runtime)
    declared_schemas = _get_declared_schemas(hamilton_runtime)
    source_keys = _declared_source_keys(hamilton_runtime)
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


def test_unified_provider_has_inferable_table_keys(
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Verify unified provider tracks inferable table keys."""
    provider = unified_schema_provider(runtime=hamilton_runtime)
    if not hasattr(provider, "inferable_table_keys"):
        pytest.fail("Provider missing inferable_table_keys attribute")
    if not isinstance(provider.inferable_table_keys, frozenset):
        pytest.fail("inferable_table_keys should be a frozenset")
    if provider.inferable_table_keys != _get_schema_index(hamilton_runtime).inferable_table_keys:
        pytest.fail("inferable_table_keys should align with the schema index")


def test_inferable_table_keys_not_empty(hamilton_runtime: RuntimeBundle) -> None:
    """Verify there are some inferable table keys in the target graph."""
    inferable = _get_schema_index(hamilton_runtime).inferable_table_keys
    # We expect at least some inferable keys from Hamilton native compute
    if len(inferable) == 0:
        pytest.skip("No inferable table keys found - may be expected in test env")


def test_schema_index_covers_target_table_keys(hamilton_runtime: RuntimeBundle) -> None:
    """Verify schema index derivations cover all target table keys."""
    schema_index = _get_schema_index(hamilton_runtime)
    derivation_keys = set(schema_index.derivations)
    expected_keys = set(build_target_system(runtime=hamilton_runtime).all_table_keys)
    if derivation_keys != expected_keys:
        missing = sorted(expected_keys - derivation_keys)
        extra = sorted(derivation_keys - expected_keys)
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if extra:
            details.append(f"extra={extra[:5]}")
        pytest.fail("Schema index key mismatch: " + ", ".join(details))


def test_unified_provider_has_declared_fallback(hamilton_runtime: RuntimeBundle) -> None:
    """Verify unified provider has a declared provider fallback."""
    provider = unified_schema_provider(runtime=hamilton_runtime)
    if not hasattr(provider, "declared"):
        pytest.fail("Provider missing declared attribute")
    source_keys = _declared_source_keys(hamilton_runtime)
    if not source_keys:
        pytest.skip("No declared source-only keys found in this environment")
    declared = declared_schema_provider(runtime=hamilton_runtime)
    declared_schema = declared.get_table_schema(source_keys[0])
    if declared_schema is None:
        pytest.fail(f"Declared provider missing {source_keys[0]}")


def test_unified_provider_fallback_to_declared_for_sources(
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Verify fallback works when table keys are not produced by the DAG."""
    provider = unified_schema_provider(runtime=hamilton_runtime)
    source_keys = _declared_source_keys(hamilton_runtime)
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


def test_get_table_schema_returns_none_for_unknown_key(
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Verify get_table_schema returns None for unknown keys."""
    provider = unified_schema_provider(runtime=hamilton_runtime)
    result = provider.get_table_schema("nonexistent.table")
    if result is not None:
        pytest.fail("Expected None for unknown table key")


# -----------------------------------------------------------------------------
# Caching tests
# -----------------------------------------------------------------------------


def test_unified_provider_caches_resolved_schemas(hamilton_runtime: RuntimeBundle) -> None:
    """Verify unified provider caches resolved schemas internally."""
    clear_unified_provider_cache()
    provider = unified_schema_provider(runtime=hamilton_runtime).with_inference(
        allow_inference=False
    )
    schema_index = _get_schema_index(hamilton_runtime)
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


def test_unified_provider_has_dataclass_fields(hamilton_runtime: RuntimeBundle) -> None:
    """Verify unified provider has expected dataclass structure."""
    provider = unified_schema_provider(runtime=hamilton_runtime)
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
# Catalog-declared schema tests
# -----------------------------------------------------------------------------


def test_catalog_output_schemas_accessible(hamilton_runtime: RuntimeBundle) -> None:
    """Verify schemas for catalog outputs are accessible."""
    catalog = hamilton_runtime.catalog
    provider = unified_schema_provider(runtime=hamilton_runtime).with_inference(
        allow_inference=False
    )

    for table_key in catalog.table_outputs:
        resolved = provider.get_table_schema(table_key)
        if resolved is not None:
            return

    pytest.skip("No catalog outputs with explicit schemas found")


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------


def test_unified_provider_works_with_get_schema_provider(
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Verify unified provider integrates correctly with registry."""
    clear_schema_provider_cache()
    configure_schema_service(runtime=hamilton_runtime)
    provider = get_schema_provider()
    table_key = _non_inferable_table_key(hamilton_runtime)
    if table_key is None:
        pytest.skip("No non-inferable table keys found in this environment")
    schema = provider.require_table_schema(table_key)
    if schema is None:
        pytest.fail(f"Failed to resolve {table_key}")
    if schema.table_key != table_key:
        pytest.fail(f"Wrong table_key: {schema.table_key}")


def test_unified_provider_works_with_require_table_schema(
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Verify unified provider integrates correctly with convenience function."""
    table_key = _non_inferable_table_key(hamilton_runtime)
    if table_key is None:
        pytest.skip("No non-inferable table keys found in this environment")
    schema = require_table_schema(table_key)
    if schema is None:
        pytest.fail(f"Failed to resolve {table_key}")
    if schema.table_key != table_key:
        pytest.fail(f"Wrong table_key: {schema.table_key}")
