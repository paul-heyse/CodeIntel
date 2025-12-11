"""Tests for all_metadata registry."""

from __future__ import annotations

from codeintel.core.plugins.registry.all_metadata import (
    ALL_PLUGIN_METADATA,
    get_global_registry_index,
    get_provider_lookup,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_not_none,
    expect_true,
)


class TestAllPluginMetadata:
    """Tests for ALL_PLUGIN_METADATA collection."""

    @staticmethod
    def test_contains_spine_plugins() -> None:
        """Verify spine plugins are in the collection."""
        names = {meta.name for meta in ALL_PLUGIN_METADATA}
        expect_in("analytics.function_metrics", names)
        expect_in("graphs.callgraph", names)
        expect_in("ingest.scip_python", names)

    @staticmethod
    def test_all_metadata_has_required_fields() -> None:
        """Verify all metadata has required fields."""
        for meta in ALL_PLUGIN_METADATA:
            expect_true(bool(meta.name))
            expect_true(bool(meta.version))
            expect_true(bool(meta.description))
            expect_true(bool(meta.domain))
            expect_true(bool(meta.kind))

    @staticmethod
    def test_no_duplicate_names() -> None:
        """Verify no duplicate plugin names."""
        names = [meta.name for meta in ALL_PLUGIN_METADATA]
        expect_equal(len(names), len(set(names)))


class TestGlobalRegistryIndex:
    """Tests for get_global_registry_index."""

    @staticmethod
    def test_index_contains_all_plugins() -> None:
        """Verify index contains all plugins."""
        index = get_global_registry_index()
        for meta in ALL_PLUGIN_METADATA:
            expect_is_not_none(index.get_by_name(meta.name))

    @staticmethod
    def test_capabilities_are_indexed() -> None:
        """Verify capabilities are properly indexed."""
        index = get_global_registry_index()
        provider = index.get_provider("analytics.function_metrics")
        expect_is_not_none(provider)
        expect_equal(provider.name if provider else "", "analytics.function_metrics")

    @staticmethod
    def test_tables_are_indexed() -> None:
        """Verify tables are properly indexed."""
        index = get_global_registry_index()
        producer = index.get_producer("graph.call_graph_edges")
        expect_is_not_none(producer)
        expect_equal(producer.name if producer else "", "graphs.callgraph")


class TestProviderLookup:
    """Tests for get_provider_lookup."""

    @staticmethod
    def test_returns_mapping() -> None:
        """Verify provider lookup returns capability → name mapping."""
        lookup = get_provider_lookup()
        expect_true(isinstance(lookup, dict))
        expect_in("analytics.function_metrics", lookup)
        expect_equal(lookup.get("analytics.function_metrics"), "analytics.function_metrics")
