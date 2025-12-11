"""Tests for capability registry index."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.core.plugins.registry.capability_index import (
    build_registry_index,
)
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from tests._helpers.assertions import expect_equal, expect_in, expect_true

if TYPE_CHECKING:
    from codeintel.core.plugins.registry.capability_index import (
        PluginRegistryIndex,
    )


@pytest.fixture
def sample_metadata() -> list[CorePluginMetadata]:
    """Create sample metadata for testing.

    Returns
    -------
    list[CorePluginMetadata]
        Metadata instances for test cases.
    """
    return [
        CorePluginMetadata(
            name="plugin.a",
            version="1.0.0",
            description="Plugin A",
            domain=PluginDomain.ANALYTICS,
            kind="metric",
            provides=("cap.a", "cap.shared"),
            produces_tables=("table.a",),
        ),
        CorePluginMetadata(
            name="plugin.b",
            version="1.0.0",
            description="Plugin B",
            domain=PluginDomain.GRAPH,
            kind="builder",
            provides=("cap.b",),
            requires=("cap.a",),
            produces_tables=("table.b",),
        ),
        CorePluginMetadata(
            name="plugin.c",
            version="1.0.0",
            description="Plugin C",
            domain=PluginDomain.INGEST,
            kind="builder",
            provides=("cap.c", "cap.shared"),
            produces_tables=("table.c",),
        ),
    ]


@pytest.fixture
def registry_index(sample_metadata: list[CorePluginMetadata]) -> PluginRegistryIndex:
    """Build index from sample metadata.

    Returns
    -------
    PluginRegistryIndex
        Index built from sample metadata.
    """
    return build_registry_index(sample_metadata)


class TestBuildRegistryIndex:
    """Tests for build_registry_index."""

    @staticmethod
    def test_by_name_lookup(sample_metadata: list[CorePluginMetadata]) -> None:
        """Verify by_name lookup works."""
        index = build_registry_index(sample_metadata)
        expect_in("plugin.a", index.by_name)
        expect_in("plugin.b", index.by_name)
        expect_in("plugin.c", index.by_name)
        expect_equal(index.by_name["plugin.a"].description, "Plugin A")

    @staticmethod
    def test_by_capability_lookup(sample_metadata: list[CorePluginMetadata]) -> None:
        """Verify by_capability lookup works."""
        index = build_registry_index(sample_metadata)
        expect_equal(index.by_capability["cap.a"].name, "plugin.a")
        expect_equal(index.by_capability["cap.b"].name, "plugin.b")

    @staticmethod
    def test_capability_override(sample_metadata: list[CorePluginMetadata]) -> None:
        """Verify last provider wins for shared capability."""
        index = build_registry_index(sample_metadata)
        expect_equal(index.by_capability["cap.shared"].name, "plugin.c")

    @staticmethod
    def test_by_output_table_lookup(sample_metadata: list[CorePluginMetadata]) -> None:
        """Verify by_output_table lookup works."""
        index = build_registry_index(sample_metadata)
        expect_equal(index.by_output_table["table.a"].name, "plugin.a")
        expect_equal(index.by_output_table["table.b"].name, "plugin.b")

    @staticmethod
    def test_empty_metadata() -> None:
        """Verify empty metadata produces empty index."""
        index = build_registry_index([])
        expect_equal(index.by_name, {})
        expect_equal(index.by_capability, {})
        expect_equal(index.by_output_table, {})


class TestPluginRegistryIndex:
    """Tests for PluginRegistryIndex methods."""

    @staticmethod
    def test_get_by_name_found(registry_index: PluginRegistryIndex) -> None:
        """Verify get_by_name returns metadata when found."""
        meta = registry_index.get_by_name("plugin.a")
        expect_true(meta is not None)
        expect_equal(meta.name if meta else None, "plugin.a")

    @staticmethod
    def test_get_by_name_not_found(registry_index: PluginRegistryIndex) -> None:
        """Verify get_by_name returns None when not found."""
        expect_equal(registry_index.get_by_name("unknown"), None)

    @staticmethod
    def test_get_provider_found(registry_index: PluginRegistryIndex) -> None:
        """Verify get_provider returns metadata when found."""
        meta = registry_index.get_provider("cap.a")
        expect_true(meta is not None)
        expect_equal(meta.name if meta else None, "plugin.a")

    @staticmethod
    def test_get_provider_not_found(registry_index: PluginRegistryIndex) -> None:
        """Verify get_provider returns None when not found."""
        expect_equal(registry_index.get_provider("unknown.cap"), None)

    @staticmethod
    def test_provider_lookup(registry_index: PluginRegistryIndex) -> None:
        """Verify provider_lookup returns name mapping."""
        lookup = registry_index.provider_lookup()
        expect_equal(lookup["cap.a"], "plugin.a")
        expect_equal(lookup["cap.b"], "plugin.b")

    @staticmethod
    def test_all_capabilities(registry_index: PluginRegistryIndex) -> None:
        """Verify all_capabilities returns all registered capabilities."""
        capabilities = registry_index.all_capabilities()
        expect_in("cap.a", capabilities)
        expect_in("cap.b", capabilities)
        expect_in("cap.shared", capabilities)

    @staticmethod
    def test_all_tables(registry_index: PluginRegistryIndex) -> None:
        """Verify all_tables returns all registered tables."""
        tables = registry_index.all_tables()
        expect_in("table.a", tables)
        expect_in("table.b", tables)
        expect_in("table.c", tables)
