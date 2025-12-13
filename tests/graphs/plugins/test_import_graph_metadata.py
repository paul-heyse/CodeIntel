"""Tests for ImportGraphPlugin metadata and options integration."""

from __future__ import annotations

from typing import Any

from codeintel.build.plugins.graphs.builders.import_graph import (
    IMPORT_GRAPH_METADATA,
    ImportGraphPlugin,
)
from codeintel.build.plugins.graphs.builders.import_graph_options import ImportGraphOptions
from codeintel.core.plugins.execution.options import ConfigSource, PluginOptionsResolver
from codeintel.core.plugins.types.metadata import PluginDomain
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_not_none,
    expect_true,
)


class DictConfigSource(ConfigSource):
    """Test config source backed by a dict."""

    def __init__(self, options: dict[str, dict[str, Any]]) -> None:
        self._options = options

    def get_plugin_options(self, plugin_name: str) -> dict[str, Any] | None:
        """Return configured options for the given plugin name.

        Returns
        -------
        dict[str, Any] | None
            Raw options mapping when configured.
        """
        return self._options.get(plugin_name)


class TestImportGraphMetadata:
    """Tests for IMPORT_GRAPH_METADATA constant."""

    @staticmethod
    def test_metadata_identity() -> None:
        """Verify metadata identity fields."""
        expect_equal(IMPORT_GRAPH_METADATA.name, "graphs.import_graph")
        expect_equal(IMPORT_GRAPH_METADATA.domain, PluginDomain.GRAPH)
        expect_equal(IMPORT_GRAPH_METADATA.kind, "builder")
        expect_equal(IMPORT_GRAPH_METADATA.stage, "edges")

    @staticmethod
    def test_metadata_capabilities() -> None:
        """Verify provides/requires are set."""
        expect_in("graph.import_graph", IMPORT_GRAPH_METADATA.provides)
        expect_in("core.modules", IMPORT_GRAPH_METADATA.requires)

    @staticmethod
    def test_metadata_tables() -> None:
        """Verify tables are set."""
        expect_in("graph.import_modules", IMPORT_GRAPH_METADATA.produces_tables)
        expect_in("graph.import_graph_edges", IMPORT_GRAPH_METADATA.produces_tables)
        expect_in("core.modules", IMPORT_GRAPH_METADATA.consumes_tables)

    @staticmethod
    def test_metadata_is_scope_aware() -> None:
        """Verify plugin is marked as scope-aware."""
        expect_true(IMPORT_GRAPH_METADATA.scope_aware)

    @staticmethod
    def test_metadata_extra_graph_kinds() -> None:
        """Verify extra.graph_kinds is set."""
        expect_equal(IMPORT_GRAPH_METADATA.extra.get("graph_kinds"), ("import_graph",))


class TestImportGraphPluginOptionsIntegration:
    """Tests for options resolution integration."""

    @staticmethod
    def test_default_options_without_resolver() -> None:
        """Verify default options when no resolver provided."""
        plugin = ImportGraphPlugin()
        opts = plugin.resolve_options()
        expect_true(opts.include_third_party)
        expect_true(not opts.include_stdlib)

    @staticmethod
    def test_options_with_config() -> None:
        """Verify config source overrides defaults."""
        source = DictConfigSource(
            {
                "graphs.import_graph": {
                    "include_stdlib": True,
                    "resolve_dynamic": True,
                    "scope_paths": ["src/"],
                },
            }
        )
        resolver = PluginOptionsResolver(source)
        plugin = ImportGraphPlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        expect_true(opts.include_stdlib)
        expect_true(opts.resolve_dynamic)
        expect_equal(opts.scope_paths, ["src/"])


class TestImportGraphPluginMetadataProperty:
    """Tests for plugin.metadata property."""

    @staticmethod
    def test_metadata_property_returns_plugin_metadata() -> None:
        """Verify metadata property returns PluginMetadata type."""
        plugin = ImportGraphPlugin()
        meta = plugin.metadata
        expect_equal(meta.name, "graphs.import_graph")
        expect_equal(meta.version, "2.0.0")

    @staticmethod
    def test_core_metadata_property() -> None:
        """Verify core_metadata returns CorePluginMetadata."""
        plugin = ImportGraphPlugin()
        core = plugin.core_metadata
        expect_is_not_none(core)
        expect_equal(core, IMPORT_GRAPH_METADATA)


class TestImportGraphOptionsModel:
    """Tests for ImportGraphOptions dataclass."""

    @staticmethod
    def test_default_values() -> None:
        """Verify default values match expected configuration."""
        opts = ImportGraphOptions()
        expect_true(opts.include_third_party)
        expect_true(not opts.include_stdlib)
