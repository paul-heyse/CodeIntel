"""Tests for SymbolUsesPlugin metadata and options integration."""

from __future__ import annotations

from typing import Any

from codeintel.core.plugins.execution.options import ConfigSource, PluginOptionsResolver
from codeintel.core.plugins.types.metadata import PluginDomain
from codeintel.graphs.plugins.builders.symbol_uses import (
    SYMBOL_USES_METADATA,
    SymbolUsesPlugin,
)
from codeintel.graphs.plugins.builders.symbol_uses_options import SymbolUsesOptions
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
            Raw options mapping when present.
        """
        return self._options.get(plugin_name)


class TestSymbolUsesMetadata:
    """Tests for SYMBOL_USES_METADATA constant."""

    @staticmethod
    def test_metadata_name() -> None:
        """Verify metadata name is canonical."""
        expect_equal(SYMBOL_USES_METADATA.name, "graphs.symbol_uses")

    @staticmethod
    def test_metadata_domain() -> None:
        """Verify metadata domain is graph."""
        expect_equal(SYMBOL_USES_METADATA.domain, PluginDomain.GRAPH)

    @staticmethod
    def test_metadata_kind_and_stage() -> None:
        """Verify kind and stage."""
        expect_equal(SYMBOL_USES_METADATA.kind, "builder")
        expect_equal(SYMBOL_USES_METADATA.stage, "edges")

    @staticmethod
    def test_metadata_capabilities() -> None:
        """Verify provides and requires."""
        expect_in("graph.symbol_uses", SYMBOL_USES_METADATA.provides)
        expect_in("core.scip_occurrences", SYMBOL_USES_METADATA.requires)
        expect_in("core.modules", SYMBOL_USES_METADATA.requires)

    @staticmethod
    def test_metadata_tables() -> None:
        """Verify produces_tables."""
        expect_in("graph.symbol_use_edges", SYMBOL_USES_METADATA.produces_tables)

    @staticmethod
    def test_metadata_is_scope_aware() -> None:
        """Verify plugin is marked as scope-aware."""
        expect_true(SYMBOL_USES_METADATA.scope_aware)

    @staticmethod
    def test_metadata_extra_graph_kinds() -> None:
        """Verify extra.graph_kinds is set."""
        expect_equal(SYMBOL_USES_METADATA.extra.get("graph_kinds"), ("symbol_use",))


class TestSymbolUsesPluginOptionsIntegration:
    """Tests for options resolution integration."""

    @staticmethod
    def test_default_options_without_resolver() -> None:
        """Verify default options when no resolver provided."""
        plugin = SymbolUsesPlugin()
        opts = plugin.resolve_options()
        expect_true(opts.include_tests)
        expect_equal(opts.scope_paths, None)

    @staticmethod
    def test_options_with_profile() -> None:
        """Verify options from config source."""
        source = DictConfigSource(
            {
                "graphs.symbol_uses": {
                    "scope_paths": ["src/"],
                    "include_tests": False,
                },
            }
        )
        resolver = PluginOptionsResolver(source)
        plugin = SymbolUsesPlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        expect_equal(opts.scope_paths, ["src/"])
        expect_true(not opts.include_tests)


class TestSymbolUsesPluginMetadataProperty:
    """Tests for plugin.metadata property."""

    @staticmethod
    def test_metadata_property_returns_plugin_metadata() -> None:
        """Verify metadata property returns PluginMetadata type."""
        plugin = SymbolUsesPlugin()
        meta = plugin.metadata
        expect_equal(meta.name, "graphs.symbol_uses")
        expect_equal(meta.version, "3.0.0")

    @staticmethod
    def test_core_metadata_property() -> None:
        """Verify core_metadata returns CorePluginMetadata."""
        plugin = SymbolUsesPlugin()
        core = plugin.core_metadata
        expect_is_not_none(core)
        expect_equal(core, SYMBOL_USES_METADATA)


class TestSymbolUsesOptionsModel:
    """Tests for SymbolUsesOptions dataclass."""

    @staticmethod
    def test_default_values() -> None:
        """Verify default values match expected configuration."""
        opts = SymbolUsesOptions()
        expect_true(opts.include_tests)
        expect_equal(opts.scope_paths, None)
