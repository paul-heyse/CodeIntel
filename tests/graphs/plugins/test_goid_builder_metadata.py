"""Tests for GoidBuilderPlugin metadata and options integration."""

from __future__ import annotations

from typing import Any

from codeintel.core.plugins.execution.options import ConfigSource, PluginOptionsResolver
from codeintel.core.plugins.types.metadata import PluginDomain
from codeintel.graphs.plugins.builders.goid import GOID_BUILDER_METADATA, GoidBuilderPlugin
from codeintel.graphs.plugins.builders.goid_options import GoidBuilderOptions
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


class TestGoidBuilderMetadata:
    """Tests for GOID_BUILDER_METADATA constant."""

    @staticmethod
    def test_metadata_name() -> None:
        """Verify metadata name is canonical."""
        expect_equal(GOID_BUILDER_METADATA.name, "graphs.goid_builder")

    @staticmethod
    def test_metadata_domain() -> None:
        """Verify metadata domain is graph."""
        expect_equal(GOID_BUILDER_METADATA.domain, PluginDomain.GRAPH)

    @staticmethod
    def test_metadata_kind_and_stage() -> None:
        """Verify kind and stage."""
        expect_equal(GOID_BUILDER_METADATA.kind, "builder")
        expect_equal(GOID_BUILDER_METADATA.stage, "goid")

    @staticmethod
    def test_metadata_capabilities() -> None:
        """Verify provides and requires."""
        expect_in("core.goids", GOID_BUILDER_METADATA.provides)
        expect_in("core.modules", GOID_BUILDER_METADATA.requires)

    @staticmethod
    def test_metadata_tables() -> None:
        """Verify produces_tables."""
        expect_in("core.goids", GOID_BUILDER_METADATA.produces_tables)
        expect_in("core.goid_crosswalk", GOID_BUILDER_METADATA.produces_tables)

    @staticmethod
    def test_metadata_is_scope_aware() -> None:
        """Verify plugin is marked as scope-aware."""
        expect_true(GOID_BUILDER_METADATA.scope_aware)


class TestGoidBuilderPluginOptionsIntegration:
    """Tests for options resolution integration."""

    @staticmethod
    def test_default_options_without_resolver() -> None:
        """Verify default options when no resolver provided."""
        plugin = GoidBuilderPlugin()
        opts = plugin.resolve_options()
        expect_true(opts.include_tests)
        expect_true(opts.include_private)
        expect_equal(opts.scope_paths, None)

    @staticmethod
    def test_options_with_profile() -> None:
        """Verify options from config source."""
        source = DictConfigSource(
            {
                "graphs.goid_builder": {
                    "scope_paths": ["src/"],
                    "include_tests": False,
                    "include_private": False,
                },
            }
        )
        resolver = PluginOptionsResolver(source)
        plugin = GoidBuilderPlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        expect_equal(opts.scope_paths, ["src/"])
        expect_true(not opts.include_tests)
        expect_true(not opts.include_private)


class TestGoidBuilderPluginMetadataProperty:
    """Tests for plugin.metadata property."""

    @staticmethod
    def test_metadata_property_returns_plugin_metadata() -> None:
        """Verify metadata property returns PluginMetadata type."""
        plugin = GoidBuilderPlugin()
        meta = plugin.metadata
        expect_equal(meta.name, "graphs.goid_builder")
        expect_equal(meta.version, "3.0.0")

    @staticmethod
    def test_core_metadata_property() -> None:
        """Verify core_metadata returns CorePluginMetadata."""
        plugin = GoidBuilderPlugin()
        core = plugin.core_metadata
        expect_is_not_none(core)
        expect_equal(core, GOID_BUILDER_METADATA)


class TestGoidBuilderOptionsModel:
    """Tests for GoidBuilderOptions dataclass."""

    @staticmethod
    def test_default_values() -> None:
        """Verify default values match expected configuration."""
        opts = GoidBuilderOptions()
        expect_true(opts.include_tests)
        expect_true(opts.include_private)
        expect_equal(opts.scope_paths, None)
