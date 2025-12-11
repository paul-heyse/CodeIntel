"""Tests for TypeCoveragePlugin metadata and options integration."""

from __future__ import annotations

from typing import Any

from codeintel.analytics.plugins.types.coverage import (
    TYPE_COVERAGE_METADATA,
    TypeCoveragePlugin,
)
from codeintel.analytics.plugins.types.options import TypeCoverageOptions
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


class TestTypeCoverageMetadata:
    """Tests for TYPE_COVERAGE_METADATA constant."""

    @staticmethod
    def test_metadata_identity() -> None:
        """Verify metadata identity fields."""
        expect_equal(TYPE_COVERAGE_METADATA.name, "analytics.type_coverage")
        expect_equal(TYPE_COVERAGE_METADATA.domain, PluginDomain.ANALYTICS)
        expect_equal(TYPE_COVERAGE_METADATA.kind, "metric")
        expect_equal(TYPE_COVERAGE_METADATA.stage, "function")

    @staticmethod
    def test_metadata_capabilities() -> None:
        """Verify provides/requires are set."""
        expect_in("analytics.type_coverage", TYPE_COVERAGE_METADATA.provides)
        expect_in("core.goids", TYPE_COVERAGE_METADATA.requires)
        expect_in("analytics.function_types", TYPE_COVERAGE_METADATA.requires)

    @staticmethod
    def test_metadata_tables() -> None:
        """Verify tables are set."""
        expect_in("analytics.type_coverage", TYPE_COVERAGE_METADATA.produces_tables)
        expect_in("core.goids", TYPE_COVERAGE_METADATA.consumes_tables)
        expect_in("analytics.function_types", TYPE_COVERAGE_METADATA.consumes_tables)


class TestTypeCoveragePluginOptionsIntegration:
    """Tests for options resolution integration."""

    @staticmethod
    def test_default_options_without_resolver() -> None:
        """Verify default options when no resolver provided."""
        plugin = TypeCoveragePlugin()
        opts = plugin.resolve_options()
        expect_true(opts.include_private)
        expect_equal(opts.strictness, "standard")

    @staticmethod
    def test_options_with_config() -> None:
        """Verify config source overrides defaults."""
        source = DictConfigSource(
            {
                "analytics.type_coverage": {
                    "include_private": False,
                    "strictness": "strict",
                    "scope_paths": ["src/"],
                },
            }
        )
        resolver = PluginOptionsResolver(source)
        plugin = TypeCoveragePlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        expect_true(not opts.include_private)
        expect_equal(opts.strictness, "strict")
        expect_equal(opts.scope_paths, ["src/"])


class TestTypeCoveragePluginMetadataProperty:
    """Tests for plugin.metadata property."""

    @staticmethod
    def test_metadata_property_returns_plugin_metadata() -> None:
        """Verify metadata property returns PluginMetadata type."""
        plugin = TypeCoveragePlugin()
        meta = plugin.metadata
        expect_equal(meta.name, "analytics.type_coverage")
        expect_equal(meta.version, "2.0.0")

    @staticmethod
    def test_core_metadata_property() -> None:
        """Verify core_metadata returns CorePluginMetadata."""
        plugin = TypeCoveragePlugin()
        core = plugin.core_metadata
        expect_is_not_none(core)
        expect_equal(core, TYPE_COVERAGE_METADATA)


class TestTypeCoverageOptionsModel:
    """Tests for TypeCoverageOptions dataclass."""

    @staticmethod
    def test_default_values() -> None:
        """Verify default values match expected configuration."""
        opts = TypeCoverageOptions()
        expect_true(opts.include_private)
        expect_equal(opts.strictness, "standard")
