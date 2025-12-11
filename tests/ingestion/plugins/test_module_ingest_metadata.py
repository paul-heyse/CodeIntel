"""Tests for ModuleIngestPlugin metadata and options integration."""

from __future__ import annotations

from typing import Any

from codeintel.core.plugins.execution.options import ConfigSource, PluginOptionsResolver
from codeintel.core.plugins.types.metadata import PluginDomain
from codeintel.ingestion.plugins.modules_options import ModuleIngestOptions
from codeintel.ingestion.plugins.modules_plugin import (
    MODULE_INGEST_METADATA,
    ModuleIngestPlugin,
)
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


class TestModuleIngestMetadata:
    """Tests for MODULE_INGEST_METADATA constant."""

    @staticmethod
    def test_metadata_identity() -> None:
        """Verify metadata identity fields."""
        expect_equal(MODULE_INGEST_METADATA.name, "ingest.modules")
        expect_equal(MODULE_INGEST_METADATA.domain, PluginDomain.INGEST)
        expect_equal(MODULE_INGEST_METADATA.kind, "builder")
        expect_equal(MODULE_INGEST_METADATA.stage, "goid")

    @staticmethod
    def test_metadata_capabilities() -> None:
        """Verify provides/requires are set."""
        expect_in("core.modules", MODULE_INGEST_METADATA.provides)
        expect_true(not MODULE_INGEST_METADATA.requires)

    @staticmethod
    def test_metadata_tables() -> None:
        """Verify tables are set."""
        expect_in("core.modules", MODULE_INGEST_METADATA.produces_tables)

    @staticmethod
    def test_metadata_is_scope_aware() -> None:
        """Verify plugin is marked as scope-aware."""
        expect_true(MODULE_INGEST_METADATA.scope_aware)


class TestModuleIngestPluginOptionsIntegration:
    """Tests for options resolution integration."""

    @staticmethod
    def test_default_options_without_resolver() -> None:
        """Verify default options when no resolver provided."""
        plugin = ModuleIngestPlugin()
        opts = plugin.resolve_options()
        expect_true(opts.include_tests)
        expect_true(not opts.include_generated)

    @staticmethod
    def test_options_with_config() -> None:
        """Verify config source overrides defaults."""
        source = DictConfigSource(
            {
                "ingest.modules": {
                    "include_tests": False,
                    "include_generated": True,
                    "scope_paths": ["src/"],
                },
            }
        )
        resolver = PluginOptionsResolver(source)
        plugin = ModuleIngestPlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        expect_true(not opts.include_tests)
        expect_true(opts.include_generated)
        expect_equal(opts.scope_paths, ["src/"])


class TestModuleIngestPluginMetadataProperty:
    """Tests for plugin.metadata property."""

    @staticmethod
    def test_metadata_property_returns_plugin_metadata() -> None:
        """Verify metadata property returns PluginMetadata type."""
        plugin = ModuleIngestPlugin()
        meta = plugin.metadata
        expect_equal(meta.name, "ingest.modules")
        expect_equal(meta.version, "2.0.0")

    @staticmethod
    def test_core_metadata_property() -> None:
        """Verify core_metadata returns CorePluginMetadata."""
        plugin = ModuleIngestPlugin()
        core = plugin.core_metadata
        expect_is_not_none(core)
        expect_equal(core, MODULE_INGEST_METADATA)


class TestModuleIngestOptionsModel:
    """Tests for ModuleIngestOptions dataclass."""

    @staticmethod
    def test_default_values() -> None:
        """Verify default values match expected configuration."""
        opts = ModuleIngestOptions()
        expect_true(opts.include_tests)
        expect_true(not opts.include_generated)
