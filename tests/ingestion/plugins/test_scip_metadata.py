"""Tests for ScipIngestPlugin metadata and options integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codeintel.core.plugins.execution.options import (
    ConfigSource,
    PluginOptionsResolver,
)
from codeintel.core.plugins.types.metadata import PluginDomain
from codeintel.ingestion.plugins.scip_plugin import (
    SCIP_INGEST_METADATA,
    ScipIngestPlugin,
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
            Raw options mapping when present.
        """
        return self._options.get(plugin_name)


class TestScipIngestMetadata:
    """Tests for SCIP_INGEST_METADATA constant."""

    @staticmethod
    def test_metadata_name() -> None:
        """Verify metadata name is canonical."""
        expect_equal(SCIP_INGEST_METADATA.name, "ingest.scip_python")

    @staticmethod
    def test_metadata_domain() -> None:
        """Verify metadata domain is ingest."""
        expect_equal(SCIP_INGEST_METADATA.domain, PluginDomain.INGEST)

    @staticmethod
    def test_metadata_kind_and_stage() -> None:
        """Verify kind and stage."""
        expect_equal(SCIP_INGEST_METADATA.kind, "builder")
        expect_equal(SCIP_INGEST_METADATA.stage, "goid")

    @staticmethod
    def test_metadata_capabilities() -> None:
        """Verify provides and requires."""
        expect_in("core.scip_symbols", SCIP_INGEST_METADATA.provides)
        expect_in("core.goid_crosswalk", SCIP_INGEST_METADATA.provides)
        expect_in("core.modules", SCIP_INGEST_METADATA.requires)

    @staticmethod
    def test_metadata_tables() -> None:
        """Verify produces_tables."""
        expect_in("core.scip_symbols", SCIP_INGEST_METADATA.produces_tables)
        expect_in("core.goid_crosswalk", SCIP_INGEST_METADATA.produces_tables)

    @staticmethod
    def test_metadata_is_scope_aware() -> None:
        """Verify plugin is marked as scope-aware."""
        expect_true(SCIP_INGEST_METADATA.scope_aware)

    @staticmethod
    def test_metadata_resource_hints() -> None:
        """Verify resource hints include tool requirement."""
        hints = SCIP_INGEST_METADATA.resource_hints
        expect_equal(hints.get("requires_tools"), ["scip-python"])


class TestScipIngestPluginOptionsIntegration:
    """Tests for options resolution integration."""

    @staticmethod
    def test_default_options_without_resolver() -> None:
        """Verify default options when no resolver provided."""
        plugin = ScipIngestPlugin()
        opts = plugin.resolve_options()
        expect_true(opts.include_references)
        expect_true(opts.include_implementations)
        expect_equal(opts.timeout_seconds, 300)

    @staticmethod
    def test_options_with_fast_profile() -> None:
        """Verify fast profile options."""
        source = DictConfigSource(
            {
                "ingest.scip_python": {
                    "include_references": False,
                    "include_implementations": False,
                    "timeout_seconds": 120,
                },
            }
        )
        resolver = PluginOptionsResolver(source)
        plugin = ScipIngestPlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        expect_true(not opts.include_references)
        expect_true(not opts.include_implementations)
        expect_equal(opts.timeout_seconds, 120)

    @staticmethod
    def test_dynamic_overrides() -> None:
        """Verify dynamic overrides are applied."""
        plugin = ScipIngestPlugin()
        scip_dir = Path.cwd() / "scip_artifacts"
        opts = plugin.resolve_options(dynamic_overrides={"scip_output_dir": scip_dir})
        expect_equal(opts.scip_output_dir, scip_dir)


class TestScipIngestPluginMetadataProperty:
    """Tests for plugin.metadata property."""

    @staticmethod
    def test_metadata_property_returns_plugin_metadata() -> None:
        """Verify metadata property returns PluginMetadata type."""
        plugin = ScipIngestPlugin()
        meta = plugin.metadata
        expect_equal(meta.name, "ingest.scip_python")
        expect_equal(meta.version, "3.0.0")

    @staticmethod
    def test_core_metadata_property() -> None:
        """Verify core_metadata returns CorePluginMetadata."""
        plugin = ScipIngestPlugin()
        core = plugin.core_metadata
        expect_is_not_none(core)
        expect_equal(core, SCIP_INGEST_METADATA)
