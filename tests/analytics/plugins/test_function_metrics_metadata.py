"""Tests for FunctionMetricsPlugin metadata and options integration."""

from __future__ import annotations

from typing import Any

from codeintel.build.plugins.analytics.functions.metrics import (
    FUNCTION_METRICS_METADATA,
    FunctionMetricsPlugin,
)
from codeintel.core.plugins.execution.options import (
    ConfigSource,
    EmptyConfigSource,
    PluginOptionsResolver,
)
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
            Raw options mapping when present.
        """
        return self._options.get(plugin_name)


class TestFunctionMetricsMetadata:
    """Tests for FUNCTION_METRICS_METADATA constant."""

    @staticmethod
    def test_metadata_name() -> None:
        """Verify metadata name is canonical."""
        expect_equal(FUNCTION_METRICS_METADATA.name, "analytics.function_metrics")

    @staticmethod
    def test_metadata_domain() -> None:
        """Verify metadata domain is analytics."""
        expect_equal(FUNCTION_METRICS_METADATA.domain, PluginDomain.ANALYTICS)

    @staticmethod
    def test_metadata_capabilities() -> None:
        """Verify provides and requires are populated."""
        expect_in("analytics.function_metrics", FUNCTION_METRICS_METADATA.provides)
        expect_in("core.goids", FUNCTION_METRICS_METADATA.requires)

    @staticmethod
    def test_metadata_tables() -> None:
        """Verify produces_tables is populated."""
        expect_in("analytics.function_metrics", FUNCTION_METRICS_METADATA.produces_tables)
        expect_in("analytics.function_types", FUNCTION_METRICS_METADATA.produces_tables)


class TestFunctionMetricsPluginOptionsIntegration:
    """Tests for options resolution integration."""

    @staticmethod
    def test_default_options_without_resolver() -> None:
        """Verify default options when no resolver provided."""
        plugin = FunctionMetricsPlugin()
        opts = plugin.resolve_options()
        expect_true(opts.include_graph_metrics)
        expect_true(opts.include_coverage_metrics)
        expect_equal(opts.complexity_threshold, 10)
        expect_equal(opts.type_strictness, "standard")

    @staticmethod
    def test_options_with_profile_config() -> None:
        """Verify profile options override defaults."""
        source = DictConfigSource(
            {
                "analytics.function_metrics": {
                    "include_graph_metrics": False,
                    "include_coverage_metrics": False,
                    "complexity_threshold": 20,
                    "type_strictness": "strict",
                    "scope_paths": ["src/"],
                },
            }
        )
        resolver = PluginOptionsResolver(source)
        plugin = FunctionMetricsPlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        expect_true(not opts.include_graph_metrics)
        expect_true(not opts.include_coverage_metrics)
        expect_equal(opts.complexity_threshold, 20)
        expect_equal(opts.type_strictness, "strict")
        expect_equal(opts.scope_paths, ["src/"])

    @staticmethod
    def test_dynamic_overrides() -> None:
        """Verify dynamic overrides are applied."""
        plugin = FunctionMetricsPlugin()
        overrides: dict[str, Any] = {"missing_function_goids": {1}}
        opts = plugin.resolve_options(dynamic_overrides=overrides)
        expect_equal(opts.get_missing_goids(), {1})


class TestFunctionMetricsPluginMetadataProperty:
    """Tests for plugin.metadata property."""

    @staticmethod
    def test_metadata_property_returns_plugin_metadata() -> None:
        """Verify metadata property returns PluginMetadata type."""
        plugin = FunctionMetricsPlugin()
        meta = plugin.metadata
        expect_equal(meta.name, "analytics.function_metrics")
        expect_equal(meta.version, "3.0.0")

    @staticmethod
    def test_core_metadata_property() -> None:
        """Verify core_metadata returns CorePluginMetadata."""
        plugin = FunctionMetricsPlugin()
        core = plugin.core_metadata
        expect_is_not_none(core)
        expect_equal(core, FUNCTION_METRICS_METADATA)


class TestFunctionMetricsPluginResolverInjection:
    """Tests for resolver injection."""

    @staticmethod
    def test_resolver_property_round_trip() -> None:
        """Verify resolver passed in constructor is used."""
        resolver = PluginOptionsResolver(EmptyConfigSource())
        plugin = FunctionMetricsPlugin(options_resolver=resolver)
        opts = plugin.resolve_options()
        expect_is_not_none(opts)
