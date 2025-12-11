"""Tests for plugin options infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from codeintel.core.plugins.execution.options import (
    ConfigSource,
    EmptyConfigSource,
    PluginConfigBundle,
    PluginOptionsResolver,
    ProfiledConfigSource,
)
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from tests._helpers.assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)

DEFAULT_THRESHOLD = 0.5
CUSTOM_THRESHOLD = 0.8


@dataclass
class SampleOptions:
    """Sample options model for testing."""

    threshold: float = DEFAULT_THRESHOLD
    enabled: bool = True
    name: str = "default"


class DictConfigSource:
    """ConfigSource backed by a dict for testing."""

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


@pytest.fixture
def sample_metadata() -> CorePluginMetadata:
    """Create sample metadata for testing.

    Returns
    -------
    CorePluginMetadata
        Sample metadata instance.
    """
    return CorePluginMetadata(
        name="test.plugin",
        version="1.0.0",
        description="Test plugin.",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        options_model=SampleOptions,
    )


class TestEmptyConfigSource:
    """Tests for EmptyConfigSource."""

    @staticmethod
    def test_always_returns_none() -> None:
        """Verify EmptyConfigSource returns None for any plugin."""
        source = EmptyConfigSource()
        expect_equal(source.get_plugin_options("any.plugin"), None)
        expect_equal(source.get_plugin_options("another.plugin"), None)

    @staticmethod
    def test_implements_protocol() -> None:
        """Verify EmptyConfigSource implements ConfigSource."""
        source = EmptyConfigSource()
        expect_true(isinstance(source, ConfigSource))


class TestPluginOptionsResolver:
    """Tests for PluginOptionsResolver."""

    @staticmethod
    def test_with_empty_source_uses_defaults(sample_metadata: CorePluginMetadata) -> None:
        """Verify default options are used with empty config."""
        resolver = PluginOptionsResolver(EmptyConfigSource())
        opts = resolver.get_options(sample_metadata, SampleOptions)
        expect_equal(opts.threshold, DEFAULT_THRESHOLD)
        expect_true(opts.enabled)
        expect_equal(opts.name, "default")

    @staticmethod
    def test_with_config_overrides_defaults(sample_metadata: CorePluginMetadata) -> None:
        """Verify config values override defaults."""
        source = DictConfigSource(
            {"test.plugin": {"threshold": CUSTOM_THRESHOLD, "name": "custom"}}
        )
        resolver = PluginOptionsResolver(source)
        opts = resolver.get_options(sample_metadata, SampleOptions)
        expect_equal(opts.threshold, CUSTOM_THRESHOLD)
        expect_true(opts.enabled)
        expect_equal(opts.name, "custom")

    @staticmethod
    def test_dynamic_overrides(sample_metadata: CorePluginMetadata) -> None:
        """Verify dynamic overrides are applied."""
        source = DictConfigSource({"test.plugin": {"threshold": CUSTOM_THRESHOLD}})
        resolver = PluginOptionsResolver(source)
        opts = resolver.get_options(
            sample_metadata,
            SampleOptions,
            dynamic_overrides={"name": "runtime"},
        )
        expect_equal(opts.threshold, CUSTOM_THRESHOLD)
        expect_equal(opts.name, "runtime")

    @staticmethod
    def test_config_source_property() -> None:
        """Verify config_source property returns the source."""
        source = EmptyConfigSource()
        resolver = PluginOptionsResolver(source)
        expect_equal(resolver.config_source, source)


class TestPluginConfigBundle:
    """Tests for PluginConfigBundle."""

    @staticmethod
    def test_get_existing_plugin() -> None:
        """Verify get returns options for existing plugin."""
        bundle = PluginConfigBundle(plugin_options={"plugin.a": {"key": "value"}})
        expect_equal(bundle.get("plugin.a"), {"key": "value"})

    @staticmethod
    def test_get_missing_plugin() -> None:
        """Verify get returns None for missing plugin."""
        bundle = PluginConfigBundle(plugin_options={"plugin.a": {"key": "value"}})
        expect_equal(bundle.get("plugin.b"), None)

    @staticmethod
    def test_none_plugin_options() -> None:
        """Verify None plugin_options is normalized to empty dict."""
        bundle = PluginConfigBundle(plugin_options=None)
        expect_equal(bundle.get("any.plugin"), None)


class TestProfiledConfigSource:
    """Tests for ProfiledConfigSource."""

    @staticmethod
    def test_base_only() -> None:
        """Verify base layer is used when no profile."""
        base = PluginConfigBundle(plugin_options={"plugin.a": {"key": "base_value"}})
        source = ProfiledConfigSource(base=base)
        opts = source.get_plugin_options("plugin.a")
        observed = expect_is_not_none(opts)
        expect_equal(observed["key"], "base_value")

    @staticmethod
    def test_profile_overrides_base() -> None:
        """Verify profile layer overrides base."""
        base = PluginConfigBundle(
            plugin_options={"plugin.a": {"key": "base", "other": "base_other"}}
        )
        profile = PluginConfigBundle(plugin_options={"plugin.a": {"key": "profile"}})
        source = ProfiledConfigSource(
            base=base,
            profile=profile,
            active_profile_name="fast",
        )
        opts = source.get_plugin_options("plugin.a")
        observed = expect_is_not_none(opts)
        expect_equal(observed["key"], "profile")
        expect_equal(observed["other"], "base_other")

    @staticmethod
    def test_cli_overrides_profile() -> None:
        """Verify CLI layer overrides profile."""
        base = PluginConfigBundle(plugin_options={"plugin.a": {"key": "base"}})
        profile = PluginConfigBundle(plugin_options={"plugin.a": {"key": "profile"}})
        cli = PluginConfigBundle(plugin_options={"plugin.a": {"key": "cli"}})
        source = ProfiledConfigSource(
            base=base,
            profile=profile,
            cli=cli,
            active_profile_name="fast",
        )
        opts = source.get_plugin_options("plugin.a")
        observed = expect_is_not_none(opts)
        expect_equal(observed["key"], "cli")

    @staticmethod
    def test_profile_not_applied_without_active_name() -> None:
        """Verify profile is not applied without active_profile_name."""
        base = PluginConfigBundle(plugin_options={"plugin.a": {"key": "base"}})
        profile = PluginConfigBundle(plugin_options={"plugin.a": {"key": "profile"}})
        source = ProfiledConfigSource(
            base=base,
            profile=profile,
            active_profile_name=None,
        )
        opts = source.get_plugin_options("plugin.a")
        observed = expect_is_not_none(opts)
        expect_equal(observed["key"], "base")

    @staticmethod
    def test_missing_plugin_returns_none() -> None:
        """Verify None is returned for unconfigured plugins."""
        source = ProfiledConfigSource()
        expect_equal(source.get_plugin_options("unknown.plugin"), None)

    @staticmethod
    def test_implements_protocol() -> None:
        """Verify ProfiledConfigSource implements ConfigSource."""
        source = ProfiledConfigSource()
        expect_true(isinstance(source, ConfigSource))
