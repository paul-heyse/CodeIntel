"""Tests for plugin registration and discovery.

This module verifies that all expected plugins are available in ALL_PLUGINS
and that they have the correct metadata.

NOTE: The analytics plugins now use TargetPlugin instead of
AnalyticsPluginProtocol. Tests access `plugin_name` directly instead of
`.metadata.name`. Registration is implicit via ALL_PLUGINS.
"""

from __future__ import annotations

import pytest

from codeintel.analytics.plugins.registration import ALL_PLUGINS
from codeintel.build.plugin import TargetPlugin


def test_all_plugins_have_names() -> None:
    """Verify ALL_PLUGINS all have unique plugin names."""
    plugin_names = {p.plugin_name for p in ALL_PLUGINS}
    # All plugins should have unique names
    assert len(plugin_names) == len(ALL_PLUGINS), "Duplicate plugin names found"


def test_expected_plugins_are_present() -> None:
    """Verify all expected plugin names are present in ALL_PLUGINS."""
    expected_plugins = {
        "functions.metrics",
        "functions.ast_features",
        "functions.effects",
        "functions.contracts",
        "functions.history",
        "coverage.functions",
        "coverage.test_edges",
        "tests.profile",
        "tests.behavioral_coverage",
        "hotspots.build",
        "subsystems.build",
        "semantic.roles",
        "data_models.build",
        "data_models.usage",
        "entrypoints.build",
        "deps.external",
        "profiles.build",
        "history.timeseries",
        "risk_factors.build",
        "config.data_flow",
    }

    registered_names = {p.plugin_name for p in ALL_PLUGINS}

    missing = expected_plugins - registered_names
    assert not missing, f"Missing expected plugins: {missing}"


def test_no_duplicate_plugin_names() -> None:
    """Verify no plugins share the same name."""
    names = [p.plugin_name for p in ALL_PLUGINS]

    # Check for duplicates
    duplicates = [name for name in names if names.count(name) > 1]
    assert not duplicates, f"Duplicate plugin names found: {set(duplicates)}"


def test_plugins_have_valid_metadata() -> None:
    """Verify all registered plugins have complete metadata."""
    for plugin in ALL_PLUGINS:
        assert plugin.plugin_name, "Plugin must have a name"
        assert plugin.plugin_description, f"Plugin {plugin.plugin_name} must have a description"
        assert plugin.plugin_version, f"Plugin {plugin.plugin_name} must have a version"


def test_plugins_are_target_plugins() -> None:
    """Verify all plugins inherit from TargetPlugin."""
    for plugin in ALL_PLUGINS:
        assert isinstance(plugin, TargetPlugin), f"{plugin.plugin_name} is not a TargetPlugin"


@pytest.mark.parametrize(
    "plugin_name",
    [
        "functions.metrics",
        "functions.ast_features",
        "coverage.functions",
        "tests.profile",
    ],
)
def test_specific_plugins_present(plugin_name: str) -> None:
    """Verify specific expected plugins are present."""
    names = {p.plugin_name for p in ALL_PLUGINS}
    assert plugin_name in names, f"Expected plugin {plugin_name} not found"
