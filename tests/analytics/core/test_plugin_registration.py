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
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_true,
)


def test_all_plugins_have_names() -> None:
    """Verify ALL_PLUGINS all have unique plugin names."""
    plugin_names = {p.plugin_name for p in ALL_PLUGINS}
    # All plugins should have unique names
    expect_equal(len(plugin_names), len(ALL_PLUGINS), label="Duplicate plugin names found")


def test_expected_plugins_are_present() -> None:
    """Verify all expected plugin names are present in ALL_PLUGINS."""
    expected_plugins = {
        "function_metrics",
        "function_ast_features",
        "function_effects",
        "function_contracts",
        "function_history",
        "coverage_functions",
        "coverage_test_edges",
        "test_profile",
        "behavioral_coverage",
        "hotspots",
        "subsystems",
        "semantic_roles",
        "data_models",
        "data_model_usage",
        "entrypoints",
        "external_deps",
        "profiles",
        "history_timeseries",
        "risk_factors",
        "config_data_flow",
    }

    registered_names = {p.plugin_name for p in ALL_PLUGINS}

    missing = expected_plugins - registered_names
    if missing:
        pytest.fail(f"Missing expected plugins: {missing}")


def test_no_duplicate_plugin_names() -> None:
    """Verify no plugins share the same name."""
    names = [p.plugin_name for p in ALL_PLUGINS]

    # Check for duplicates
    duplicates = [name for name in names if names.count(name) > 1]
    if duplicates:
        pytest.fail(f"Duplicate plugin names found: {set(duplicates)}")


def test_plugins_have_valid_metadata() -> None:
    """Verify all registered plugins have complete metadata."""
    for plugin in ALL_PLUGINS:
        expect_true(plugin.plugin_name, message="plugin name present")
        expect_true(
            plugin.plugin_description,
            message=f"plugin {plugin.plugin_name} description present",
        )
        expect_true(plugin.plugin_version, message=f"plugin {plugin.plugin_name} version present")


def test_plugins_are_target_plugins() -> None:
    """Verify all plugins inherit from TargetPlugin."""
    for plugin in ALL_PLUGINS:
        expect_is_instance(
            plugin, TargetPlugin, label=f"{plugin.plugin_name} is not a TargetPlugin"
        )


@pytest.mark.parametrize(
    "plugin_name",
    [
        "function_metrics",
        "function_ast_features",
        "coverage_functions",
        "test_profile",
    ],
)
def test_specific_plugins_present(plugin_name: str) -> None:
    """Verify specific expected plugins are present."""
    names = {p.plugin_name for p in ALL_PLUGINS}
    expect_in(plugin_name, names, label=f"Expected plugin {plugin_name} not found")
