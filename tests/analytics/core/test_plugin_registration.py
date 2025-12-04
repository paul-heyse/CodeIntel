"""Tests for plugin registration and discovery.

This module verifies that all expected plugins are correctly registered
with the unified plugin registry and that the registration system works
correctly.
"""

from __future__ import annotations

import pytest

from codeintel.analytics.core.registry import get_registry
from codeintel.analytics.plugins.registration import (
    ALL_PLUGINS,
    ensure_plugins_registered,
)

# Minimum count for multi-plugin tests
MIN_MULTI_PLUGIN_COUNT = 2


def test_ensure_plugins_registered_is_idempotent() -> None:
    """Verify that calling ensure_plugins_registered multiple times is safe."""
    # First call should register
    ensure_plugins_registered()
    registry = get_registry()
    count_after_first = len(registry.list_all())

    # Second call should be a no-op
    ensure_plugins_registered()
    count_after_second = len(registry.list_all())

    assert count_after_first == count_after_second


def test_all_plugins_constant_matches_registry() -> None:
    """Verify ALL_PLUGINS matches what's in the registry."""
    ensure_plugins_registered()
    registry = get_registry()

    plugin_names_from_constant = {p.metadata.name for p in ALL_PLUGINS}
    plugin_names_from_registry = {p.metadata.name for p in registry.list_all()}

    # All constant plugins should be in registry
    assert plugin_names_from_constant <= plugin_names_from_registry


def test_expected_plugins_are_registered() -> None:
    """Verify all expected plugin names are present."""
    ensure_plugins_registered()
    registry = get_registry()

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
        # Note: core_graph_metrics is a graphs plugin, not an analytics plugin
    }

    registered_names = {p.metadata.name for p in registry.list_all()}

    missing = expected_plugins - registered_names
    assert not missing, f"Missing expected plugins: {missing}"


def test_no_duplicate_plugin_names() -> None:
    """Verify no plugins share the same name."""
    ensure_plugins_registered()
    registry = get_registry()

    plugins = registry.list_all()
    names = [p.metadata.name for p in plugins]

    # Check for duplicates
    duplicates = [name for name in names if names.count(name) > 1]
    assert not duplicates, f"Duplicate plugin names found: {set(duplicates)}"


def test_plugins_have_valid_metadata() -> None:
    """Verify all registered plugins have complete metadata."""
    ensure_plugins_registered()
    registry = get_registry()

    for plugin in registry.list_all():
        meta = plugin.metadata
        assert meta.name, "Plugin must have a name"
        assert meta.description, f"Plugin {meta.name} must have a description"
        assert meta.stage, f"Plugin {meta.name} must have a stage"


def test_plugins_can_be_retrieved_by_name() -> None:
    """Verify plugins can be retrieved by name."""
    ensure_plugins_registered()
    registry = get_registry()

    for plugin in ALL_PLUGINS:
        retrieved = registry.get(plugin.metadata.name)
        assert retrieved is plugin


def test_nonexistent_plugin_raises_keyerror() -> None:
    """Verify retrieving a non-existent plugin raises KeyError."""
    ensure_plugins_registered()
    registry = get_registry()

    with pytest.raises(KeyError):
        registry.get("nonexistent.plugin.name")


def test_plugins_providing_analytics_tables() -> None:
    """Verify plugins that provide analytics tables are discoverable."""
    ensure_plugins_registered()
    registry = get_registry()

    # Check that we can find plugins providing function_metrics
    providers = registry.list_providing("analytics.function_metrics")
    assert len(providers) > 0, "Expected at least one plugin providing function_metrics"


def test_plugins_by_stage() -> None:
    """Verify plugins can be filtered by stage."""
    ensure_plugins_registered()
    registry = get_registry()

    function_plugins = registry.list_by_stage("function")
    assert len(function_plugins) > 0, "Expected at least one function-stage plugin"

    test_plugins = registry.list_by_stage("test")
    assert len(test_plugins) > 0, "Expected at least one test-stage plugin"


def test_plan_single_plugin() -> None:
    """Verify planning a single plugin works."""
    ensure_plugins_registered()
    registry = get_registry()

    plan = registry.plan(["functions.metrics"])
    assert len(plan.plugins) >= 1
    assert "functions.metrics" in plan.ordered_names


def test_plan_multiple_plugins() -> None:
    """Verify planning multiple plugins works."""
    ensure_plugins_registered()
    registry = get_registry()

    plan = registry.plan(["functions.metrics", "functions.ast_features"])
    assert len(plan.plugins) >= MIN_MULTI_PLUGIN_COUNT
    assert "functions.metrics" in plan.ordered_names
    assert "functions.ast_features" in plan.ordered_names


def test_plan_with_disabled_plugin() -> None:
    """Verify disabled plugins are skipped in the plan."""
    ensure_plugins_registered()
    registry = get_registry()

    plan = registry.plan(
        ["functions.metrics", "functions.ast_features"],
        disabled=["functions.ast_features"],
    )

    assert "functions.metrics" in plan.ordered_names
    assert "functions.ast_features" not in plan.ordered_names
    assert any(s.name == "functions.ast_features" for s in plan.skipped)


def test_plan_unknown_plugin_skipped() -> None:
    """Verify planning with unknown plugin adds it to skipped list."""
    ensure_plugins_registered()
    registry = get_registry()

    # Unknown plugins are skipped rather than raising errors
    plan = registry.plan(["nonexistent.plugin"])
    assert len(plan.plugins) == 0
    assert any(s.name == "nonexistent.plugin" and s.reason == "missing_dependency" for s in plan.skipped)
