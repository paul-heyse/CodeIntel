"""Policy behavior tests for graph planning entrypoints."""

from __future__ import annotations

import pytest

from codeintel.graphs.core.registry import (
    DependencyPolicy,
    PlanningOptions,
    SelectionPolicy,
    plan_graph_plugins,
)
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.fakes.graph_plugins import GraphPluginBuilder, plugin_registrar


def test_lenient_selection_skips_unknown_plugin() -> None:
    """Lenient selection records missing_graph skips for unknown plugins."""
    plan = plan_graph_plugins(
        plugin_names=("nonexistent_plugin",),
        plan_options=PlanningOptions(
            selection_policy=SelectionPolicy.LENIENT,
            requested_required=False,
        ),
    )
    skipped = {skip.name: skip.reason for skip in plan.skipped_plugins}
    expect_equal(skipped["nonexistent_plugin"], "missing_graph")


def test_lenient_selection_with_required_requests_raises() -> None:
    """Lenient selection still raises when requests are marked required."""
    with pytest.raises(ValueError, match="is not registered"):
        plan_graph_plugins(
            plugin_names=("nonexistent_plugin",),
            plan_options=PlanningOptions(selection_policy=SelectionPolicy.LENIENT),
        )


def test_strict_selection_raises_on_unknown_plugin() -> None:
    """Strict selection raises when requested plugin is unknown."""
    with pytest.raises(ValueError, match="is not registered"):
        plan_graph_plugins(
            plugin_names=("nonexistent_plugin",),
            plan_options=PlanningOptions(selection_policy=SelectionPolicy.STRICT),
        )


def test_dependency_skip_records_missing_dependency() -> None:
    """Skip dependency policy records missing_dependency instead of raising."""
    main_name = "missing_dep_test"
    missing_dep = "missing_dep_missing"
    with plugin_registrar([GraphPluginBuilder(name=main_name, depends_on=(missing_dep,)).build()]):
        plan = plan_graph_plugins(
            plugin_names=(main_name,),
            plan_options=PlanningOptions(
                dependency_policy=DependencyPolicy.SKIP,
                selection_policy=SelectionPolicy.LENIENT,
            ),
        )
    skipped = {skip.name: skip.reason for skip in plan.skipped_plugins}
    expect_equal(skipped[missing_dep], "missing_dependency")
    expect_true(missing_dep in plan.dep_graph)


def test_validation_style_policies_fail_when_missing() -> None:
    """Strict policies with stubs disabled should surface missing plugin errors."""
    with pytest.raises(ValueError, match="not registered"):
        plan_graph_plugins(
            plugin_names=("nonexistent_plugin",),
            plan_options=PlanningOptions(
                selection_policy=SelectionPolicy.STRICT,
                dependency_policy=DependencyPolicy.STRICT,
                use_stubs=False,
                allow_missing_dependencies=False,
            ),
        )
