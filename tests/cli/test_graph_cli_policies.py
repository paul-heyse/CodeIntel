"""Graph CLI planning policy behavior via command handler."""

from __future__ import annotations

import pytest

from codeintel.cli.graphs_handlers import (
    GraphPluginsOptions,
    OutputFormat,
    PlanMode,
    graph_plugins_handler_structured,
)
from codeintel.cli.result_types import GraphPlanResult
from codeintel.graphs.core.registry import (
    DependencyPolicy,
    SelectionPolicy,
    reset_graph_registry,
)
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.fakes.graph_plugins import GraphPluginBuilder, plugin_registrar


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_graph_registry()


def test_cli_lenient_selection_skips_unknown_plugin() -> None:
    """Default lenient selection should mark unknown plugins as skipped."""
    options = GraphPluginsOptions(
        mode=PlanMode.PLAN,
        names=("nonexistent_plugin",),
        enable=None,
        disable=(),
        selection_policy=SelectionPolicy.LENIENT,
        dependency_policy=DependencyPolicy.STRICT,
        validation_mode=False,
        output_format=OutputFormat.JSON,
    )

    result = graph_plugins_handler_structured(options)
    expect_true(result.success)
    expect_true(isinstance(result.data, GraphPlanResult))
    plan_result = result.data
    # Use .skipped attribute (list of dicts with name/reason)
    skipped = {entry["name"]: entry["reason"] for entry in plan_result.skipped}
    expect_equal(skipped["nonexistent_plugin"], "missing_graph")


def test_cli_strict_selection_falls_back_on_unknown_plugin() -> None:
    """Strict selection should fail planning and return empty plan."""
    options = GraphPluginsOptions(
        mode=PlanMode.PLAN,
        names=("nonexistent_plugin",),
        enable=None,
        disable=(),
        selection_policy=SelectionPolicy.STRICT,
        dependency_policy=DependencyPolicy.STRICT,
        validation_mode=False,
        output_format=OutputFormat.TEXT,
    )

    result = graph_plugins_handler_structured(options)
    # When strict selection fails, we get an empty plan (planning returned None)
    expect_true(result.success)
    expect_true(isinstance(result.data, GraphPlanResult))
    plan_result = result.data
    expect_equal(plan_result.plan_id, "empty")


def test_cli_dependency_skip_records_missing_dependency() -> None:
    """Dependency skip policy should record missing dependencies instead of raising."""
    main_name = "cli_missing_dep_main"
    missing_dep = "cli_missing_dep_missing"

    with plugin_registrar([GraphPluginBuilder(name=main_name, depends_on=(missing_dep,)).build()]):
        options = GraphPluginsOptions(
            mode=PlanMode.PLAN,
            names=(main_name,),
            enable=None,
            disable=(),
            selection_policy=SelectionPolicy.LENIENT,
            dependency_policy=DependencyPolicy.SKIP,
            validation_mode=False,
            output_format=OutputFormat.JSON,
        )
        result = graph_plugins_handler_structured(options)

    expect_true(result.success)
    expect_true(isinstance(result.data, GraphPlanResult))
    plan_result = result.data
    # Use .skipped attribute (list of dicts with name/reason)
    skipped = {entry["name"]: entry["reason"] for entry in plan_result.skipped}
    expect_equal(skipped[missing_dep], "missing_dependency")


def test_cli_validation_mode_enforces_strict_policy() -> None:
    """Validation mode should enforce strict selection/dependency and avoid stubs."""
    options = GraphPluginsOptions(
        mode=PlanMode.PLAN,
        names=("nonexistent_plugin",),
        enable=None,
        disable=(),
        selection_policy=SelectionPolicy.LENIENT,
        dependency_policy=DependencyPolicy.SKIP,
        validation_mode=True,
        output_format=OutputFormat.TEXT,
    )

    result = graph_plugins_handler_structured(options)
    # When validation mode fails, we get an empty plan (planning returned None)
    expect_true(result.success)
    expect_true(isinstance(result.data, GraphPlanResult))
    plan_result = result.data
    expect_equal(plan_result.plan_id, "empty")
