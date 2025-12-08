"""Tests for graph CLI planning policy options."""

from __future__ import annotations

from codeintel.cli.commands import graphs as graphs_cmd
from codeintel.graphs.core.protocol import GraphPluginPlan
from codeintel.graphs.core.registry import DependencyPolicy, PlanningOptions, SelectionPolicy
from tests._helpers.assertions import expect_equal, expect_true


def test_plan_plugins_respects_selection_and_dependency(monkeypatch: object) -> None:
    """CLI passes selection/dependency policy into planning."""
    captured: dict[str, object] = {}

    def _fake_plan_graph_plugins(**kwargs: object) -> GraphPluginPlan:
        captured["kwargs"] = kwargs
        return GraphPluginPlan(plugins=(), plan_id="stub", skipped_plugins=(), dep_graph={})

    monkeypatch.setattr(graphs_cmd, "plan_graph_plugins", _fake_plan_graph_plugins)
    options = graphs_cmd.GraphPluginsOptions(
        mode=graphs_cmd.PlanMode.PLAN,
        names=None,
        enable=None,
        disable=(),
        selection_policy=SelectionPolicy.STRICT,
        dependency_policy=DependencyPolicy.SKIP,
        validation_mode=False,
        output_format=graphs_cmd.OutputFormat.TEXT,
    )

    graphs_cmd.graph_plugins_handler(options)

    plan_opts = captured["kwargs"]["plan_options"]
    expect_true(isinstance(plan_opts, PlanningOptions))
    expect_equal(plan_opts.selection_policy, SelectionPolicy.STRICT)
    expect_equal(plan_opts.dependency_policy, DependencyPolicy.SKIP)
    expect_true(plan_opts.use_stubs)


def test_plan_plugins_validation_mode_forces_strict(monkeypatch: object) -> None:
    """Validation mode enforces strict policies and disables stubs."""
    captured: dict[str, object] = {}

    def _fake_plan_graph_plugins(**kwargs: object) -> GraphPluginPlan:
        captured["kwargs"] = kwargs
        return GraphPluginPlan(plugins=(), plan_id="stub", skipped_plugins=(), dep_graph={})

    monkeypatch.setattr(graphs_cmd, "plan_graph_plugins", _fake_plan_graph_plugins)
    options = graphs_cmd.GraphPluginsOptions(
        mode=graphs_cmd.PlanMode.PLAN,
        names=None,
        enable=None,
        disable=(),
        selection_policy=SelectionPolicy.LENIENT,
        dependency_policy=DependencyPolicy.SKIP,
        validation_mode=True,
        output_format=graphs_cmd.OutputFormat.TEXT,
    )

    graphs_cmd.graph_plugins_handler(options)

    plan_opts = captured["kwargs"]["plan_options"]
    expect_true(isinstance(plan_opts, PlanningOptions))
    expect_equal(plan_opts.selection_policy, SelectionPolicy.STRICT)
    expect_equal(plan_opts.dependency_policy, DependencyPolicy.STRICT)
    expect_true(not plan_opts.use_stubs)
    expect_true(not plan_opts.allow_missing_dependencies)
