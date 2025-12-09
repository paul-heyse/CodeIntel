"""Graph CLI planning policy behavior via command handler."""

from __future__ import annotations

import json

import pytest

from codeintel.cli.graphs_handlers import (
    GraphPluginsOptions,
    OutputFormat,
    PlanMode,
    graph_plugins_handler,
)
from codeintel.graphs.core.registry import (
    DependencyPolicy,
    SelectionPolicy,
    reset_graph_registry,
)
from tests._helpers.assertions import expect_equal, expect_in, expect_true
from tests._helpers.fakes.graph_plugins import GraphPluginBuilder, plugin_registrar


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_graph_registry()


def test_cli_lenient_selection_skips_unknown_plugin(capsys: pytest.CaptureFixture[str]) -> None:
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

    graph_plugins_handler(options)
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    skipped = {entry["name"]: entry["reason"] for entry in payload["skipped_plugins"]}
    expect_equal(skipped["nonexistent_plugin"], "missing_graph")


def test_cli_strict_selection_falls_back_on_unknown_plugin(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Strict selection should fail planning and trigger fallback output."""
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

    graph_plugins_handler(options)
    captured = capsys.readouterr()
    expect_in("Failed to compute plan; showing available plugins", captured.err)


def test_cli_dependency_skip_records_missing_dependency(
    capsys: pytest.CaptureFixture[str],
) -> None:
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
        graph_plugins_handler(options)

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    skipped = {entry["name"]: entry["reason"] for entry in payload["skipped_plugins"]}
    expect_equal(skipped[missing_dep], "missing_dependency")
    expect_true(missing_dep in payload["dep_graph"])


def test_cli_validation_mode_enforces_strict_policy(capsys: pytest.CaptureFixture[str]) -> None:
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

    graph_plugins_handler(options)
    captured = capsys.readouterr()
    expect_in("Failed to compute plan; showing available plugins", captured.err)
