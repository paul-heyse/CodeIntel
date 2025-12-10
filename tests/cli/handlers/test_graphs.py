"""Tests for graphs handlers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from codeintel.cli.config.model import CliConfig
from codeintel.cli.handlers.context import HandlerContext
from codeintel.cli.handlers.graphs import (
    GraphPlanResult,
    GraphPlanStage,
    GraphPluginInfo,
    GraphPluginsResult,
    PlanMode,
    graph_plugins_list_handler,
    graph_plugins_plan_handler,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)


def _make_mock_context(params: dict[str, Any]) -> HandlerContext:
    """Create a HandlerContext for testing.

    Parameters
    ----------
    params
        Parameters to include in the context.

    Returns
    -------
    HandlerContext
        Test context with provided params.
    """
    mock_config = MagicMock(spec=CliConfig)
    return HandlerContext(
        config=mock_config,
        operation_id="graphs.test",
        _params=params,
    )


def test_graph_plugin_info_to_dict() -> None:
    """Verify GraphPluginInfo.to_dict returns correct structure."""
    info = GraphPluginInfo(
        name="test_plugin",
        description="A test plugin",
        stage="analysis",
        enabled_by_default=True,
        depends_on=["dep1", "dep2"],
        provides=["feature1"],
    )

    data = info.to_dict()

    expect_equal(data["name"], "test_plugin")
    expect_equal(data["description"], "A test plugin")
    expect_equal(data["stage"], "analysis")
    expect_true(data["enabled_by_default"])
    expect_equal(data["depends_on"], ["dep1", "dep2"])
    expect_equal(data["provides"], ["feature1"])


def test_graph_plugins_result_to_dict() -> None:
    """Verify GraphPluginsResult.to_dict returns correct structure."""
    result = GraphPluginsResult(
        plugins=[
            GraphPluginInfo(
                name="plugin1",
                description="Plugin 1",
                stage="analysis",
                enabled_by_default=True,
                depends_on=[],
                provides=["feature1"],
            ),
            GraphPluginInfo(
                name="plugin2",
                description="Plugin 2",
                stage="transform",
                enabled_by_default=False,
                depends_on=["plugin1"],
                provides=["feature2"],
            ),
        ],
        count=2,
    )

    data = result.to_dict()

    expect_equal(data["count"], 2)
    plugins = data["plugins"]
    expect_true(isinstance(plugins, list))
    if isinstance(plugins, list):
        expect_equal(len(plugins), 2)
        expect_equal(plugins[0]["name"], "plugin1")
        expect_equal(plugins[1]["name"], "plugin2")


def test_graph_plan_stage_to_dict() -> None:
    """Verify GraphPlanStage.to_dict returns correct structure."""
    stage = GraphPlanStage(
        stage=1,
        plugins=["plugin1", "plugin2"],
    )

    data = stage.to_dict()

    expect_equal(data["stage"], 1)
    expect_equal(data["plugins"], ["plugin1", "plugin2"])


def test_graph_plan_result_to_dict() -> None:
    """Verify GraphPlanResult.to_dict returns correct structure."""
    result = GraphPlanResult(
        stages=[
            GraphPlanStage(stage=1, plugins=["plugin1"]),
            GraphPlanStage(stage=2, plugins=["plugin2", "plugin3"]),
        ],
        total_plugins=3,
        disabled=["disabled_plugin"],
    )

    data = result.to_dict()

    expect_equal(data["total_plugins"], 3)
    stages = data["stages"]
    expect_true(isinstance(stages, list))
    if isinstance(stages, list):
        expect_equal(len(stages), 2)
        expect_equal(stages[0]["stage"], 1)
        expect_equal(stages[1]["plugins"], ["plugin2", "plugin3"])
    expect_equal(data["disabled"], ["disabled_plugin"])


def test_plan_mode_values() -> None:
    """Verify PlanMode enum values."""
    expect_equal(PlanMode.LIST.value, "list")
    expect_equal(PlanMode.PLAN.value, "plan")


@patch("codeintel.cli.handlers.graphs.list_graph_plugins")
def test_graph_plugins_list_handler_success(mock_list_plugins: MagicMock) -> None:
    """Verify graph_plugins_list_handler returns plugins successfully."""
    # Create mock plugin
    mock_plugin = MagicMock()
    mock_plugin.metadata.name = "test_plugin"
    mock_plugin.metadata.description = "A test plugin"
    mock_plugin.metadata.stage = "analysis"
    mock_plugin.metadata.enabled_by_default = True
    mock_plugin.metadata.depends_on = ("dep1",)
    mock_plugin.metadata.provides = ("feature1",)

    mock_list_plugins.return_value = [mock_plugin]

    ctx = _make_mock_context({})

    result = graph_plugins_list_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    data = result.data
    if data is not None:
        expect_equal(data.count, 1)
        expect_equal(data.plugins[0].name, "test_plugin")


@patch("codeintel.cli.handlers.graphs.list_graph_plugins")
def test_graph_plugins_list_handler_with_names_filter(mock_list_plugins: MagicMock) -> None:
    """Verify graph_plugins_list_handler filters by names."""
    mock_plugin1 = MagicMock()
    mock_plugin1.metadata.name = "plugin1"
    mock_plugin1.metadata.description = "Plugin 1"
    mock_plugin1.metadata.stage = "analysis"
    mock_plugin1.metadata.enabled_by_default = True
    mock_plugin1.metadata.depends_on = ()
    mock_plugin1.metadata.provides = ()

    mock_plugin2 = MagicMock()
    mock_plugin2.metadata.name = "plugin2"
    mock_plugin2.metadata.description = "Plugin 2"
    mock_plugin2.metadata.stage = "transform"
    mock_plugin2.metadata.enabled_by_default = True
    mock_plugin2.metadata.depends_on = ()
    mock_plugin2.metadata.provides = ()

    mock_list_plugins.return_value = [mock_plugin1, mock_plugin2]

    ctx = _make_mock_context({"names": ["plugin1"]})

    result = graph_plugins_list_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.count, 1)
        expect_equal(data.plugins[0].name, "plugin1")


@patch("codeintel.cli.handlers.graphs.list_graph_plugins")
def test_graph_plugins_list_handler_include_disabled(mock_list_plugins: MagicMock) -> None:
    """Verify graph_plugins_list_handler includes disabled plugins when requested."""
    mock_enabled = MagicMock()
    mock_enabled.metadata.name = "enabled_plugin"
    mock_enabled.metadata.description = "Enabled"
    mock_enabled.metadata.stage = "analysis"
    mock_enabled.metadata.enabled_by_default = True
    mock_enabled.metadata.depends_on = ()
    mock_enabled.metadata.provides = ()

    mock_disabled = MagicMock()
    mock_disabled.metadata.name = "disabled_plugin"
    mock_disabled.metadata.description = "Disabled"
    mock_disabled.metadata.stage = "transform"
    mock_disabled.metadata.enabled_by_default = False
    mock_disabled.metadata.depends_on = ()
    mock_disabled.metadata.provides = ()

    mock_list_plugins.return_value = [mock_enabled, mock_disabled]

    # Without include_disabled - should only return enabled
    ctx = _make_mock_context({"include_disabled": False})
    result = graph_plugins_list_handler(ctx)
    data1 = result.data
    if data1 is not None:
        expect_equal(data1.count, 1)
        expect_equal(data1.plugins[0].name, "enabled_plugin")

    # With include_disabled - should return both
    ctx = _make_mock_context({"include_disabled": True})
    result = graph_plugins_list_handler(ctx)
    data2 = result.data
    if data2 is not None:
        expect_equal(data2.count, 2)


@patch("codeintel.cli.handlers.graphs.list_graph_plugins")
def test_graph_plugins_list_handler_empty(mock_list_plugins: MagicMock) -> None:
    """Verify graph_plugins_list_handler handles empty results."""
    mock_list_plugins.return_value = []

    ctx = _make_mock_context({})

    result = graph_plugins_list_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.count, 0)
        expect_equal(data.plugins, [])


@patch("codeintel.cli.handlers.graphs.plan_graph_plugins")
def test_graph_plugins_plan_handler_success(mock_plan_plugins: MagicMock) -> None:
    """Verify graph_plugins_plan_handler returns plan successfully."""
    mock_plugin = MagicMock()
    mock_plugin.metadata.name = "test_plugin"

    mock_plan = MagicMock()
    mock_plan.plugins = (mock_plugin,)
    mock_plan.skipped_plugins = ()

    mock_plan_plugins.return_value = mock_plan

    ctx = _make_mock_context({})

    result = graph_plugins_plan_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    data = result.data
    if data is not None:
        expect_equal(data.total_plugins, 1)
        expect_equal(len(data.stages), 1)


@patch("codeintel.cli.handlers.graphs.plan_graph_plugins")
def test_graph_plugins_plan_handler_with_skipped(mock_plan_plugins: MagicMock) -> None:
    """Verify graph_plugins_plan_handler includes skipped plugins."""
    mock_plugin = MagicMock()
    mock_plugin.metadata.name = "test_plugin"

    mock_skip = MagicMock()
    mock_skip.name = "skipped_plugin"

    mock_plan = MagicMock()
    mock_plan.plugins = (mock_plugin,)
    mock_plan.skipped_plugins = (mock_skip,)

    mock_plan_plugins.return_value = mock_plan

    ctx = _make_mock_context({})

    result = graph_plugins_plan_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.disabled, ["skipped_plugin"])


def test_graph_plugins_plan_handler_invalid_selection_policy() -> None:
    """Verify graph_plugins_plan_handler handles invalid selection policy."""
    ctx = _make_mock_context({"selection_policy": "invalid_policy"})

    result = graph_plugins_plan_handler(ctx)

    expect_true(not result.success)
    error = result.error
    if error is not None:
        expect_equal(error.type, "urn:codeintel:graphs:invalid-policy")


def test_graph_plugins_plan_handler_invalid_dependency_policy() -> None:
    """Verify graph_plugins_plan_handler handles invalid dependency policy."""
    ctx = _make_mock_context({"dependency_policy": "invalid_policy"})

    result = graph_plugins_plan_handler(ctx)

    expect_true(not result.success)
    error = result.error
    if error is not None:
        expect_equal(error.type, "urn:codeintel:graphs:invalid-policy")


@patch("codeintel.cli.handlers.graphs.plan_graph_plugins")
def test_graph_plugins_plan_handler_with_enable_disable(mock_plan_plugins: MagicMock) -> None:
    """Verify graph_plugins_plan_handler passes enable/disable params."""
    mock_plan = MagicMock()
    mock_plan.plugins = ()
    mock_plan.skipped_plugins = ()

    mock_plan_plugins.return_value = mock_plan

    ctx = _make_mock_context(
        {
            "names": ["plugin1"],
            "enable": ["plugin2"],
            "disable": ["plugin3"],
        }
    )

    result = graph_plugins_plan_handler(ctx)

    expect_true(result.success)

    # Verify the plan_graph_plugins was called with correct params
    call_kwargs = mock_plan_plugins.call_args.kwargs
    expect_equal(call_kwargs["plugin_names"], ["plugin1"])
    expect_equal(call_kwargs["enabled"], ["plugin2"])
    expect_equal(call_kwargs["disabled"], ["plugin3"])
