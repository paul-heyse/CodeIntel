"""Tests for graphs handlers."""

from __future__ import annotations

from codeintel.cli.handlers.graphs import (
    GraphPlanResult,
    GraphPlanStage,
    GraphPluginInfo,
    GraphPluginsResult,
    PlanMode,
    graph_plugins_list_handler,
    graph_plugins_plan_handler,
)
from codeintel.core.plugins.types.result import PluginResult
from codeintel.graphs.core.protocol import (
    GraphPluginExecutionContext,
    GraphPluginMetadata,
    GraphPluginProtocol,
    GraphPluginStage,
)
from codeintel.graphs.core.registry import reset_graph_registry
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.cli_context import make_command_context
from tests._helpers.fakes.graph_plugins import plugin_registrar


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


def _plugin(
    name: str,
    *,
    enabled: bool = True,
    depends_on: tuple[str, ...] = (),
    provides: tuple[str, ...] = (),
    stage: GraphPluginStage = "edges",
) -> GraphPluginProtocol:
    """Create a simple graph plugin with configurable metadata.

    Returns
    -------
    object
        Plugin class instance with configured metadata.
    """

    class SimplePlugin:
        metadata = GraphPluginMetadata(
            name=name,
            description=f"Plugin {name}",
            kind="builder",
            stage=stage,
            enabled_by_default=enabled,
            depends_on=depends_on,
            provides=provides,
        )

        def execute(self, ctx: GraphPluginExecutionContext) -> PluginResult:
            _ = self
            _ = ctx
            return PluginResult.ok()

    return SimplePlugin()


def test_graph_plugins_list_handler_success() -> None:
    """Verify graph_plugins_list_handler returns plugins successfully."""
    reset_graph_registry()
    plugin = _plugin("test_plugin", depends_on=("dep1",), provides=("feature1",))

    with plugin_registrar([plugin]), make_command_context({}, operation_id="graphs.test") as ctx:
        result = graph_plugins_list_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    data = result.data
    if data is not None:
        expect_equal(data.count, 1)
        expect_equal(data.plugins[0].name, "test_plugin")


def test_graph_plugins_list_handler_with_names_filter() -> None:
    """Verify graph_plugins_list_handler filters by names."""
    reset_graph_registry()
    plugin1 = _plugin("plugin1")
    plugin2 = _plugin("plugin2", stage="core")

    with (
        plugin_registrar([plugin1, plugin2]),
        make_command_context({"names": ["plugin1"]}, operation_id="graphs.test") as ctx,
    ):
        result = graph_plugins_list_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_equal(data.count, 1)
        expect_equal(data.plugins[0].name, "plugin1")


def test_graph_plugins_list_handler_include_disabled() -> None:
    """Verify graph_plugins_list_handler includes disabled plugins when requested."""
    reset_graph_registry()
    enabled_plugin = _plugin("enabled_plugin", enabled=True)
    disabled_plugin = _plugin("disabled_plugin", enabled=False, stage="core")

    with (
        plugin_registrar([enabled_plugin, disabled_plugin]),
        make_command_context({"include_disabled": False}, operation_id="graphs.test") as ctx,
    ):
        result = graph_plugins_list_handler(ctx)
    data1 = result.data
    if data1 is not None:
        expect_equal(data1.count, 1)
        expect_equal(data1.plugins[0].name, "enabled_plugin")

    with (
        plugin_registrar([enabled_plugin, disabled_plugin]),
        make_command_context({"include_disabled": True}, operation_id="graphs.test") as ctx,
    ):
        result = graph_plugins_list_handler(ctx)
    data2 = result.data
    if data2 is not None:
        expect_equal(data2.count, 2)


def test_graph_plugins_list_handler_empty() -> None:
    """Verify graph_plugins_list_handler handles empty results."""
    reset_graph_registry()

    with make_command_context({}, operation_id="graphs.test") as ctx:
        result = graph_plugins_list_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_true(data.count >= 0)


def test_graph_plugins_plan_handler_success() -> None:
    """Verify graph_plugins_plan_handler returns plan successfully."""
    reset_graph_registry()
    plugin = _plugin("test_plugin")

    with plugin_registrar([plugin]), make_command_context({}, operation_id="graphs.test") as ctx:
        result = graph_plugins_plan_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    data = result.data
    if data is not None:
        expect_equal(data.total_plugins, 1)
        expect_equal(len(data.stages), 1)


def test_graph_plugins_plan_handler_with_skipped() -> None:
    """Verify graph_plugins_plan_handler includes skipped plugins."""
    reset_graph_registry()
    plugin = _plugin("test_plugin", depends_on=("missing_plugin",))

    with plugin_registrar([plugin]), make_command_context({}, operation_id="graphs.test") as ctx:
        result = graph_plugins_plan_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        expect_true("missing_plugin" in data.disabled)


def test_graph_plugins_plan_handler_invalid_selection_policy() -> None:
    """Verify graph_plugins_plan_handler handles invalid selection policy."""
    with make_command_context(
        {"selection_policy": "invalid_policy"},
        operation_id="graphs.test",
    ) as ctx:
        result = graph_plugins_plan_handler(ctx)

    expect_true(not result.success)
    error = result.error
    if error is not None:
        expect_equal(error.type, "urn:codeintel:graphs:invalid-policy")


def test_graph_plugins_plan_handler_invalid_dependency_policy() -> None:
    """Verify graph_plugins_plan_handler handles invalid dependency policy."""
    with make_command_context(
        {"dependency_policy": "invalid_policy"},
        operation_id="graphs.test",
    ) as ctx:
        result = graph_plugins_plan_handler(ctx)

    expect_true(not result.success)
    error = result.error
    if error is not None:
        expect_equal(error.type, "urn:codeintel:graphs:invalid-policy")


def test_graph_plugins_plan_handler_with_enable_disable() -> None:
    """Verify graph_plugins_plan_handler honors enable/disable params."""
    reset_graph_registry()
    plugin1 = _plugin("plugin1", depends_on=("plugin3",))
    plugin2 = _plugin("plugin2")
    plugin3 = _plugin("plugin3")

    with (
        plugin_registrar([plugin1, plugin2, plugin3]),
        make_command_context(
            {
                "names": ["plugin1"],
                "enable": ["plugin2"],
                "disable": ["plugin3"],
            },
            operation_id="graphs.test",
        ) as ctx,
    ):
        result = graph_plugins_plan_handler(ctx)

    expect_true(result.success)
    data = result.data
    if data is not None:
        planned_plugins = [p for stage in data.stages for p in stage.plugins]
        expect_true("plugin1" in planned_plugins)
        expect_true("plugin2" in planned_plugins)
        expect_true("plugin3" not in planned_plugins)
        expect_true("plugin3" in data.disabled)
