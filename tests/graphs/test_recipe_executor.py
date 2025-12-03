"""Tests for graph recipe executor.

This module tests the RecipeExecutor for running graph recipes,
orchestrating plugin execution across stages with support for
parallelism and failure handling.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.core.protocol import (
    FunctionalGraphPlugin,
    GraphPluginMetadata,
    GraphPluginProtocol,
)
from codeintel.graphs.core.registry import (
    register_graph_plugin,
    unregister_graph_plugin,
)
from codeintel.graphs.core.result import GraphPluginResult
from codeintel.graphs.recipes.dsl import (
    GraphRecipe,
    GraphRecipeOptions,
    GraphStage,
    graph_recipe,
    graph_stage,
)
from codeintel.graphs.recipes.executor import (
    RecipeExecutionResult,
    RecipeExecutor,
    RecipeExecutorContext,
    StageExecutionResult,
    execute_graph_recipe,
)

if TYPE_CHECKING:
    from codeintel.graphs.core.context import GraphExecutionContext
    from codeintel.storage.gateway import StorageGateway

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_STAGE_COUNT_ONE: Final[int] = 1
EXPECTED_STAGE_COUNT_TWO: Final[int] = 2
EXPECTED_PLUGIN_COUNT_THREE: Final[int] = 3
SUCCESS_ROW_COUNT: Final[int] = 42
MS_TO_SECONDS: Final[float] = 1000.0


# ---------------------------------------------------------------------------
# Test Plugin Helpers
# ---------------------------------------------------------------------------


def _make_succeeding_plugin(
    name: str = "test.succeeding",
    row_count: int = SUCCESS_ROW_COUNT,
) -> GraphPluginProtocol:
    """Create a plugin that always succeeds.

    Parameters
    ----------
    name
        Plugin name.
    row_count
        Row count to return.

    Returns
    -------
    GraphPluginProtocol
        Succeeding plugin.
    """
    metadata = GraphPluginMetadata(
        name=name,
        description="Test plugin that succeeds",
        kind="metric",
        stage="core",
    )

    def execute_fn(ctx: GraphExecutionContext) -> GraphPluginResult:
        _ = ctx  # Required by protocol
        return GraphPluginResult(
            success=True,
            row_counts={"test_table": row_count},
        )

    return FunctionalGraphPlugin(_metadata=metadata, _execute_fn=execute_fn)


def _make_failing_plugin(
    name: str = "test.failing",
    error_message: str = "Test failure",
) -> GraphPluginProtocol:
    """Create a plugin that always fails.

    Parameters
    ----------
    name
        Plugin name.
    error_message
        Error message.

    Returns
    -------
    GraphPluginProtocol
        Failing plugin.
    """
    metadata = GraphPluginMetadata(
        name=name,
        description="Test plugin that fails",
        kind="metric",
        stage="core",
    )

    def execute_fn(ctx: GraphExecutionContext) -> GraphPluginResult:
        _ = ctx  # Required by protocol
        raise RuntimeError(error_message)

    return FunctionalGraphPlugin(_metadata=metadata, _execute_fn=execute_fn)


class _PluginRegistrar:
    """Context manager to register and unregister test plugins."""

    def __init__(self, plugins: list[GraphPluginProtocol]) -> None:
        """Initialize with plugins to register.

        Parameters
        ----------
        plugins
            Plugins to register.
        """
        self._plugins = plugins

    def __enter__(self) -> Self:
        """Register all plugins.

        Returns
        -------
        Self
            Self for context manager protocol.
        """
        for plugin in self._plugins:
            register_graph_plugin(plugin)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Unregister plugins on exit."""
        for plugin in self._plugins:
            with contextlib.suppress(KeyError):
                unregister_graph_plugin(plugin.metadata.name)


# ---------------------------------------------------------------------------
# Fixtures and Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(tmp_path: Path) -> SnapshotRef:
    """Create a snapshot for testing.

    Parameters
    ----------
    tmp_path
        Temporary directory.

    Returns
    -------
    SnapshotRef
        Snapshot reference.
    """
    return SnapshotRef(
        repo="test-repo",
        commit="abc123",
        repo_root=tmp_path,
    )


def _make_simple_recipe(
    name: str = "test_recipe",
    plugin_names: tuple[str, ...] = ("test.succeeding",),
    *,
    parallel: bool = False,
    fail_fast: bool = True,
) -> GraphRecipe:
    """Create a simple single-stage recipe.

    Parameters
    ----------
    name
        Recipe name.
    plugin_names
        Plugin names for the stage.
    parallel
        Whether to execute in parallel.
    fail_fast
        Whether to fail fast.

    Returns
    -------
    GraphRecipe
        Recipe definition.
    """
    return graph_recipe(
        name=name,
        description="Test recipe",
        stages=[
            graph_stage(
                name="stage1",
                plugins=list(plugin_names),
                parallel=parallel,
                fail_fast=fail_fast,
            )
        ],
    )


def _make_multi_stage_recipe(
    name: str = "multi_stage_recipe",
    stage_configs: list[tuple[str, tuple[str, ...], bool]] | None = None,
) -> GraphRecipe:
    """Create a multi-stage recipe.

    Parameters
    ----------
    name
        Recipe name.
    stage_configs
        List of (stage_name, plugin_names, fail_fast) tuples.

    Returns
    -------
    GraphRecipe
        Recipe definition.
    """
    if stage_configs is None:
        stage_configs = [
            ("stage1", ("test.succeeding",), True),
            ("stage2", ("test.succeeding",), True),
        ]

    stages = [
        graph_stage(
            name=stage_name,
            plugins=list(plugins),
            fail_fast=fail_fast,
        )
        for stage_name, plugins, fail_fast in stage_configs
    ]

    return graph_recipe(
        name=name,
        description="Multi-stage test recipe",
        stages=stages,
    )


# ---------------------------------------------------------------------------
# StageExecutionResult Tests
# ---------------------------------------------------------------------------


def test_stage_execution_result_attributes() -> None:
    """StageExecutionResult has correct attributes."""
    from codeintel.graphs.core.result import GraphPluginRunRecord  # noqa: PLC0415

    records = (
        GraphPluginRunRecord(
            name="test_plugin",
            status="succeeded",
            started_at="2024-01-01T00:00:00Z",
            ended_at="2024-01-01T00:00:01Z",
            duration_ms=MS_TO_SECONDS,
            attempts=1,
            partial=False,
            error=None,
            meta={},
        ),
    )
    result = StageExecutionResult(
        stage_name="test_stage",
        records=records,
        success=True,
        duration_ms=MS_TO_SECONDS,
    )

    assert result.stage_name == "test_stage"
    assert len(result.records) == EXPECTED_STAGE_COUNT_ONE
    assert result.success is True
    assert result.duration_ms == MS_TO_SECONDS


# ---------------------------------------------------------------------------
# RecipeExecutionResult Tests
# ---------------------------------------------------------------------------


def test_recipe_execution_result_all_records() -> None:
    """RecipeExecutionResult aggregates records from all stages."""
    from codeintel.graphs.core.result import GraphPluginRunRecord  # noqa: PLC0415

    record1 = GraphPluginRunRecord(
        name="plugin1",
        status="succeeded",
        started_at="2024-01-01T00:00:00Z",
        ended_at="2024-01-01T00:00:01Z",
        duration_ms=MS_TO_SECONDS,
        attempts=1,
        partial=False,
        error=None,
        meta={},
    )
    record2 = GraphPluginRunRecord(
        name="plugin2",
        status="succeeded",
        started_at="2024-01-01T00:00:01Z",
        ended_at="2024-01-01T00:00:02Z",
        duration_ms=MS_TO_SECONDS,
        attempts=1,
        partial=False,
        error=None,
        meta={},
    )
    stage1 = StageExecutionResult(
        stage_name="stage1",
        records=(record1,),
        success=True,
        duration_ms=MS_TO_SECONDS,
    )
    stage2 = StageExecutionResult(
        stage_name="stage2",
        records=(record2,),
        success=True,
        duration_ms=MS_TO_SECONDS,
    )
    result = RecipeExecutionResult(
        recipe_name="test",
        run_id="run123",
        stages=(stage1, stage2),
        success=True,
        duration_ms=MS_TO_SECONDS * EXPECTED_STAGE_COUNT_TWO,
        started_at="2024-01-01T00:00:00Z",
        ended_at="2024-01-01T00:00:02Z",
    )

    assert len(result.all_records) == EXPECTED_STAGE_COUNT_TWO
    assert result.success_count == EXPECTED_STAGE_COUNT_TWO
    assert result.failure_count == 0
    assert result.skip_count == 0


def test_recipe_execution_result_counts_mixed_statuses() -> None:
    """RecipeExecutionResult counts mixed success/fail/skip."""
    from codeintel.graphs.core.result import GraphPluginRunRecord  # noqa: PLC0415

    succeeded_record = GraphPluginRunRecord(
        name="succeeded_plugin",
        status="succeeded",
        started_at="2024-01-01T00:00:00Z",
        ended_at="2024-01-01T00:00:01Z",
        duration_ms=MS_TO_SECONDS,
        attempts=1,
        partial=False,
        error=None,
        meta={},
    )
    failed_record = GraphPluginRunRecord(
        name="failed_plugin",
        status="failed",
        started_at="2024-01-01T00:00:00Z",
        ended_at="2024-01-01T00:00:01Z",
        duration_ms=MS_TO_SECONDS,
        attempts=1,
        partial=True,
        error="Error",
        meta={},
    )
    skipped_record = GraphPluginRunRecord(
        name="skipped_plugin",
        status="skipped",
        started_at="2024-01-01T00:00:00Z",
        ended_at="2024-01-01T00:00:01Z",
        duration_ms=0.0,
        attempts=0,
        partial=False,
        error=None,
        meta={},
    )
    stage = StageExecutionResult(
        stage_name="mixed_stage",
        records=(succeeded_record, failed_record, skipped_record),
        success=False,
        duration_ms=MS_TO_SECONDS,
    )
    result = RecipeExecutionResult(
        recipe_name="test",
        run_id="run123",
        stages=(stage,),
        success=False,
        duration_ms=MS_TO_SECONDS,
        started_at="2024-01-01T00:00:00Z",
        ended_at="2024-01-01T00:00:01Z",
    )

    assert result.success_count == EXPECTED_STAGE_COUNT_ONE
    assert result.failure_count == EXPECTED_STAGE_COUNT_ONE
    assert result.skip_count == EXPECTED_STAGE_COUNT_ONE


# ---------------------------------------------------------------------------
# RecipeExecutor Tests
# ---------------------------------------------------------------------------


def test_executor_basic_success(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """RecipeExecutor executes a simple recipe successfully."""
    snapshot = _make_snapshot(tmp_path)

    succeeding_plugin = _make_succeeding_plugin()
    recipe = _make_simple_recipe(plugin_names=(succeeding_plugin.metadata.name,))

    with _PluginRegistrar([succeeding_plugin]):
        context = RecipeExecutorContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            force_sequential=True,
        )
        executor = RecipeExecutor(context)
        result = executor.execute(recipe)

    assert result.success is True
    assert len(result.stages) == EXPECTED_STAGE_COUNT_ONE
    assert result.stages[0].success is True
    assert result.success_count == EXPECTED_STAGE_COUNT_ONE
    assert result.failure_count == 0


def test_executor_plugin_failure_handling(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """RecipeExecutor handles plugin failures."""
    snapshot = _make_snapshot(tmp_path)

    failing_plugin = _make_failing_plugin()
    recipe = _make_simple_recipe(
        plugin_names=(failing_plugin.metadata.name,),
        fail_fast=True,
    )

    with _PluginRegistrar([failing_plugin]):
        context = RecipeExecutorContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            force_sequential=True,
        )
        executor = RecipeExecutor(context)
        result = executor.execute(recipe)

    assert result.success is False
    assert result.failure_count == EXPECTED_STAGE_COUNT_ONE
    assert result.stages[0].success is False


def test_executor_multi_stage_sequential(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """RecipeExecutor executes multiple stages in order."""
    snapshot = _make_snapshot(tmp_path)

    plugin1 = _make_succeeding_plugin(name="test.succeeding.1")
    plugin2 = _make_succeeding_plugin(name="test.succeeding.2")

    recipe = _make_multi_stage_recipe(
        stage_configs=[
            ("stage1", (plugin1.metadata.name,), True),
            ("stage2", (plugin2.metadata.name,), True),
        ]
    )

    with _PluginRegistrar([plugin1, plugin2]):
        context = RecipeExecutorContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            force_sequential=True,
        )
        executor = RecipeExecutor(context)
        result = executor.execute(recipe)

    assert result.success is True
    assert len(result.stages) == EXPECTED_STAGE_COUNT_TWO
    assert result.stages[0].stage_name == "stage1"
    assert result.stages[1].stage_name == "stage2"
    assert result.success_count == EXPECTED_STAGE_COUNT_TWO


def test_executor_fail_fast_stops_on_failure(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """RecipeExecutor stops on failure when fail_fast is True."""
    snapshot = _make_snapshot(tmp_path)

    failing_plugin = _make_failing_plugin(name="test.failing.early")
    succeeding_plugin = _make_succeeding_plugin(name="test.succeeding.later")

    recipe = _make_multi_stage_recipe(
        stage_configs=[
            ("stage1", (failing_plugin.metadata.name,), True),
            ("stage2", (succeeding_plugin.metadata.name,), True),
        ]
    )

    with _PluginRegistrar([failing_plugin, succeeding_plugin]):
        context = RecipeExecutorContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            force_sequential=True,
        )
        executor = RecipeExecutor(context)
        result = executor.execute(recipe)

    assert result.success is False
    # Only first stage should have run
    assert len(result.stages) == EXPECTED_STAGE_COUNT_ONE
    assert result.stages[0].stage_name == "stage1"


def test_executor_missing_plugin_handled(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """RecipeExecutor handles missing plugins gracefully."""
    snapshot = _make_snapshot(tmp_path)

    # Create recipe with non-existent plugin
    recipe = graph_recipe(
        name="missing_plugin_recipe",
        description="Recipe with missing plugin",
        stages=[
            graph_stage(
                name="stage1",
                plugins=["nonexistent.plugin"],
                fail_fast=False,
            )
        ],
    )

    context = RecipeExecutorContext(
        gateway=fresh_gateway,
        snapshot=snapshot,
        force_sequential=True,
    )
    executor = RecipeExecutor(context)
    result = executor.execute(recipe)

    # Should complete but with no successful plugins
    assert len(result.stages) == EXPECTED_STAGE_COUNT_ONE
    assert len(result.stages[0].records) == 0


def test_executor_result_has_timing_info(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """RecipeExecutor result includes timing information."""
    snapshot = _make_snapshot(tmp_path)

    plugin = _make_succeeding_plugin()
    recipe = _make_simple_recipe(plugin_names=(plugin.metadata.name,))

    with _PluginRegistrar([plugin]):
        context = RecipeExecutorContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            force_sequential=True,
        )
        executor = RecipeExecutor(context)
        result = executor.execute(recipe)

    assert result.duration_ms >= 0
    assert result.started_at is not None
    assert result.ended_at is not None
    assert result.run_id is not None


def test_executor_result_has_unique_run_id(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """RecipeExecutor generates unique run IDs."""
    snapshot = _make_snapshot(tmp_path)

    plugin = _make_succeeding_plugin()
    recipe = _make_simple_recipe(plugin_names=(plugin.metadata.name,))

    with _PluginRegistrar([plugin]):
        context = RecipeExecutorContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            force_sequential=True,
        )
        executor = RecipeExecutor(context)
        result1 = executor.execute(recipe)
        result2 = executor.execute(recipe)

    assert result1.run_id != result2.run_id


def test_executor_multiple_plugins_in_stage(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """RecipeExecutor handles multiple plugins in one stage."""
    snapshot = _make_snapshot(tmp_path)

    plugin1 = _make_succeeding_plugin(name="test.multi.1")
    plugin2 = _make_succeeding_plugin(name="test.multi.2")
    plugin3 = _make_succeeding_plugin(name="test.multi.3")

    recipe = graph_recipe(
        name="multi_plugin_recipe",
        description="Recipe with multiple plugins in one stage",
        stages=[
            graph_stage(
                name="stage1",
                plugins=[
                    plugin1.metadata.name,
                    plugin2.metadata.name,
                    plugin3.metadata.name,
                ],
                fail_fast=True,
            )
        ],
    )

    with _PluginRegistrar([plugin1, plugin2, plugin3]):
        context = RecipeExecutorContext(
            gateway=fresh_gateway,
            snapshot=snapshot,
            force_sequential=True,
        )
        executor = RecipeExecutor(context)
        result = executor.execute(recipe)

    assert result.success is True
    assert len(result.stages[0].records) == EXPECTED_PLUGIN_COUNT_THREE
    assert result.success_count == EXPECTED_PLUGIN_COUNT_THREE


# ---------------------------------------------------------------------------
# execute_graph_recipe Function Tests
# ---------------------------------------------------------------------------


def test_execute_graph_recipe_convenience_function(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """execute_graph_recipe convenience function works."""
    snapshot = _make_snapshot(tmp_path)

    plugin = _make_succeeding_plugin()
    recipe = _make_simple_recipe(plugin_names=(plugin.metadata.name,))

    with _PluginRegistrar([plugin]):
        result = execute_graph_recipe(
            recipe,
            gateway=fresh_gateway,
            snapshot=snapshot,
        )

    assert result.success is True
    assert len(result.stages) == EXPECTED_STAGE_COUNT_ONE


# ---------------------------------------------------------------------------
# DSL Tests
# ---------------------------------------------------------------------------


def test_graph_stage_creation() -> None:
    """GraphStage can be created with correct attributes."""
    stage = graph_stage(
        name="test_stage",
        plugins=["plugin1", "plugin2"],
        parallel=True,
        fail_fast=False,
        optional=True,
    )

    assert stage.name == "test_stage"
    assert stage.plugins == ("plugin1", "plugin2")
    assert stage.parallel is True
    assert stage.fail_fast is False
    assert stage.optional is True


def test_graph_stage_defaults() -> None:
    """GraphStage has correct defaults."""
    stage = graph_stage(
        name="test_stage",
        plugins=["plugin1"],
    )

    assert stage.parallel is False
    assert stage.fail_fast is True
    assert stage.optional is False


def test_graph_recipe_creation() -> None:
    """GraphRecipe can be created with correct attributes."""
    stage = graph_stage(name="s1", plugins=["p1", "p2"])
    recipe = graph_recipe(
        name="test_recipe",
        description="Test description",
        stages=[stage],
        version="2.0",
    )

    assert recipe.name == "test_recipe"
    assert recipe.description == "Test description"
    assert len(recipe.stages) == EXPECTED_STAGE_COUNT_ONE
    assert recipe.version == "2.0"


def test_graph_recipe_all_plugins() -> None:
    """GraphRecipe.all_plugins returns all unique plugins."""
    stage1 = graph_stage(name="s1", plugins=["p1", "p2"])
    stage2 = graph_stage(name="s2", plugins=["p2", "p3"])
    recipe = graph_recipe(
        name="test",
        stages=[stage1, stage2],
    )

    all_plugins = recipe.all_plugins
    assert all_plugins == ("p1", "p2", "p3")


def test_graph_recipe_options_defaults() -> None:
    """GraphRecipeOptions has correct defaults."""
    options = GraphRecipeOptions()

    assert options.dry_run is False
    assert options.skip_on_unchanged is False
    max_parallel_default: Final[int] = 4
    assert options.max_parallel == max_parallel_default
    assert options.timeout_ms is None


def test_graph_recipe_with_options() -> None:
    """GraphRecipe can be created with options."""
    max_parallel_value: Final[int] = 8
    timeout_ms_value: Final[int] = 5000
    options = GraphRecipeOptions(
        dry_run=True,
        skip_on_unchanged=True,
        max_parallel=max_parallel_value,
        timeout_ms=timeout_ms_value,
    )
    stage = graph_stage(name="s1", plugins=["p1"])
    recipe = graph_recipe(
        name="test",
        stages=[stage],
        options=options,
    )

    assert recipe.options.dry_run is True
    assert recipe.options.skip_on_unchanged is True
    assert recipe.options.max_parallel == max_parallel_value
    assert recipe.options.timeout_ms == timeout_ms_value


# ---------------------------------------------------------------------------
# Dataclass Frozen Tests
# ---------------------------------------------------------------------------


def test_stage_execution_result_frozen() -> None:
    """StageExecutionResult is frozen."""
    import pytest  # noqa: PLC0415

    result = StageExecutionResult(
        stage_name="test",
        records=(),
        success=True,
        duration_ms=0.0,
    )
    with pytest.raises(AttributeError):
        result.success = False  # type: ignore[misc]


def test_recipe_execution_result_frozen() -> None:
    """RecipeExecutionResult is frozen."""
    import pytest  # noqa: PLC0415

    result = RecipeExecutionResult(
        recipe_name="test",
        run_id="run123",
        stages=(),
        success=True,
        duration_ms=0.0,
        started_at="2024-01-01T00:00:00Z",
        ended_at="2024-01-01T00:00:01Z",
    )
    with pytest.raises(AttributeError):
        result.success = False  # type: ignore[misc]


def test_graph_stage_frozen() -> None:
    """GraphStage is frozen."""
    import pytest  # noqa: PLC0415

    stage = GraphStage(
        name="test",
        plugins=("p1",),
    )
    with pytest.raises(AttributeError):
        stage.name = "changed"  # type: ignore[misc]


def test_graph_recipe_frozen() -> None:
    """GraphRecipe is frozen."""
    import pytest  # noqa: PLC0415

    recipe = GraphRecipe(
        name="test",
        description="Test",
        stages=(),
    )
    with pytest.raises(AttributeError):
        recipe.name = "changed"  # type: ignore[misc]
