"""End-to-end pipeline scenario tests for graph plugins.

This module tests realistic pipeline execution scenarios using the golden
dataset, exercising actual plugin orchestration paths rather than isolated
unit tests.

Scenarios tested:
1. Full pipeline execution with builder and metrics plugins
2. Incremental updates with partial data changes
3. Error recovery when intermediate plugins fail
4. Large dataset stress testing
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.core.context import GraphExecutionContext
from codeintel.graphs.core.protocol import (
    FunctionalGraphPlugin,
    GraphPluginMetadata,
    GraphPluginStage,
)
from codeintel.graphs.core.registry import get_graph_registry, register_graph_plugin
from codeintel.graphs.core.result import GraphPluginResult
from codeintel.graphs.engine import NxGraphEngine
from codeintel.graphs.recipes.dsl import graph_recipe, graph_stage
from codeintel.graphs.recipes.executor import (
    RecipeExecutor,
    RecipeExecutorContext,
    execute_graph_recipe,
)
from codeintel.storage.schemas import apply_all_schemas
from tests._helpers.gateway import open_ingestion_gateway_with_macros
from tests._helpers.seeds.golden_graphs import (
    GOLDEN_COMMIT,
    GOLDEN_MODULE_COUNT,
    GOLDEN_REPO,
    seed_golden_graphs,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_MIN_MODULES: Final = 20
EXPECTED_MIN_CALL_EDGES: Final = 30
EXPECTED_MIN_IMPORT_EDGES: Final = 30
SCENARIO_TIMEOUT_MS: Final = 30000
EXPECTED_STAGE_COUNT: Final = 2
EXPECTED_INCREMENTAL_EXECUTIONS: Final = 2
BUILDER_STAGE: Final[GraphPluginStage] = "structure"
METRIC_STAGE: Final[GraphPluginStage] = "stats"


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def golden_gateway() -> Iterator[StorageGateway]:
    """Provide a gateway seeded with golden graph data.

    Yields
    ------
    StorageGateway
        Gateway with golden dataset seeded.
    """
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    apply_all_schemas(gateway.con)
    seed_golden_graphs(gateway, repo=GOLDEN_REPO, commit=GOLDEN_COMMIT)
    try:
        yield gateway
    finally:
        gateway.close()


@pytest.fixture
def golden_snapshot(tmp_path: Path) -> SnapshotRef:
    """Provide a snapshot reference for the golden dataset.

    Parameters
    ----------
    tmp_path
        Pytest temporary path.

    Returns
    -------
    SnapshotRef
        Snapshot reference.
    """
    return SnapshotRef(repo=GOLDEN_REPO, commit=GOLDEN_COMMIT, repo_root=tmp_path)


def _make_context(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    force_sequential: bool = False,
) -> RecipeExecutorContext:
    """Create an execution context for testing.

    Parameters
    ----------
    gateway
        Storage gateway.
    snapshot
        Snapshot reference.
    force_sequential
        Whether to force sequential execution.

    Returns
    -------
    RecipeExecutorContext
        Configured execution context.
    """
    return RecipeExecutorContext(
        gateway=gateway,
        snapshot=snapshot,
        force_sequential=force_sequential,
    )


# ---------------------------------------------------------------------------
# Test Plugins for Scenarios
# ---------------------------------------------------------------------------


class _CountingPlugin:
    """Plugin that counts invocations for testing execution order."""

    execution_count: int = 0
    last_repo: str | None = None

    @classmethod
    def reset(cls) -> None:
        """Reset the counter."""
        cls.execution_count = 0
        cls.last_repo = None


def _make_counting_plugin(name: str, stage: GraphPluginStage) -> FunctionalGraphPlugin:
    """Create a plugin that increments a counter on execution.

    Parameters
    ----------
    name
        Plugin name.
    stage
        Plugin stage.

    Returns
    -------
    FunctionalGraphPlugin
        Plugin that counts executions.
    """

    def execute(ctx: GraphExecutionContext) -> GraphPluginResult:
        _CountingPlugin.execution_count += 1
        _CountingPlugin.last_repo = ctx.snapshot.repo
        return GraphPluginResult.ok(row_counts={f"test.{name}": 1})

    metadata = GraphPluginMetadata(
        name=name,
        description=f"Test counting plugin: {name}",
        stage=stage,
        kind="builder",
        produces_tables=(f"test.{name}",),
        depends_on=(),
        provides=(name,),
    )

    return FunctionalGraphPlugin(_metadata=metadata, _execute_fn=execute)


def _make_failing_plugin(
    name: str, stage: GraphPluginStage, *, error_msg: str
) -> FunctionalGraphPlugin:
    """Create a plugin that fails with an error.

    Parameters
    ----------
    name
        Plugin name.
    stage
        Plugin stage.
    error_msg
        Error message to return.

    Returns
    -------
    FunctionalGraphPlugin
        Plugin that fails.
    """

    def execute(_ctx: GraphExecutionContext) -> GraphPluginResult:
        return GraphPluginResult.fail(error_msg)

    metadata = GraphPluginMetadata(
        name=name,
        description=f"Test failing plugin: {name}",
        stage=stage,
        kind="builder",
        produces_tables=(),
        depends_on=(),
        provides=(),
    )

    return FunctionalGraphPlugin(_metadata=metadata, _execute_fn=execute)


# ---------------------------------------------------------------------------
# Scenario 1: Full Pipeline Execution
# ---------------------------------------------------------------------------


def test_full_pipeline_with_golden_data(
    golden_gateway: StorageGateway, golden_snapshot: SnapshotRef
) -> None:
    """Execute a full pipeline with builder plugins on golden data.

    This test verifies that plugins can execute in sequence with realistic
    data volumes and produce expected outputs.
    """
    _CountingPlugin.reset()

    # Create test plugins for each stage
    stage1_plugin = _make_counting_plugin("stage1_builder", BUILDER_STAGE)
    stage2_plugin = _make_counting_plugin("stage2_metrics", METRIC_STAGE)

    registry = get_graph_registry()

    # Register plugins
    register_graph_plugin(stage1_plugin)
    register_graph_plugin(stage2_plugin)

    try:
        # Build a recipe with multiple stages
        recipe = graph_recipe(
            name="test_full_pipeline",
            stages=[
                graph_stage(
                    name="builder_stage",
                    plugins=["stage1_builder"],
                    parallel=False,
                ),
                graph_stage(
                    name="metrics_stage",
                    plugins=["stage2_metrics"],
                    parallel=False,
                ),
            ],
        )

        ctx = _make_context(golden_gateway, golden_snapshot)
        executor = RecipeExecutor(ctx)

        # Execute the recipe
        result = executor.execute(recipe)

        # Verify execution
        assert result.success, f"Recipe failed: {result}"
        assert len(result.stages) == EXPECTED_STAGE_COUNT
        assert _CountingPlugin.execution_count == EXPECTED_STAGE_COUNT
        assert _CountingPlugin.last_repo == GOLDEN_REPO

    finally:
        # Cleanup
        registry.unregister("stage1_builder")
        registry.unregister("stage2_metrics")


def test_pipeline_accesses_golden_data(golden_gateway: StorageGateway) -> None:
    """Verify that plugins can access the seeded golden data.

    This test ensures the golden dataset is properly accessible through
    the execution context and storage resources.
    """
    # Query the golden data directly
    module_row = golden_gateway.con.execute(
        "SELECT COUNT(*) FROM core.modules WHERE repo = ? AND commit = ?",
        [GOLDEN_REPO, GOLDEN_COMMIT],
    ).fetchone()
    assert module_row is not None
    module_count = module_row[0]

    call_edge_row = golden_gateway.con.execute(
        "SELECT COUNT(*) FROM graph.call_graph_edges WHERE repo = ? AND commit = ?",
        [GOLDEN_REPO, GOLDEN_COMMIT],
    ).fetchone()
    assert call_edge_row is not None
    call_edge_count = call_edge_row[0]

    import_edge_row = golden_gateway.con.execute(
        "SELECT COUNT(*) FROM graph.import_graph_edges WHERE repo = ? AND commit = ?",
        [GOLDEN_REPO, GOLDEN_COMMIT],
    ).fetchone()
    assert import_edge_row is not None
    import_edge_count = import_edge_row[0]

    # Verify realistic data volumes
    assert module_count >= EXPECTED_MIN_MODULES, (
        f"Expected {EXPECTED_MIN_MODULES}+ modules, got {module_count}"
    )
    assert call_edge_count >= EXPECTED_MIN_CALL_EDGES, (
        f"Expected {EXPECTED_MIN_CALL_EDGES}+ call edges, got {call_edge_count}"
    )
    assert import_edge_count >= EXPECTED_MIN_IMPORT_EDGES, (
        f"Expected {EXPECTED_MIN_IMPORT_EDGES}+ import edges, got {import_edge_count}"
    )


# ---------------------------------------------------------------------------
# Scenario 2: Incremental Updates
# ---------------------------------------------------------------------------


def test_incremental_update_scenario(
    golden_gateway: StorageGateway, golden_snapshot: SnapshotRef
) -> None:
    """Test plugin re-execution after partial data changes.

    This simulates an incremental update where some data has changed
    and plugins need to be re-run.
    """
    _CountingPlugin.reset()

    plugin = _make_counting_plugin("incremental_plugin", BUILDER_STAGE)
    registry = get_graph_registry()
    register_graph_plugin(plugin)

    try:
        recipe = graph_recipe(
            name="incremental_test",
            stages=[graph_stage(name="update", plugins=["incremental_plugin"], parallel=False)],
        )

        ctx = _make_context(golden_gateway, golden_snapshot)
        executor = RecipeExecutor(ctx)

        # First execution
        result1 = executor.execute(recipe)
        assert result1.success
        assert _CountingPlugin.execution_count == 1

        # Simulate data change by adding a new module
        golden_gateway.con.execute(
            """
            INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "new.module",
                "new/module.py",
                GOLDEN_REPO,
                GOLDEN_COMMIT,
                "python",
                "[]",
                "[]",
            ],
        )

        # Re-execute after change
        result2 = executor.execute(recipe)
        assert result2.success
        assert _CountingPlugin.execution_count == EXPECTED_INCREMENTAL_EXECUTIONS

    finally:
        registry.unregister("incremental_plugin")


# ---------------------------------------------------------------------------
# Scenario 3: Error Recovery
# ---------------------------------------------------------------------------


def test_error_recovery_continues_other_plugins(
    golden_gateway: StorageGateway, golden_snapshot: SnapshotRef
) -> None:
    """Test that pipeline continues after non-fatal plugin failure.

    When one plugin fails, other plugins in parallel should still execute.
    """
    _CountingPlugin.reset()

    success_plugin = _make_counting_plugin("success_plugin", BUILDER_STAGE)
    failing_plugin = _make_failing_plugin(
        "failing_plugin", BUILDER_STAGE, error_msg="Intentional test failure"
    )

    registry = get_graph_registry()
    register_graph_plugin(success_plugin)
    register_graph_plugin(failing_plugin)

    try:
        # Stage with both plugins (parallel execution)
        recipe = graph_recipe(
            name="error_recovery_test",
            stages=[
                graph_stage(
                    name="mixed_stage",
                    plugins=["success_plugin", "failing_plugin"],
                    parallel=True,
                )
            ],
        )

        ctx = _make_context(golden_gateway, golden_snapshot)
        executor = RecipeExecutor(ctx)

        result = executor.execute(recipe)

        # Recipe should complete (not crash)
        assert len(result.stages) == 1

        # Check individual plugin results
        stage_result = result.stages[0]

        success_records = [r for r in stage_result.records if r.status == "succeeded"]
        failed_records = [r for r in stage_result.records if r.status == "failed"]

        # The successful plugin should have run
        assert len(success_records) >= 1
        # The failing plugin should have recorded failure
        assert len(failed_records) >= 1

    finally:
        registry.unregister("success_plugin")
        registry.unregister("failing_plugin")


def test_fatal_error_stops_pipeline(
    golden_gateway: StorageGateway, golden_snapshot: SnapshotRef
) -> None:
    """Test that fatal errors properly stop pipeline execution."""
    _CountingPlugin.reset()

    # Create a plugin that raises an exception
    def raise_exception(_ctx: GraphExecutionContext) -> GraphPluginResult:
        msg = "Fatal test error"
        raise RuntimeError(msg)

    metadata = GraphPluginMetadata(
        name="fatal_plugin",
        description="Test fatal plugin that raises an exception",
        stage=BUILDER_STAGE,
        kind="builder",
        produces_tables=(),
        depends_on=(),
        provides=(),
    )

    fatal_plugin = FunctionalGraphPlugin(_metadata=metadata, _execute_fn=raise_exception)
    after_plugin = _make_counting_plugin("after_fatal", BUILDER_STAGE)

    registry = get_graph_registry()
    register_graph_plugin(fatal_plugin)
    register_graph_plugin(after_plugin)

    try:
        recipe = graph_recipe(
            name="fatal_error_test",
            stages=[
                graph_stage(name="stage1", plugins=["fatal_plugin"], parallel=False),
                graph_stage(name="stage2", plugins=["after_fatal"], parallel=False),
            ],
        )

        ctx = _make_context(golden_gateway, golden_snapshot)
        executor = RecipeExecutor(ctx)

        # Execute - should handle the error
        result = executor.execute(recipe)

        # The recipe should not be marked as success
        assert not result.success

    finally:
        registry.unregister("fatal_plugin")
        registry.unregister("after_fatal")


# ---------------------------------------------------------------------------
# Scenario 4: Large Dataset Stress Test
# ---------------------------------------------------------------------------


def _insert_stress_modules(gateway: StorageGateway, *, count: int) -> None:
    """Seed additional modules for stress testing."""
    for i in range(count):
        gateway.con.execute(
            """
            INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                f"stress.module_{i}",
                f"stress/module_{i}.py",
                GOLDEN_REPO,
                GOLDEN_COMMIT,
                "python",
                "[]",
                "[]",
            ],
        )


def _insert_stress_edges(gateway: StorageGateway, *, count: int) -> None:
    """Seed additional call graph edges for stress testing."""
    for i in range(count):
        caller_goid = 1000 + (i % 50)
        callee_goid = 1000 + ((i + 10) % 50)
        gateway.con.execute(
            """
            INSERT INTO graph.call_graph_edges (
                repo, commit, caller_goid_h128, callee_goid_h128,
                callsite_path, callsite_line, callsite_col,
                language, kind, resolved_via, confidence, evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                GOLDEN_REPO,
                GOLDEN_COMMIT,
                caller_goid,
                callee_goid,
                f"stress/module_{i % 100}.py",
                10 + i,
                4,
                "python",
                "direct",
                "local_name",
                0.9,
                "{}",
            ],
        )


def _fetch_repo_counts(gateway: StorageGateway) -> tuple[int, int]:
    """Return module and call-edge counts for the golden repo.

    Parameters
    ----------
    gateway
        Storage gateway with seeded data.

    Returns
    -------
    tuple[int, int]
        Module count and call edge count for the golden repository.

    Raises
    ------
    RuntimeError
        If the counts cannot be retrieved.
    """
    modules_row = gateway.con.execute(
        "SELECT COUNT(*) FROM core.modules WHERE repo = ?",
        [GOLDEN_REPO],
    ).fetchone()
    edges_row = gateway.con.execute(
        "SELECT COUNT(*) FROM graph.call_graph_edges WHERE repo = ?",
        [GOLDEN_REPO],
    ).fetchone()
    if modules_row is None or edges_row is None:
        message = "Failed to fetch stress test counts"
        raise RuntimeError(message)
    return modules_row[0], edges_row[0]


def test_large_dataset_execution(
    golden_gateway: StorageGateway, golden_snapshot: SnapshotRef
) -> None:
    """Test plugin execution with a larger dataset volume."""
    additional_modules = 100
    additional_edges = 200

    _insert_stress_modules(golden_gateway, count=additional_modules)
    _insert_stress_edges(golden_gateway, count=additional_edges)

    total_modules, total_edges = _fetch_repo_counts(golden_gateway)

    expected_module_min = GOLDEN_MODULE_COUNT + additional_modules
    expected_edge_min = EXPECTED_MIN_CALL_EDGES + additional_edges

    assert total_modules >= expected_module_min
    assert total_edges >= expected_edge_min

    # Run a plugin on the large dataset
    _CountingPlugin.reset()
    plugin = _make_counting_plugin("large_data_plugin", BUILDER_STAGE)
    registry = get_graph_registry()
    register_graph_plugin(plugin)

    try:
        recipe = graph_recipe(
            name="large_data_test",
            stages=[graph_stage(name="process", plugins=["large_data_plugin"], parallel=False)],
        )

        ctx = _make_context(golden_gateway, golden_snapshot)
        executor = RecipeExecutor(ctx)

        result = executor.execute(recipe)
        assert result.success
        assert _CountingPlugin.execution_count == 1

    finally:
        registry.unregister("large_data_plugin")


# ---------------------------------------------------------------------------
# Scenario 5: Graph Engine Integration
# ---------------------------------------------------------------------------


def test_graph_engine_with_golden_data(
    golden_gateway: StorageGateway, golden_snapshot: SnapshotRef
) -> None:
    """Test that NxGraphEngine correctly loads golden graph data.

    This verifies the full integration path from seeded data through
    the graph engine to NetworkX graphs.
    """
    engine = NxGraphEngine(gateway=golden_gateway, snapshot=golden_snapshot)

    # Load call graph
    call_graph = engine.call_graph()
    assert call_graph.number_of_nodes() > 0
    assert call_graph.number_of_edges() >= EXPECTED_MIN_CALL_EDGES

    # Load import graph
    import_graph = engine.import_graph()
    assert import_graph.number_of_nodes() > 0
    assert import_graph.number_of_edges() >= EXPECTED_MIN_IMPORT_EDGES

    # Verify graph caching works
    call_graph_2 = engine.call_graph()
    assert call_graph_2 is call_graph  # Same object (cached)


def test_convenience_execute_function(
    golden_gateway: StorageGateway, golden_snapshot: SnapshotRef
) -> None:
    """Test the execute_graph_recipe convenience function."""
    _CountingPlugin.reset()

    plugin = _make_counting_plugin("convenience_plugin", BUILDER_STAGE)
    registry = get_graph_registry()
    register_graph_plugin(plugin)

    try:
        recipe = graph_recipe(
            name="convenience_test",
            stages=[graph_stage(name="run", plugins=["convenience_plugin"], parallel=False)],
        )

        result = execute_graph_recipe(
            recipe=recipe,
            gateway=golden_gateway,
            snapshot=golden_snapshot,
        )

        assert result.success
        assert _CountingPlugin.execution_count == 1

    finally:
        registry.unregister("convenience_plugin")
