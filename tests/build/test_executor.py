"""Unit tests for the build executor module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.build.executor import (
    BuildExecutor,
    BuildResult,
    StageExecutionResult,
)
from codeintel.build.manifest import BuildRunRecord, OutputManifest
from codeintel.build.plan import BuildPlan, PlanGenerator, PlanStage, PlanStep
from codeintel.build.registry import get_target_graph
from codeintel.build.resolver import ResolutionResult
from codeintel.build.targets import OutputTarget, TargetGraph, TargetModule
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# =============================================================================
# Test Fixtures
# =============================================================================


def _create_test_graph() -> TargetGraph:
    """Create a minimal test graph for executor tests.

    Returns
    -------
    TargetGraph
        Graph with modules -> ast -> function_metrics chain.
    """
    graph = TargetGraph()

    modules_target = OutputTarget(
        name="modules",
        module="ingestion",
        plugin="repo_scan",
        tables=("core.modules",),
        dependencies=(),
        description="Repository module index",
    )

    ast_target = OutputTarget(
        name="ast",
        module="ingestion",
        plugin="ast_extract",
        tables=("core.ast_nodes",),
        dependencies=("modules",),
        description="AST extraction",
    )

    goids_target = OutputTarget(
        name="goids",
        module="graphs",
        plugin="goid_builder",
        tables=("core.goids",),
        dependencies=("ast",),
        description="GOID construction",
    )

    metrics_target = OutputTarget(
        name="function_metrics",
        module="analytics",
        plugin="function_metrics",
        tables=("analytics.function_metrics",),
        dependencies=("goids",),
        description="Function metrics",
    )

    graph.register(modules_target)
    graph.register(ast_target)
    graph.register(goids_target)
    graph.register(metrics_target)

    return graph


def _make_snapshot() -> SnapshotRef:
    """Create a test snapshot reference.

    Returns
    -------
    SnapshotRef
        Test snapshot for testing.
    """
    return SnapshotRef(
        repo="test-org/test-repo",
        commit="abc123",
        repo_root=Path.cwd(),
    )


def _make_paths() -> BuildPaths:
    """Create test build paths.

    Returns
    -------
    BuildPaths
        Test paths for testing.
    """
    return BuildPaths.from_repo_root(Path.cwd())


@dataclass
class FakeBuildTracking:
    """Fake build tracking accessor for testing.

    Implements the minimal interface required by BuildExecutor:
    - start_run(record)
    - complete_run(run_id, status, computed_targets, skipped_targets, error_summary)
    - save_manifest(manifest)
    - load_manifest(target, repo, commit)
    """

    manifests: dict[str, OutputManifest] = field(default_factory=dict)
    runs: dict[str, BuildRunRecord] = field(default_factory=dict)

    def save_manifest(self, manifest: OutputManifest) -> None:
        """Save a manifest.

        Parameters
        ----------
        manifest
            Manifest to save.
        """
        self.manifests[manifest.target] = manifest

    def load_manifest(
        self,
        target: str,
        repo: str,
        commit: str,
    ) -> OutputManifest | None:
        """Load a manifest.

        Parameters
        ----------
        target
            Target name.
        repo
            Repository.
        commit
            Commit.

        Returns
        -------
        OutputManifest | None
            Manifest if found.
        """
        # Use all parameters to avoid unused warnings
        _ = (repo, commit)
        return self.manifests.get(target)

    def start_run(self, record: BuildRunRecord) -> None:
        """Start a run.

        Parameters
        ----------
        record
            Run record to start.
        """
        self.runs[record.run_id] = record

    def complete_run(
        self,
        run_id: str,
        status: str,
        computed_targets: tuple[str, ...],
        skipped_targets: tuple[str, ...],
        error_summary: str | None = None,
    ) -> None:
        """Complete a run.

        Parameters
        ----------
        run_id
            Run identifier.
        status
            Final status.
        computed_targets
            Computed targets.
        skipped_targets
            Skipped targets.
        error_summary
            Error summary (optional).
        """
        # Store the completion data in the run record
        if run_id in self.runs:
            # In a real impl, we'd update the record
            _ = (status, computed_targets, skipped_targets, error_summary)


@dataclass
class FakeStorageGateway:
    """Fake storage gateway for testing BuildExecutor.

    This fake implements the minimal interface required by BuildExecutor,
    specifically the .build attribute for build tracking.
    """

    build: FakeBuildTracking = field(default_factory=FakeBuildTracking)


@pytest.fixture
def executor_graph() -> TargetGraph:
    """Provide the test graph for executor tests.

    Returns
    -------
    TargetGraph
        Test graph instance.
    """
    return _create_test_graph()


@pytest.fixture
def fake_gateway() -> FakeStorageGateway:
    """Provide a fake StorageGateway.

    Returns
    -------
    FakeStorageGateway
        Fake gateway with build tracking.
    """
    return FakeStorageGateway()


@pytest.fixture
def test_snapshot() -> SnapshotRef:
    """Provide a test snapshot reference.

    Returns
    -------
    SnapshotRef
        Test snapshot.
    """
    return _make_snapshot()


@pytest.fixture
def test_paths() -> BuildPaths:
    """Provide test build paths.

    Returns
    -------
    BuildPaths
        Test paths.
    """
    return _make_paths()


@pytest.fixture
def test_tools() -> ToolsConfig:
    """Provide test tools config.

    Returns
    -------
    ToolsConfig
        Default tools config for testing.
    """
    return ToolsConfig.default()


def _make_plan(
    requested: tuple[str, ...],
    stages: tuple[PlanStage, ...],
    skipped: tuple[str, ...] = (),
    blocked: tuple[str, ...] = (),
) -> BuildPlan:
    """Create a BuildPlan for testing.

    Returns
    -------
    BuildPlan
        Test plan.
    """
    return BuildPlan(
        requested_targets=requested,
        stages=stages,
        skipped_targets=skipped,
        blocked_targets=blocked,
    )


def _make_stage(
    module: TargetModule,
    steps: tuple[PlanStep, ...],
) -> PlanStage:
    """Create a PlanStage for testing.

    Returns
    -------
    PlanStage
        Test stage.
    """
    return PlanStage(module=module, steps=steps)


def _make_step(
    target: str,
    module: TargetModule,
    plugin: str,
) -> PlanStep:
    """Create a PlanStep for testing.

    Returns
    -------
    PlanStep
        Test step.
    """
    return PlanStep(
        target=target,
        module=module,
        plugin=plugin,
        estimated_duration_ms=1000,
        dependencies=(),
        reason="missing",
    )


# =============================================================================
# StageExecutionResult Tests
# =============================================================================


class TestStageExecutionResult:
    """Tests for StageExecutionResult dataclass."""

    def test_create_result(self) -> None:
        """Create a stage execution result."""
        result = StageExecutionResult(
            module="analytics",
            completed=("function_metrics", "hotspots"),
            failed=(),
            durations_ms={"function_metrics": 5000, "hotspots": 2000},
            row_counts={"function_metrics": 1500},
        )
        assert result.module == "analytics"
        assert len(result.completed) == 2
        assert len(result.failed) == 0

    def test_success_true(self) -> None:
        """Success returns True when no failures."""
        result = StageExecutionResult(
            module="ingestion",
            completed=("modules",),
            failed=(),
        )
        assert result.success is True

    def test_success_false_with_failed(self) -> None:
        """Success returns False when targets failed."""
        result = StageExecutionResult(
            module="ingestion",
            completed=(),
            failed=("modules",),
        )
        assert result.success is False

    def test_success_false_with_error(self) -> None:
        """Success returns False when error occurred."""
        result = StageExecutionResult(
            module="ingestion",
            completed=(),
            failed=(),
            error="Something went wrong",
        )
        assert result.success is False


# =============================================================================
# BuildResult Tests
# =============================================================================


class TestBuildResult:
    """Tests for BuildResult dataclass."""

    def test_create_result(self) -> None:
        """Create a build result."""
        plan = _make_plan(
            requested=("function_metrics",),
            stages=(),
        )
        result = BuildResult(
            run_id="build-123",
            plan=plan,
            status="succeeded",
            completed_targets=("modules", "ast"),
            failed_targets=(),
            skipped_targets=(),
            duration_ms=10000,
        )
        assert result.run_id == "build-123"
        assert result.status == "succeeded"
        assert len(result.completed_targets) == 2

    def test_success_true(self) -> None:
        """Success returns True when status is succeeded."""
        plan = _make_plan(requested=(), stages=())
        result = BuildResult(
            run_id="build-123",
            plan=plan,
            status="succeeded",
            completed_targets=(),
            failed_targets=(),
            skipped_targets=(),
            duration_ms=100,
        )
        assert result.success is True

    def test_success_false(self) -> None:
        """Success returns False when status is failed."""
        plan = _make_plan(requested=(), stages=())
        result = BuildResult(
            run_id="build-123",
            plan=plan,
            status="failed",
            completed_targets=(),
            failed_targets=("modules",),
            skipped_targets=(),
            duration_ms=100,
            error_summary="Build failed",
        )
        assert result.success is False

    def test_to_dict(self) -> None:
        """Result serializes correctly."""
        plan = _make_plan(
            requested=("function_metrics",),
            stages=(),
            skipped=("goids",),
        )
        result = BuildResult(
            run_id="build-123",
            plan=plan,
            status="succeeded",
            completed_targets=("modules",),
            failed_targets=(),
            skipped_targets=("goids",),
            duration_ms=5000,
        )
        data = result.to_dict()

        assert data["run_id"] == "build-123"
        assert data["status"] == "succeeded"
        assert data["completed_targets"] == ["modules"]
        assert data["skipped_targets"] == ["goids"]
        assert data["duration_ms"] == 5000
        assert "plan" in data


# =============================================================================
# BuildExecutor Tests
# =============================================================================


class TestBuildExecutorInit:
    """Tests for BuildExecutor initialization."""

    def test_init(
        self,
        executor_graph: TargetGraph,
        fake_gateway: FakeStorageGateway,
        test_snapshot: SnapshotRef,
        test_paths: BuildPaths,
        test_tools: ToolsConfig,
    ) -> None:
        """Create a build executor."""
        # Cast to satisfy type checker - FakeStorageGateway is duck-typed
        gateway: StorageGateway = fake_gateway  # type: ignore[assignment]
        executor = BuildExecutor(
            graph=executor_graph,
            gateway=gateway,
            snapshot=test_snapshot,
            paths=test_paths,
            tools=test_tools,
        )
        # Verify executor was created (access public behavior)
        plan = _make_plan(requested=(), stages=())
        result = executor.execute(plan, dry_run=True)
        assert result.success is True


class TestBuildExecutorRunId:
    """Tests for run ID generation."""

    def test_run_id_format(
        self,
        executor_graph: TargetGraph,
        fake_gateway: FakeStorageGateway,
        test_snapshot: SnapshotRef,
        test_paths: BuildPaths,
        test_tools: ToolsConfig,
    ) -> None:
        """Run ID has expected format."""
        gateway: StorageGateway = fake_gateway  # type: ignore[assignment]
        executor = BuildExecutor(
            graph=executor_graph,
            gateway=gateway,
            snapshot=test_snapshot,
            paths=test_paths,
            tools=test_tools,
        )
        plan = _make_plan(requested=(), stages=())
        result = executor.execute(plan, dry_run=True)

        run_id = result.run_id
        assert run_id.startswith("build-")
        parts = run_id.split("-")
        assert len(parts) == 4
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
        assert len(parts[3]) == 8  # hex suffix

    def test_run_ids_unique(
        self,
        executor_graph: TargetGraph,
        fake_gateway: FakeStorageGateway,
        test_snapshot: SnapshotRef,
        test_paths: BuildPaths,
        test_tools: ToolsConfig,
    ) -> None:
        """Run IDs are unique across executions."""
        gateway: StorageGateway = fake_gateway  # type: ignore[assignment]
        executor = BuildExecutor(
            graph=executor_graph,
            gateway=gateway,
            snapshot=test_snapshot,
            paths=test_paths,
            tools=test_tools,
        )
        plan = _make_plan(requested=(), stages=())

        run_ids = [executor.execute(plan, dry_run=True).run_id for _ in range(10)]
        assert len(set(run_ids)) == 10


class TestBuildExecutorEmptyPlan:
    """Tests for executing empty plans."""

    def test_execute_empty_plan(
        self,
        executor_graph: TargetGraph,
        fake_gateway: FakeStorageGateway,
        test_snapshot: SnapshotRef,
        test_paths: BuildPaths,
        test_tools: ToolsConfig,
    ) -> None:
        """Empty plan returns immediately with success."""
        gateway: StorageGateway = fake_gateway  # type: ignore[assignment]
        executor = BuildExecutor(
            graph=executor_graph,
            gateway=gateway,
            snapshot=test_snapshot,
            paths=test_paths,
            tools=test_tools,
        )

        plan = _make_plan(
            requested=("function_metrics",),
            stages=(),
            skipped=("function_metrics",),
        )

        result = executor.execute(plan)

        assert result.success is True
        assert result.completed_targets == ()
        assert result.failed_targets == ()
        assert result.skipped_targets == ("function_metrics",)


class TestBuildExecutorDryRun:
    """Tests for dry run execution."""

    def test_execute_dry_run(
        self,
        executor_graph: TargetGraph,
        fake_gateway: FakeStorageGateway,
        test_snapshot: SnapshotRef,
        test_paths: BuildPaths,
        test_tools: ToolsConfig,
    ) -> None:
        """Dry run returns plan info without executing."""
        gateway: StorageGateway = fake_gateway  # type: ignore[assignment]
        executor = BuildExecutor(
            graph=executor_graph,
            gateway=gateway,
            snapshot=test_snapshot,
            paths=test_paths,
            tools=test_tools,
        )

        step = _make_step("modules", "ingestion", "repo_scan")
        stage = _make_stage("ingestion", (step,))
        plan = _make_plan(
            requested=("modules",),
            stages=(stage,),
        )

        result = executor.execute(plan, dry_run=True)

        assert result.success is True
        assert result.completed_targets == ()  # Nothing actually computed
        assert result.status == "succeeded"

    def test_dry_run_records_tracking(
        self,
        executor_graph: TargetGraph,
        fake_gateway: FakeStorageGateway,
        test_snapshot: SnapshotRef,
        test_paths: BuildPaths,
        test_tools: ToolsConfig,
    ) -> None:
        """Dry run still records run tracking."""
        gateway: StorageGateway = fake_gateway  # type: ignore[assignment]
        executor = BuildExecutor(
            graph=executor_graph,
            gateway=gateway,
            snapshot=test_snapshot,
            paths=test_paths,
            tools=test_tools,
        )

        plan = _make_plan(requested=(), stages=())
        executor.execute(plan, dry_run=True)

        # Should have recorded a run
        assert len(fake_gateway.build.runs) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestBuildExecutorIntegration:
    """Integration tests for BuildExecutor with real registry."""

    def test_with_real_registry(self) -> None:
        """BuildExecutor works with real target registry."""
        graph = get_target_graph()
        fake_gw = FakeStorageGateway()
        gateway: StorageGateway = fake_gw  # type: ignore[assignment]
        snapshot = _make_snapshot()
        paths = _make_paths()
        tools = ToolsConfig.default()

        executor = BuildExecutor(
            graph=graph,
            gateway=gateway,
            snapshot=snapshot,
            paths=paths,
            tools=tools,
        )

        # Empty plan should work
        plan = _make_plan(requested=(), stages=())
        result = executor.execute(plan)

        assert result.success is True

    def test_plan_generator_to_executor(self) -> None:
        """Plan from PlanGenerator works with executor."""
        graph = _create_test_graph()
        fake_gw = FakeStorageGateway()
        gateway: StorageGateway = fake_gw  # type: ignore[assignment]
        snapshot = _make_snapshot()
        paths = _make_paths()
        tools = ToolsConfig.default()

        # Create a resolution result
        resolution = ResolutionResult(
            requested=("function_metrics",),
            to_compute=(),  # Empty - nothing to compute
            to_skip=("function_metrics",),
            blocked=(),
            reasons={},
        )

        # Generate plan
        generator = PlanGenerator(graph)
        plan = generator.generate(resolution)

        # Execute
        executor = BuildExecutor(
            graph=graph,
            gateway=gateway,
            snapshot=snapshot,
            paths=paths,
            tools=tools,
        )
        result = executor.execute(plan)

        assert result.success is True
        assert result.skipped_targets == ("function_metrics",)
