"""Unit tests for the build executor module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from codeintel.build.executor import (
    BuildExecutor,
    BuildResult,
    ExecutorEnv,
    StageExecutionResult,
)
from codeintel.build.manifest import BuildRunRecord, OutputManifest
from codeintel.build.plan import BuildPlan, PlanGenerator, PlanStage, PlanStep
from codeintel.build.registry import get_target_graph
from codeintel.build.resolver import ResolutionResult
from codeintel.build.targets import OutputTarget, TargetGraph, TargetModule, TargetOptions
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import expect_equal, expect_in, expect_length, expect_true

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

    modules_target = OutputTarget.from_tables(
        name="modules",
        module="ingestion",
        plugin="repo_scan",
        tables=("core.modules",),
        options=TargetOptions(description="Repository module index"),
    )

    ast_target = OutputTarget.from_tables(
        name="ast",
        module="ingestion",
        plugin="ast_extract",
        tables=("core.ast_nodes",),
        options=TargetOptions(dependencies=("modules",), description="AST extraction"),
    )

    goids_target = OutputTarget.from_tables(
        name="goids",
        module="graphs",
        plugin="goid_builder",
        tables=("core.goids",),
        options=TargetOptions(dependencies=("ast",), description="GOID construction"),
    )

    metrics_target = OutputTarget.from_tables(
        name="function_metrics",
        module="analytics",
        plugin="function_metrics",
        tables=("analytics.function_metrics",),
        options=TargetOptions(dependencies=("goids",), description="Function metrics"),
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


def _make_env(
    gateway: FakeStorageGateway,
    snapshot: SnapshotRef,
    paths: BuildPaths,
    tools: ToolsConfig,
) -> ExecutorEnv:
    """Construct executor env from test fixtures.

    Returns
    -------
    ExecutorEnv
        Bundled execution environment for BuildExecutor.
    """
    return ExecutorEnv(
        gateway=cast("StorageGateway", gateway),
        snapshot=snapshot,
        paths=paths,
        tools=tools,
    )


def _make_executor(
    graph: TargetGraph,
    env: ExecutorEnv,
    *,
    fail_fast: bool = False,
) -> BuildExecutor:
    """Build executor with bundled env for tests.

    Returns
    -------
    BuildExecutor
        Executor configured for the provided graph and environment.
    """
    return BuildExecutor(graph=graph, env=env, fail_fast=fail_fast)


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

    @staticmethod
    def test_create_result() -> None:
        """Create a stage execution result."""
        result = StageExecutionResult(
            module="analytics",
            completed=("function_metrics", "hotspots"),
            failed=(),
            durations_ms={"function_metrics": 5000, "hotspots": 2000},
            row_counts={"function_metrics": 1500},
        )
        expect_equal(result.module, "analytics")
        expect_length(result.completed, 2)
        expect_length(result.failed, 0)

    @staticmethod
    def test_success_true() -> None:
        """Success returns True when no failures."""
        result = StageExecutionResult(
            module="ingestion",
            completed=("modules",),
            failed=(),
        )
        expect_true(result.success is True)

    @staticmethod
    def test_success_false_with_failed() -> None:
        """Success returns False when targets failed."""
        result = StageExecutionResult(
            module="ingestion",
            completed=(),
            failed=("modules",),
        )
        expect_true(result.success is False)

    @staticmethod
    def test_success_false_with_error() -> None:
        """Success returns False when error occurred."""
        result = StageExecutionResult(
            module="ingestion",
            completed=(),
            failed=(),
            error="Something went wrong",
        )
        expect_true(result.success is False)


# =============================================================================
# BuildResult Tests
# =============================================================================


class TestBuildResult:
    """Tests for BuildResult dataclass."""

    @staticmethod
    def test_create_result() -> None:
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
        expect_equal(result.run_id, "build-123")
        expect_equal(result.status, "succeeded")
        expect_length(result.completed_targets, 2)

    @staticmethod
    def test_success_true() -> None:
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
        expect_true(result.success is True)

    @staticmethod
    def test_success_false() -> None:
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
        expect_true(result.success is False)

    @staticmethod
    def test_to_dict() -> None:
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

        expect_equal(data["run_id"], "build-123")
        expect_equal(data["status"], "succeeded")
        expect_equal(data["completed_targets"], ["modules"])
        expect_equal(data["skipped_targets"], ["goids"])
        expect_equal(data["duration_ms"], 5000)
        expect_in("plan", data)


# =============================================================================
# BuildExecutor Tests
# =============================================================================


class TestBuildExecutorInit:
    """Tests for BuildExecutor initialization."""

    @staticmethod
    def test_init(
        executor_graph: TargetGraph,
        fake_gateway: FakeStorageGateway,
        test_snapshot: SnapshotRef,
        test_paths: BuildPaths,
        test_tools: ToolsConfig,
    ) -> None:
        """Create a build executor."""
        env = _make_env(fake_gateway, test_snapshot, test_paths, test_tools)
        executor = _make_executor(graph=executor_graph, env=env)
        # Verify executor was created (access public behavior)
        plan = _make_plan(requested=(), stages=())
        result = executor.execute(plan, dry_run=True)
        expect_true(result.success is True)


class TestBuildExecutorRunId:
    """Tests for run ID generation."""

    @staticmethod
    def test_run_id_format(
        executor_graph: TargetGraph,
        fake_gateway: FakeStorageGateway,
        test_snapshot: SnapshotRef,
        test_paths: BuildPaths,
        test_tools: ToolsConfig,
    ) -> None:
        """Run ID has expected format."""
        env = _make_env(fake_gateway, test_snapshot, test_paths, test_tools)
        executor = _make_executor(graph=executor_graph, env=env)
        plan = _make_plan(requested=(), stages=())
        result = executor.execute(plan, dry_run=True)

        run_id = result.run_id
        expect_true(run_id.startswith("build-"))
        parts = run_id.split("-")
        expect_length(parts, 4)
        expect_length(parts[1], 8)  # YYYYMMDD
        expect_length(parts[2], 6)  # HHMMSS
        expect_length(parts[3], 8)  # hex suffix

    @staticmethod
    def test_run_ids_unique(
        executor_graph: TargetGraph,
        fake_gateway: FakeStorageGateway,
        test_snapshot: SnapshotRef,
        test_paths: BuildPaths,
        test_tools: ToolsConfig,
    ) -> None:
        """Run IDs are unique across executions."""
        env = _make_env(fake_gateway, test_snapshot, test_paths, test_tools)
        executor = _make_executor(graph=executor_graph, env=env)
        plan = _make_plan(requested=(), stages=())

        run_ids = [executor.execute(plan, dry_run=True).run_id for _ in range(10)]
        expect_length(set(run_ids), 10)


class TestBuildExecutorEmptyPlan:
    """Tests for executing empty plans."""

    @staticmethod
    def test_execute_empty_plan(
        executor_graph: TargetGraph,
        fake_gateway: FakeStorageGateway,
        test_snapshot: SnapshotRef,
        test_paths: BuildPaths,
        test_tools: ToolsConfig,
    ) -> None:
        """Empty plan returns immediately with success."""
        env = _make_env(fake_gateway, test_snapshot, test_paths, test_tools)
        executor = _make_executor(graph=executor_graph, env=env)

        plan = _make_plan(
            requested=("function_metrics",),
            stages=(),
            skipped=("function_metrics",),
        )

        result = executor.execute(plan)

        expect_true(result.success is True)
        expect_equal(result.completed_targets, ())
        expect_equal(result.failed_targets, ())
        expect_equal(result.skipped_targets, ("function_metrics",))


class TestBuildExecutorDryRun:
    """Tests for dry run execution."""

    @staticmethod
    def test_execute_dry_run(
        executor_graph: TargetGraph,
        fake_gateway: FakeStorageGateway,
        test_snapshot: SnapshotRef,
        test_paths: BuildPaths,
        test_tools: ToolsConfig,
    ) -> None:
        """Dry run returns plan info without executing."""
        env = _make_env(fake_gateway, test_snapshot, test_paths, test_tools)
        executor = _make_executor(graph=executor_graph, env=env)

        step = _make_step("modules", "ingestion", "repo_scan")
        stage = _make_stage("ingestion", (step,))
        plan = _make_plan(
            requested=("modules",),
            stages=(stage,),
        )

        result = executor.execute(plan, dry_run=True)

        expect_true(result.success is True)
        expect_equal(result.completed_targets, ())  # Nothing actually computed
        expect_equal(result.status, "succeeded")

    @staticmethod
    def test_dry_run_records_tracking(
        executor_graph: TargetGraph,
        fake_gateway: FakeStorageGateway,
        test_snapshot: SnapshotRef,
        test_paths: BuildPaths,
        test_tools: ToolsConfig,
    ) -> None:
        """Dry run still records run tracking."""
        env = _make_env(fake_gateway, test_snapshot, test_paths, test_tools)
        executor = _make_executor(graph=executor_graph, env=env)

        plan = _make_plan(requested=(), stages=())
        executor.execute(plan, dry_run=True)

        # Should have recorded a run
        expect_length(fake_gateway.build.runs, 1)


# =============================================================================
# Integration Tests
# =============================================================================


class TestBuildExecutorIntegration:
    """Integration tests for BuildExecutor with real registry."""

    @staticmethod
    def test_with_real_registry() -> None:
        """BuildExecutor works with real target registry."""
        graph = get_target_graph()
        fake_gw = FakeStorageGateway()
        snapshot = _make_snapshot()
        paths = _make_paths()
        tools = ToolsConfig.default()

        env = _make_env(fake_gw, snapshot, paths, tools)
        executor = _make_executor(graph=graph, env=env)

        # Empty plan should work
        plan = _make_plan(requested=(), stages=())
        result = executor.execute(plan)

        expect_true(result.success is True)

    @staticmethod
    def test_plan_generator_to_executor() -> None:
        """Plan from PlanGenerator works with executor."""
        graph = _create_test_graph()
        fake_gw = FakeStorageGateway()
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
        env = _make_env(fake_gw, snapshot, paths, tools)
        executor = _make_executor(graph=graph, env=env)
        result = executor.execute(plan)

        expect_true(result.success is True)
        expect_equal(result.skipped_targets, ("function_metrics",))
