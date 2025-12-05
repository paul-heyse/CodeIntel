"""Unit tests for the plan generation module."""

from __future__ import annotations

import json

import pytest

from codeintel.core.build.plan import (
    BuildPlan,
    PlanGenerator,
    PlanStage,
    PlanStep,
    format_duration,
)
from codeintel.core.build.registry import get_target_graph
from codeintel.core.build.resolver import ResolutionReason, ResolutionResult
from codeintel.core.build.targets import OutputTarget, TargetGraph
from tests._helpers import assert_frozen

# =============================================================================
# Test Fixtures
# =============================================================================


def _create_test_graph() -> TargetGraph:
    r"""Create a minimal test graph for plan tests.

    Graph structure:
        modules (root)
           |
           v
          ast
         /   \
        v     v
      goids  typing
        |
        v
    function_metrics

    Returns
    -------
    TargetGraph
        Graph with modules -> ast -> goids/typing -> function_metrics chain.
    """
    graph = TargetGraph()

    # Root target with no dependencies
    modules_target = OutputTarget(
        name="modules",
        module="ingestion",
        plugin="repo_scan",
        tables=("core.modules",),
        dependencies=(),
        description="Repository module index",
        estimated_duration_ms=1000,
    )

    # Target depending on modules
    ast_target = OutputTarget(
        name="ast",
        module="ingestion",
        plugin="ast_extract",
        tables=("core.ast_nodes",),
        dependencies=("modules",),
        description="AST extraction",
        estimated_duration_ms=5000,
    )

    # Target depending on ast
    goids_target = OutputTarget(
        name="goids",
        module="graphs",
        plugin="goid_builder",
        tables=("core.goids",),
        dependencies=("ast",),
        description="GOID construction",
        estimated_duration_ms=10000,
    )

    # Independent target depending on ast
    typing_target = OutputTarget(
        name="typing",
        module="ingestion",
        plugin="typing_ingest",
        tables=("analytics.typedness",),
        dependencies=("ast",),
        description="Type analysis",
        estimated_duration_ms=3000,
    )

    # Target depending on goids
    metrics_target = OutputTarget(
        name="function_metrics",
        module="analytics",
        plugin="function_metrics",
        tables=("analytics.function_metrics",),
        dependencies=("goids",),
        description="Function metrics",
        estimated_duration_ms=8000,
    )

    graph.register(modules_target)
    graph.register(ast_target)
    graph.register(goids_target)
    graph.register(typing_target)
    graph.register(metrics_target)

    return graph


@pytest.fixture
def plan_graph() -> TargetGraph:
    """Provide the test graph for plan tests.

    Returns
    -------
    TargetGraph
        Graph with modules -> ast -> goids/typing -> function_metrics chain.
    """
    return _create_test_graph()


@pytest.fixture
def generator(plan_graph: TargetGraph) -> PlanGenerator:
    """Provide a PlanGenerator with test graph.

    Parameters
    ----------
    plan_graph
        Test graph fixture.

    Returns
    -------
    PlanGenerator
        Generator configured with test graph.
    """
    return PlanGenerator(plan_graph)


def _make_resolution(
    requested: tuple[str, ...],
    to_compute: tuple[str, ...],
    to_skip: tuple[str, ...] = (),
    blocked: tuple[str, ...] = (),
    reasons: dict[str, ResolutionReason] | None = None,
) -> ResolutionResult:
    """Create a ResolutionResult for testing.

    Parameters
    ----------
    requested
        Requested targets.
    to_compute
        Targets to compute.
    to_skip
        Targets to skip.
    blocked
        Blocked targets.
    reasons
        Optional reason mapping.

    Returns
    -------
    ResolutionResult
        Test resolution result.
    """
    if reasons is None:
        reasons = {
            name: ResolutionReason(kind="missing", details="No manifest exists")
            for name in to_compute
        }
    return ResolutionResult(
        requested=requested,
        to_compute=to_compute,
        to_skip=to_skip,
        blocked=blocked,
        reasons=reasons,
    )


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestFormatDuration:
    """Tests for format_duration helper."""

    def test_format_none(self) -> None:
        """None returns empty string."""
        result = format_duration(None)
        assert not result  # Empty string is falsey

    def test_format_milliseconds(self) -> None:
        """Small values show milliseconds."""
        assert format_duration(500) == ", ~500ms"
        assert format_duration(999) == ", ~999ms"

    def test_format_seconds(self) -> None:
        """Large values show seconds."""
        assert format_duration(1000) == ", ~1s"
        assert format_duration(5000) == ", ~5s"
        assert format_duration(90000) == ", ~90s"


# =============================================================================
# PlanStep Tests
# =============================================================================


class TestPlanStep:
    """Tests for PlanStep dataclass."""

    def test_create_step(self) -> None:
        """Create a plan step with all fields."""
        step = PlanStep(
            target="ast",
            module="ingestion",
            plugin="ast_extract",
            estimated_duration_ms=5000,
            dependencies=("modules",),
            reason="Target is stale",
        )
        assert step.target == "ast"
        assert step.module == "ingestion"
        assert step.plugin == "ast_extract"
        assert step.estimated_duration_ms == 5000
        assert step.dependencies == ("modules",)
        assert step.reason == "Target is stale"

    def test_step_is_frozen(self) -> None:
        """Verify step is immutable."""
        step = PlanStep(
            target="ast",
            module="ingestion",
            plugin="ast_extract",
            estimated_duration_ms=5000,
            dependencies=(),
            reason="",
        )
        assert_frozen(step, "target", "other")

    def test_step_to_dict(self) -> None:
        """Step serializes correctly."""
        step = PlanStep(
            target="ast",
            module="ingestion",
            plugin="ast_extract",
            estimated_duration_ms=5000,
            dependencies=("modules",),
            reason="Target is stale",
        )
        result = step.to_dict()

        assert result["target"] == "ast"
        assert result["module"] == "ingestion"
        assert result["plugin"] == "ast_extract"
        assert result["estimated_duration_ms"] == 5000
        assert result["dependencies"] == ["modules"]
        assert result["reason"] == "Target is stale"

    def test_step_to_dict_none_duration(self) -> None:
        """Step with None duration serializes correctly."""
        step = PlanStep(
            target="ast",
            module="ingestion",
            plugin="ast_extract",
            estimated_duration_ms=None,
            dependencies=(),
            reason="",
        )
        result = step.to_dict()
        assert result["estimated_duration_ms"] is None


# =============================================================================
# PlanStage Tests
# =============================================================================


class TestPlanStage:
    """Tests for PlanStage dataclass."""

    def test_create_stage(self) -> None:
        """Create a plan stage with steps."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "missing")
        step2 = PlanStep("ast", "ingestion", "ast_extract", 5000, ("modules",), "cascade")
        stage = PlanStage(module="ingestion", steps=(step1, step2))

        assert stage.module == "ingestion"
        assert len(stage.steps) == 2

    def test_step_count(self) -> None:
        """Step count returns correct value."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "")
        step2 = PlanStep("ast", "ingestion", "ast_extract", 5000, (), "")
        stage = PlanStage(module="ingestion", steps=(step1, step2))

        assert stage.step_count == 2

    def test_stage_duration_sums_steps(self) -> None:
        """Stage duration is sum of step durations."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "")
        step2 = PlanStep("ast", "ingestion", "ast_extract", 5000, (), "")
        stage = PlanStage(module="ingestion", steps=(step1, step2))

        assert stage.estimated_duration_ms == 6000

    def test_stage_duration_none_if_any_unknown(self) -> None:
        """Stage duration is None if any step is None."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "")
        step2 = PlanStep("ast", "ingestion", "ast_extract", None, (), "")
        stage = PlanStage(module="ingestion", steps=(step1, step2))

        assert stage.estimated_duration_ms is None

    def test_empty_stage_duration(self) -> None:
        """Empty stage has zero duration."""
        stage = PlanStage(module="ingestion", steps=())
        assert stage.estimated_duration_ms == 0

    def test_stage_to_dict(self) -> None:
        """Stage serializes correctly."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "missing")
        step2 = PlanStep("ast", "ingestion", "ast_extract", 5000, ("modules",), "cascade")
        stage = PlanStage(module="ingestion", steps=(step1, step2))

        result = stage.to_dict()

        assert result["module"] == "ingestion"
        assert result["step_count"] == 2
        assert result["estimated_duration_ms"] == 6000
        assert len(result["steps"]) == 2


# =============================================================================
# BuildPlan Tests
# =============================================================================


class TestBuildPlan:
    """Tests for BuildPlan dataclass."""

    def test_create_plan(self) -> None:
        """Create a build plan with stages."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "")
        stage1 = PlanStage(module="ingestion", steps=(step1,))

        step2 = PlanStep("goids", "graphs", "goid_builder", 10000, ("ast",), "")
        stage2 = PlanStage(module="graphs", steps=(step2,))

        plan = BuildPlan(
            requested_targets=("function_metrics",),
            stages=(stage1, stage2),
            skipped_targets=("coverage",),
            blocked_targets=(),
        )

        assert plan.requested_targets == ("function_metrics",)
        assert len(plan.stages) == 2
        assert plan.skipped_targets == ("coverage",)
        assert plan.blocked_targets == ()

    def test_total_steps(self) -> None:
        """Total steps sums across stages."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "")
        step2 = PlanStep("ast", "ingestion", "ast_extract", 5000, (), "")
        stage1 = PlanStage(module="ingestion", steps=(step1, step2))

        step3 = PlanStep("goids", "graphs", "goid_builder", 10000, (), "")
        stage2 = PlanStage(module="graphs", steps=(step3,))

        plan = BuildPlan(
            requested_targets=("goids",),
            stages=(stage1, stage2),
            skipped_targets=(),
            blocked_targets=(),
        )

        assert plan.total_steps == 3

    def test_plan_duration_sums_stages(self) -> None:
        """Plan duration is sum of stage durations."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "")
        stage1 = PlanStage(module="ingestion", steps=(step1,))

        step2 = PlanStep("goids", "graphs", "goid_builder", 10000, (), "")
        stage2 = PlanStage(module="graphs", steps=(step2,))

        plan = BuildPlan(
            requested_targets=("goids",),
            stages=(stage1, stage2),
            skipped_targets=(),
            blocked_targets=(),
        )

        assert plan.estimated_duration_ms == 11000

    def test_plan_duration_none_if_any_unknown(self) -> None:
        """Plan duration is None if any stage is None."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "")
        stage1 = PlanStage(module="ingestion", steps=(step1,))

        step2 = PlanStep("goids", "graphs", "goid_builder", None, (), "")
        stage2 = PlanStage(module="graphs", steps=(step2,))

        plan = BuildPlan(
            requested_targets=("goids",),
            stages=(stage1, stage2),
            skipped_targets=(),
            blocked_targets=(),
        )

        assert plan.estimated_duration_ms is None

    def test_is_empty_true(self) -> None:
        """Empty plan returns True."""
        plan = BuildPlan(
            requested_targets=("x",),
            stages=(),
            skipped_targets=("x",),
            blocked_targets=(),
        )
        assert plan.is_empty() is True

    def test_is_empty_false(self) -> None:
        """Non-empty plan returns False."""
        step = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "")
        stage = PlanStage(module="ingestion", steps=(step,))
        plan = BuildPlan(
            requested_targets=("modules",),
            stages=(stage,),
            skipped_targets=(),
            blocked_targets=(),
        )
        assert plan.is_empty() is False

    def test_plan_to_dict(self) -> None:
        """Plan serializes correctly."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "missing")
        stage1 = PlanStage(module="ingestion", steps=(step1,))

        plan = BuildPlan(
            requested_targets=("function_metrics",),
            stages=(stage1,),
            skipped_targets=("coverage", "typing"),
            blocked_targets=("external",),
        )

        result = plan.to_dict()

        assert result["requested_targets"] == ["function_metrics"]
        assert result["total_steps"] == 1
        assert result["estimated_duration_ms"] == 1000
        assert len(result["stages"]) == 1
        assert result["skipped_targets"] == ["coverage", "typing"]
        assert result["blocked_targets"] == ["external"]

    def test_plan_to_dict_round_trip(self) -> None:
        """Plan serializes to valid JSON."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "missing")
        stage1 = PlanStage(module="ingestion", steps=(step1,))

        plan = BuildPlan(
            requested_targets=("function_metrics",),
            stages=(stage1,),
            skipped_targets=(),
            blocked_targets=(),
        )

        # Should not raise
        json_str = json.dumps(plan.to_dict())
        parsed = json.loads(json_str)

        assert parsed["total_steps"] == 1


# =============================================================================
# BuildPlan.format_summary Tests
# =============================================================================


class TestFormatSummary:
    """Tests for BuildPlan.format_summary method."""

    def test_format_summary_basic(self) -> None:
        """Summary includes expected sections."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "missing")
        step2 = PlanStep("ast", "ingestion", "ast_extract", 5000, (), "cascade")
        stage = PlanStage(module="ingestion", steps=(step1, step2))

        plan = BuildPlan(
            requested_targets=("function_metrics",),
            stages=(stage,),
            skipped_targets=(),
            blocked_targets=(),
        )

        summary = plan.format_summary()

        assert "Build Plan for: function_metrics" in summary
        assert "Stage 1: Ingestion" in summary
        assert "modules" in summary
        assert "ast" in summary
        assert "Total: 2 steps" in summary

    def test_format_summary_with_skipped(self) -> None:
        """Summary shows skipped count."""
        step = PlanStep("ast", "ingestion", "ast_extract", 5000, (), "stale")
        stage = PlanStage(module="ingestion", steps=(step,))

        plan = BuildPlan(
            requested_targets=("ast",),
            stages=(stage,),
            skipped_targets=("modules", "coverage", "typing"),
            blocked_targets=(),
        )

        summary = plan.format_summary()

        assert "Skipped: 3 targets" in summary

    def test_format_summary_with_blocked(self) -> None:
        """Summary shows blocked count."""
        step = PlanStep("ast", "ingestion", "ast_extract", 5000, (), "stale")
        stage = PlanStage(module="ingestion", steps=(step,))

        plan = BuildPlan(
            requested_targets=("ast",),
            stages=(stage,),
            skipped_targets=(),
            blocked_targets=("external_data",),
        )

        summary = plan.format_summary()

        assert "Blocked: 1 targets" in summary

    def test_format_summary_multi_stage(self) -> None:
        """Summary shows multiple stages."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "")
        stage1 = PlanStage(module="ingestion", steps=(step1,))

        step2 = PlanStep("goids", "graphs", "goid_builder", 10000, (), "")
        stage2 = PlanStage(module="graphs", steps=(step2,))

        step3 = PlanStep("function_metrics", "analytics", "function_metrics", 8000, (), "")
        stage3 = PlanStage(module="analytics", steps=(step3,))

        plan = BuildPlan(
            requested_targets=("function_metrics",),
            stages=(stage1, stage2, stage3),
            skipped_targets=(),
            blocked_targets=(),
        )

        summary = plan.format_summary()

        assert "Stage 1: Ingestion" in summary
        assert "Stage 2: Graphs" in summary
        assert "Stage 3: Analytics" in summary

    def test_format_summary_empty_plan(self) -> None:
        """Summary handles empty plan."""
        plan = BuildPlan(
            requested_targets=("function_metrics",),
            stages=(),
            skipped_targets=("function_metrics",),
            blocked_targets=(),
        )

        summary = plan.format_summary()

        assert "Total: 0 steps" in summary
        assert "Skipped: 1 targets" in summary

    def test_format_summary_with_duration(self) -> None:
        """Summary includes duration estimate."""
        step = PlanStep("ast", "ingestion", "ast_extract", 5000, (), "stale")
        stage = PlanStage(module="ingestion", steps=(step,))

        plan = BuildPlan(
            requested_targets=("ast",),
            stages=(stage,),
            skipped_targets=(),
            blocked_targets=(),
        )

        summary = plan.format_summary()

        assert "~5s" in summary


# =============================================================================
# PlanGenerator Tests
# =============================================================================


class TestPlanGenerator:
    """Tests for PlanGenerator class."""

    def test_generate_empty_resolution(
        self,
        generator: PlanGenerator,
    ) -> None:
        """Empty to_compute produces empty plan."""
        resolution = _make_resolution(
            requested=("function_metrics",),
            to_compute=(),
            to_skip=("function_metrics",),
        )

        plan = generator.generate(resolution)

        assert plan.is_empty()
        assert plan.total_steps == 0
        assert plan.stages == ()
        assert plan.skipped_targets == ("function_metrics",)

    def test_generate_single_target(
        self,
        generator: PlanGenerator,
    ) -> None:
        """Single target produces one-step plan."""
        resolution = _make_resolution(
            requested=("modules",),
            to_compute=("modules",),
        )

        plan = generator.generate(resolution)

        assert plan.total_steps == 1
        assert len(plan.stages) == 1
        assert plan.stages[0].module == "ingestion"
        assert plan.stages[0].steps[0].target == "modules"

    def test_generate_multi_module(
        self,
        generator: PlanGenerator,
    ) -> None:
        """Targets across modules create multiple stages."""
        resolution = _make_resolution(
            requested=("function_metrics",),
            to_compute=("modules", "ast", "goids", "function_metrics"),
        )

        plan = generator.generate(resolution)

        # Should have 3 stages: ingestion, graphs, analytics
        assert len(plan.stages) == 3
        assert plan.stages[0].module == "ingestion"
        assert plan.stages[1].module == "graphs"
        assert plan.stages[2].module == "analytics"

    def test_stages_in_module_order(
        self,
        generator: PlanGenerator,
    ) -> None:
        """Stages are ordered: ingestion -> graphs -> analytics."""
        # Create resolution with targets in reverse order
        resolution = _make_resolution(
            requested=("function_metrics",),
            to_compute=("function_metrics", "goids", "modules", "ast"),
        )

        plan = generator.generate(resolution)

        modules = [stage.module for stage in plan.stages]
        assert modules == ["ingestion", "graphs", "analytics"]

    def test_empty_modules_skipped(
        self,
        generator: PlanGenerator,
    ) -> None:
        """Modules with no targets don't create stages."""
        # Only ingestion targets
        resolution = _make_resolution(
            requested=("ast",),
            to_compute=("modules", "ast"),
        )

        plan = generator.generate(resolution)

        assert len(plan.stages) == 1
        assert plan.stages[0].module == "ingestion"

    def test_step_gets_target_metadata(
        self,
        generator: PlanGenerator,
    ) -> None:
        """Step gets duration and plugin from OutputTarget."""
        resolution = _make_resolution(
            requested=("ast",),
            to_compute=("ast",),
        )

        plan = generator.generate(resolution)
        step = plan.stages[0].steps[0]

        assert step.plugin == "ast_extract"
        assert step.estimated_duration_ms == 5000
        assert step.dependencies == ("modules",)

    def test_step_gets_reason_from_resolution(
        self,
        generator: PlanGenerator,
    ) -> None:
        """Step gets reason details from resolution."""
        reasons = {
            "ast": ResolutionReason(kind="stale", details="Target is stale: hash changed"),
        }
        resolution = _make_resolution(
            requested=("ast",),
            to_compute=("ast",),
            reasons=reasons,
        )

        plan = generator.generate(resolution)
        step = plan.stages[0].steps[0]

        assert step.reason == "Target is stale: hash changed"

    def test_preserves_skipped_and_blocked(
        self,
        generator: PlanGenerator,
    ) -> None:
        """Plan preserves skipped and blocked from resolution."""
        resolution = ResolutionResult(
            requested=("function_metrics",),
            to_compute=("modules",),
            to_skip=("ast", "goids"),
            blocked=("external_data",),
            reasons={"modules": ResolutionReason(kind="missing", details="missing")},
        )

        plan = generator.generate(resolution)

        assert plan.skipped_targets == ("ast", "goids")
        assert plan.blocked_targets == ("external_data",)

    def test_preserves_requested_targets(
        self,
        generator: PlanGenerator,
    ) -> None:
        """Plan preserves requested targets from resolution."""
        resolution = _make_resolution(
            requested=("function_metrics", "typing"),
            to_compute=("modules",),
        )

        plan = generator.generate(resolution)

        assert plan.requested_targets == ("function_metrics", "typing")


# =============================================================================
# Integration Tests
# =============================================================================


class TestPlanGeneratorIntegration:
    """Integration tests for plan generation."""

    def test_full_chain_plan(
        self,
        generator: PlanGenerator,
    ) -> None:
        """Generate plan for full dependency chain."""
        resolution = _make_resolution(
            requested=("function_metrics",),
            to_compute=("modules", "ast", "goids", "function_metrics"),
        )

        plan = generator.generate(resolution)

        # Verify structure
        assert plan.total_steps == 4
        assert len(plan.stages) == 3

        # Verify stage contents
        ingestion_targets = [s.target for s in plan.stages[0].steps]
        graphs_targets = [s.target for s in plan.stages[1].steps]
        analytics_targets = [s.target for s in plan.stages[2].steps]

        assert "modules" in ingestion_targets
        assert "ast" in ingestion_targets
        assert "goids" in graphs_targets
        assert "function_metrics" in analytics_targets

    def test_with_real_registry(self) -> None:
        """Plan generation with full target registry."""
        graph = get_target_graph()
        generator = PlanGenerator(graph)

        # Create resolution for a real target
        resolution = _make_resolution(
            requested=("function_metrics",),
            to_compute=("modules", "ast", "goids", "function_metrics"),
        )

        plan = generator.generate(resolution)

        # Should produce valid plan
        assert plan.total_steps == 4
        assert not plan.is_empty()

        # Should serialize without error
        plan_dict = plan.to_dict()
        assert "stages" in plan_dict

    def test_summary_format_integration(
        self,
        generator: PlanGenerator,
    ) -> None:
        """Summary format works with generated plan."""
        reasons = {
            "modules": ResolutionReason(kind="missing", details="No manifest exists"),
            "ast": ResolutionReason(kind="cascade", details="Cascade from modules"),
            "goids": ResolutionReason(kind="cascade", details="Cascade from ast"),
            "function_metrics": ResolutionReason(kind="cascade", details="Cascade from goids"),
        }
        resolution = ResolutionResult(
            requested=("function_metrics",),
            to_compute=("modules", "ast", "goids", "function_metrics"),
            to_skip=("typing",),
            blocked=(),
            reasons=reasons,
        )

        plan = generator.generate(resolution)
        summary = plan.format_summary()

        # Verify all expected content
        assert "function_metrics" in summary
        assert "modules" in summary
        assert "Ingestion" in summary
        assert "Graphs" in summary
        assert "Analytics" in summary
        assert "Skipped: 1 targets" in summary
