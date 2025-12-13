"""Unit tests for the plan generation module."""

from __future__ import annotations

import json

import pytest
from codeintel.build.plan import (
    BuildPlan,
    PlanGenerator,
    PlanStage,
    PlanStep,
    format_duration,
)
from codeintel.build.resolver import ResolutionReason, ResolutionResult

from codeintel.build.registry import get_target_graph
from codeintel.build.targets import OutputTarget, TargetGraph, TargetOptions
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_none,
    expect_length,
    expect_true,
)


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

    typing_target = OutputTarget.from_tables(
        name="typing",
        module="ingestion",
        plugin="typing_ingest",
        tables=("analytics.typedness",),
        options=TargetOptions(dependencies=("ast",), description="Type analysis"),
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


class TestFormatDuration:
    """Tests for format_duration helper."""

    @staticmethod
    def test_format_none() -> None:
        """None returns empty string."""
        result = format_duration(None)
        expect_false(bool(result))

    @staticmethod
    def test_format_milliseconds() -> None:
        """Small values show milliseconds."""
        expect_equal(format_duration(500), ", ~500ms")
        expect_equal(format_duration(999), ", ~999ms")

    @staticmethod
    def test_format_seconds() -> None:
        """Large values show seconds."""
        expect_equal(format_duration(1000), ", ~1s")
        expect_equal(format_duration(5000), ", ~5s")
        expect_equal(format_duration(90000), ", ~90s")


class TestPlanStep:
    """Tests for PlanStep dataclass."""

    @staticmethod
    def test_create_step() -> None:
        """Create a plan step with all fields."""
        step = PlanStep(
            target="ast",
            module="ingestion",
            plugin="ast_extract",
            estimated_duration_ms=5000,
            dependencies=("modules",),
            reason="Target is stale",
        )
        expect_equal(step.target, "ast")
        expect_equal(step.module, "ingestion")
        expect_equal(step.plugin, "ast_extract")
        expect_equal(step.estimated_duration_ms, 5000)
        expect_equal(step.dependencies, ("modules",))
        expect_equal(step.reason, "Target is stale")

    @staticmethod
    def test_step_is_frozen() -> None:
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

    @staticmethod
    def test_step_to_dict() -> None:
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

        expect_equal(result["target"], "ast")
        expect_equal(result["module"], "ingestion")
        expect_equal(result["plugin"], "ast_extract")
        expect_equal(result["estimated_duration_ms"], 5000)
        expect_equal(result["dependencies"], ["modules"])
        expect_equal(result["reason"], "Target is stale")

    @staticmethod
    def test_step_to_dict_none_duration() -> None:
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
        expect_is_none(result["estimated_duration_ms"])


class TestPlanStage:
    """Tests for PlanStage dataclass."""

    @staticmethod
    def test_create_stage() -> None:
        """Create a plan stage with steps."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "missing")
        step2 = PlanStep("ast", "ingestion", "ast_extract", 5000, ("modules",), "cascade")
        stage = PlanStage(module="ingestion", steps=(step1, step2))

        expect_equal(stage.module, "ingestion")
        expect_length(stage.steps, 2)

    @staticmethod
    def test_step_count() -> None:
        """Step count returns correct value."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "")
        step2 = PlanStep("ast", "ingestion", "ast_extract", 5000, (), "")
        stage = PlanStage(module="ingestion", steps=(step1, step2))

        expect_equal(stage.step_count, 2)

    @staticmethod
    def test_stage_duration_sums_steps() -> None:
        """Stage duration is sum of step durations."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "")
        step2 = PlanStep("ast", "ingestion", "ast_extract", 5000, (), "")
        stage = PlanStage(module="ingestion", steps=(step1, step2))

        expect_equal(stage.estimated_duration_ms, 6000)

    @staticmethod
    def test_stage_duration_none_if_any_unknown() -> None:
        """Stage duration is None if any step is None."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "")
        step2 = PlanStep("ast", "ingestion", "ast_extract", None, (), "")
        stage = PlanStage(module="ingestion", steps=(step1, step2))

        expect_is_none(stage.estimated_duration_ms)

    @staticmethod
    def test_empty_stage_duration() -> None:
        """Empty stage has zero duration."""
        stage = PlanStage(module="ingestion", steps=())
        expect_equal(stage.estimated_duration_ms, 0)

    @staticmethod
    def test_stage_to_dict() -> None:
        """Stage serializes correctly."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "missing")
        step2 = PlanStep("ast", "ingestion", "ast_extract", 5000, ("modules",), "cascade")
        stage = PlanStage(module="ingestion", steps=(step1, step2))

        result = stage.to_dict()

        expect_equal(result["module"], "ingestion")
        expect_equal(result["step_count"], 2)
        expect_equal(result["estimated_duration_ms"], 6000)
        expect_length(result["steps"], 2)


class TestBuildPlan:
    """Tests for BuildPlan dataclass."""

    @staticmethod
    def test_create_plan() -> None:
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

        expect_equal(plan.requested_targets, ("function_metrics",))
        expect_length(plan.stages, 2)
        expect_equal(plan.skipped_targets, ("coverage",))
        expect_equal(plan.blocked_targets, ())

    @staticmethod
    def test_total_steps() -> None:
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

        expect_equal(plan.total_steps, 3)

    @staticmethod
    def test_plan_duration_sums_stages() -> None:
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

        expect_equal(plan.estimated_duration_ms, 11000)

    @staticmethod
    def test_plan_duration_none_if_any_unknown() -> None:
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

        expect_is_none(plan.estimated_duration_ms)

    @staticmethod
    def test_is_empty_true() -> None:
        """Empty plan returns True."""
        plan = BuildPlan(
            requested_targets=("x",),
            stages=(),
            skipped_targets=("x",),
            blocked_targets=(),
        )
        expect_true(plan.is_empty())

    @staticmethod
    def test_is_empty_false() -> None:
        """Non-empty plan returns False."""
        step = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "")
        stage = PlanStage(module="ingestion", steps=(step,))
        plan = BuildPlan(
            requested_targets=("modules",),
            stages=(stage,),
            skipped_targets=(),
            blocked_targets=(),
        )
        expect_false(plan.is_empty())

    @staticmethod
    def test_plan_to_dict() -> None:
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

        expect_equal(result["requested_targets"], ["function_metrics"])
        expect_equal(result["total_steps"], 1)
        expect_equal(result["estimated_duration_ms"], 1000)
        expect_length(result["stages"], 1)
        expect_equal(result["skipped_targets"], ["coverage", "typing"])
        expect_equal(result["blocked_targets"], ["external"])

    @staticmethod
    def test_plan_to_dict_round_trip() -> None:
        """Plan serializes to valid JSON."""
        step1 = PlanStep("modules", "ingestion", "repo_scan", 1000, (), "missing")
        stage1 = PlanStage(module="ingestion", steps=(step1,))

        plan = BuildPlan(
            requested_targets=("function_metrics",),
            stages=(stage1,),
            skipped_targets=(),
            blocked_targets=(),
        )

        json_str = json.dumps(plan.to_dict())
        parsed = json.loads(json_str)

        expect_equal(parsed["total_steps"], 1)


class TestFormatSummary:
    """Tests for BuildPlan.format_summary method."""

    @staticmethod
    def test_format_summary_basic() -> None:
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

        expect_in("Build Plan for: function_metrics", summary)
        expect_in("Stage 1: Ingestion", summary)
        expect_in("modules", summary)
        expect_in("ast", summary)
        expect_in("Total: 2 steps", summary)

    @staticmethod
    def test_format_summary_with_skipped() -> None:
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

        expect_in("Skipped: 3 targets", summary)

    @staticmethod
    def test_format_summary_with_blocked() -> None:
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

        expect_in("Blocked: 1 targets", summary)

    @staticmethod
    def test_format_summary_multi_stage() -> None:
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

        expect_in("Stage 1: Ingestion", summary)
        expect_in("Stage 2: Graphs", summary)
        expect_in("Stage 3: Analytics", summary)

    @staticmethod
    def test_format_summary_empty_plan() -> None:
        """Summary handles empty plan."""
        plan = BuildPlan(
            requested_targets=("function_metrics",),
            stages=(),
            skipped_targets=("function_metrics",),
            blocked_targets=(),
        )

        summary = plan.format_summary()

        expect_in("Total: 0 steps", summary)
        expect_in("Skipped: 1 targets", summary)

    @staticmethod
    def test_format_summary_with_duration() -> None:
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

        expect_in("~5s", summary)


class TestPlanGenerator:
    """Tests for PlanGenerator class."""

    @staticmethod
    def test_generate_empty_resolution(generator: PlanGenerator) -> None:
        """Empty to_compute produces empty plan."""
        resolution = _make_resolution(
            requested=("function_metrics",),
            to_compute=(),
            to_skip=("function_metrics",),
        )

        plan = generator.generate(resolution)

        expect_true(plan.is_empty())
        expect_equal(plan.total_steps, 0)
        expect_equal(plan.stages, ())
        expect_equal(plan.skipped_targets, ("function_metrics",))

    @staticmethod
    def test_generate_single_target(generator: PlanGenerator) -> None:
        """Single target produces one-step plan."""
        resolution = _make_resolution(
            requested=("modules",),
            to_compute=("modules",),
        )

        plan = generator.generate(resolution)

        expect_equal(plan.total_steps, 1)
        expect_length(plan.stages, 1)
        expect_equal(plan.stages[0].module, "ingestion")
        expect_equal(plan.stages[0].steps[0].target, "modules")

    @staticmethod
    def test_generate_multi_module(generator: PlanGenerator) -> None:
        """Targets across modules create multiple stages."""
        resolution = _make_resolution(
            requested=("function_metrics",),
            to_compute=("modules", "ast", "goids", "function_metrics"),
        )

        plan = generator.generate(resolution)

        expect_length(plan.stages, 3)
        expect_equal(plan.stages[0].module, "ingestion")
        expect_equal(plan.stages[1].module, "graphs")
        expect_equal(plan.stages[2].module, "analytics")

    @staticmethod
    def test_stages_in_module_order(generator: PlanGenerator) -> None:
        """Stages are ordered: ingestion -> graphs -> analytics."""
        resolution = _make_resolution(
            requested=("function_metrics",),
            to_compute=("function_metrics", "goids", "modules", "ast"),
        )

        plan = generator.generate(resolution)

        modules = [stage.module for stage in plan.stages]
        expect_equal(modules, ["ingestion", "graphs", "analytics"])

    @staticmethod
    def test_empty_modules_skipped(generator: PlanGenerator) -> None:
        """Modules with no targets don't create stages."""
        resolution = _make_resolution(
            requested=("ast",),
            to_compute=("modules", "ast"),
        )

        plan = generator.generate(resolution)

        expect_length(plan.stages, 1)
        expect_equal(plan.stages[0].module, "ingestion")

    @staticmethod
    def test_step_gets_target_metadata(generator: PlanGenerator) -> None:
        """Step gets duration and plugin from OutputTarget."""
        resolution = _make_resolution(
            requested=("ast",),
            to_compute=("ast",),
        )

        plan = generator.generate(resolution)
        step = plan.stages[0].steps[0]

        expect_equal(step.plugin, "ast_extract")
        expect_equal(step.estimated_duration_ms, 5000)
        expect_equal(step.dependencies, ("modules",))

    @staticmethod
    def test_step_gets_reason_from_resolution(generator: PlanGenerator) -> None:
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

        expect_equal(step.reason, "Target is stale: hash changed")

    @staticmethod
    def test_preserves_skipped_and_blocked(generator: PlanGenerator) -> None:
        """Plan preserves skipped and blocked from resolution."""
        resolution = ResolutionResult(
            requested=("function_metrics",),
            to_compute=("modules",),
            to_skip=("ast", "goids"),
            blocked=("external_data",),
            reasons={"modules": ResolutionReason(kind="missing", details="missing")},
        )

        plan = generator.generate(resolution)

        expect_equal(plan.skipped_targets, ("ast", "goids"))
        expect_equal(plan.blocked_targets, ("external_data",))

    @staticmethod
    def test_preserves_requested_targets(generator: PlanGenerator) -> None:
        """Plan preserves requested targets from resolution."""
        resolution = _make_resolution(
            requested=("function_metrics", "typing"),
            to_compute=("modules",),
        )

        plan = generator.generate(resolution)

        expect_equal(plan.requested_targets, ("function_metrics", "typing"))


class TestPlanGeneratorIntegration:
    """Integration tests for plan generation."""

    @staticmethod
    def test_full_chain_plan(
        generator: PlanGenerator,
    ) -> None:
        """Generate plan for full dependency chain."""
        resolution = _make_resolution(
            requested=("function_metrics",),
            to_compute=("modules", "ast", "goids", "function_metrics"),
        )

        plan = generator.generate(resolution)

        expect_equal(plan.total_steps, 4)
        expect_length(plan.stages, 3)

        ingestion_targets = [s.target for s in plan.stages[0].steps]
        graphs_targets = [s.target for s in plan.stages[1].steps]
        analytics_targets = [s.target for s in plan.stages[2].steps]

        expect_in("modules", ingestion_targets)
        expect_in("ast", ingestion_targets)
        expect_in("goids", graphs_targets)
        expect_in("function_metrics", analytics_targets)

    @staticmethod
    def test_with_real_registry() -> None:
        """Plan generation with full target registry."""
        graph = get_target_graph()
        generator = PlanGenerator(graph)

        resolution = _make_resolution(
            requested=("function_metrics",),
            to_compute=("modules", "ast", "goids", "function_metrics"),
        )

        plan = generator.generate(resolution)

        expect_equal(plan.total_steps, 4)
        expect_false(plan.is_empty())

        plan_dict = plan.to_dict()
        expect_in("stages", plan_dict)

    @staticmethod
    def test_summary_format_integration(
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

        expect_in("function_metrics", summary)
        expect_in("modules", summary)
        expect_in("Ingestion", summary)
        expect_in("Graphs", summary)
        expect_in("Analytics", summary)
        expect_in("Skipped: 1 targets", summary)
