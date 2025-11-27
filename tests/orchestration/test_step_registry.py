"""Tests for the pipeline step registry."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.pipeline.orchestration.core import (
    PipelineStep,
    StepMetadata,
    StepPhase,
)
from codeintel.pipeline.orchestration.registry import build_registry
from codeintel.pipeline.orchestration.steps import (
    PIPELINE_DEPS,
    PIPELINE_SEQUENCE,
    PIPELINE_STEPS,
    REGISTRY,
)
from tests._helpers.expect import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_true,
)

if TYPE_CHECKING:
    from codeintel.pipeline.orchestration.core import PipelineContext


STEP_PAIR_COUNT = 2


@dataclass
class MockStep:
    """Mock step for testing."""

    name: str
    description: str
    phase: StepPhase
    deps: Sequence[str]
    run_count: int = 0
    observed_repos: list[str] = field(default_factory=list)

    def run(self, ctx: PipelineContext) -> None:
        """Record run count and context repo."""
        self.run_count += 1
        self.observed_repos.append(getattr(ctx, "repo", ""))


def _registry_with_steps(steps: Iterable[PipelineStep]) -> dict[str, PipelineStep]:
    return {step.name: step for step in steps}


def test_build_registry_from_dicts() -> None:
    """Test building a registry from step dictionaries."""
    step_a = MockStep(
        name="step_a",
        description="Step A",
        phase=StepPhase.INGESTION,
        deps=(),
    )
    step_b = MockStep(
        name="step_b",
        description="Step B",
        phase=StepPhase.ANALYTICS,
        deps=("step_a",),
    )
    dict_a = _registry_with_steps([step_a])
    dict_b = _registry_with_steps([step_b])

    registry = build_registry(dict_a, dict_b)

    expect_equal(len(registry), STEP_PAIR_COUNT, label="registry size")
    expect_in("step_a", registry, label="step_a present")
    expect_in("step_b", registry, label="step_b present")
    expect_true(registry["step_a"] is step_a, message="step_a instance stored")
    expect_true(registry["step_b"] is step_b, message="step_b instance stored")


def test_get_step() -> None:
    """Test retrieving steps by name."""
    step = MockStep(
        name="test_step",
        description="Test",
        phase=StepPhase.GRAPHS,
        deps=(),
    )
    registry = build_registry({"test_step": step})

    expect_true(registry.get("test_step") is step, message="existing step lookup")
    expect_true(registry.get("nonexistent") is None, message="missing step lookup")


def test_getitem_raises_keyerror() -> None:
    """Test that __getitem__ raises KeyError for unknown steps."""
    registry = build_registry({})

    with pytest.raises(KeyError, match="Unknown pipeline step"):
        registry["nonexistent"]


def test_list_all() -> None:
    """Test listing all steps as metadata."""
    step_a = MockStep(
        name="step_a",
        description="First step",
        phase=StepPhase.INGESTION,
        deps=(),
    )
    step_b = MockStep(
        name="step_b",
        description="Second step",
        phase=StepPhase.EXPORT,
        deps=("step_a",),
    )
    registry = build_registry(_registry_with_steps([step_a, step_b]))

    metadata_list = registry.list_all()

    expect_equal(len(metadata_list), STEP_PAIR_COUNT, label="metadata length")
    expect_equal(metadata_list[0].name, "step_a", label="first step name")
    expect_equal(metadata_list[0].description, "First step", label="first description")
    expect_equal(metadata_list[0].phase, StepPhase.INGESTION, label="first phase")
    expect_equal(metadata_list[1].name, "step_b", label="second step name")
    expect_equal(metadata_list[1].deps, ("step_a",), label="second deps")


@pytest.mark.parametrize(
    ("phase", "expected_names"),
    [
        (StepPhase.INGESTION, ("ingest",)),
        (StepPhase.ANALYTICS, ("analyze",)),
        (StepPhase.EXPORT, ()),
    ],
)
def test_list_by_phase(phase: StepPhase, expected_names: tuple[str, ...]) -> None:
    """Test filtering steps by phase."""
    ingestion_step = MockStep(
        name="ingest",
        description="Ingestion",
        phase=StepPhase.INGESTION,
        deps=(),
    )
    analytics_step = MockStep(
        name="analyze",
        description="Analytics",
        phase=StepPhase.ANALYTICS,
        deps=(),
    )
    registry = build_registry(_registry_with_steps([ingestion_step, analytics_step]))

    steps = registry.list_by_phase(phase)
    expect_equal(len(steps), len(expected_names), label=f"{phase.value} count")
    for step in steps:
        expect_in(step.name, expected_names, label=f"{phase.value} names")


def test_dependency_graph() -> None:
    """Test getting the full dependency graph."""
    step_a = MockStep(
        name="a",
        description="A",
        phase=StepPhase.INGESTION,
        deps=(),
    )
    step_b = MockStep(
        name="b",
        description="B",
        phase=StepPhase.INGESTION,
        deps=("a",),
    )
    step_c = MockStep(
        name="c",
        description="C",
        phase=StepPhase.ANALYTICS,
        deps=("a", "b"),
    )
    registry = build_registry(_registry_with_steps([step_a, step_b, step_c]))

    deps = registry.dependency_graph()

    expect_equal(deps, {"a": (), "b": ("a",), "c": ("a", "b")}, label="deps graph")


def test_expand_with_deps() -> None:
    """Test expanding step selection to include dependencies."""
    step_a = MockStep(name="a", description="A", phase=StepPhase.INGESTION, deps=())
    step_b = MockStep(
        name="b",
        description="B",
        phase=StepPhase.INGESTION,
        deps=("a",),
    )
    step_c = MockStep(
        name="c",
        description="C",
        phase=StepPhase.ANALYTICS,
        deps=("b",),
    )
    registry = build_registry(_registry_with_steps([step_a, step_b, step_c]))

    expanded = registry.expand_with_deps(["c"])

    expect_equal(expanded, {"a", "b", "c"}, label="expanded deps")


def test_expand_with_deps_unknown_step() -> None:
    """Test that expand_with_deps raises KeyError for unknown steps."""
    registry = build_registry({})

    with pytest.raises(KeyError):
        registry.expand_with_deps(["unknown"])


def test_topological_order() -> None:
    """Test topological sorting of steps."""
    step_a = MockStep(name="a", description="A", phase=StepPhase.INGESTION, deps=())
    step_b = MockStep(
        name="b",
        description="B",
        phase=StepPhase.INGESTION,
        deps=("a",),
    )
    step_c = MockStep(
        name="c",
        description="C",
        phase=StepPhase.ANALYTICS,
        deps=("a", "b"),
    )
    registry = build_registry(_registry_with_steps([step_a, step_b, step_c]))

    ordered = registry.topological_order(["c", "b", "a"])

    expect_true(ordered.index("a") < ordered.index("b"), message="a before b")
    expect_true(ordered.index("b") < ordered.index("c"), message="b before c")


def test_topological_order_cycle_detection() -> None:
    """Test that cyclic dependencies are detected."""
    step_a = MockStep(
        name="a",
        description="A",
        phase=StepPhase.INGESTION,
        deps=("b",),
    )
    step_b = MockStep(
        name="b",
        description="B",
        phase=StepPhase.INGESTION,
        deps=("a",),
    )
    registry = build_registry(_registry_with_steps([step_a, step_b]))

    with pytest.raises(RuntimeError, match="Circular dependencies"):
        registry.topological_order(["a", "b"])


def test_as_dict() -> None:
    """Test converting registry to dictionary."""
    step = MockStep(name="test", description="Test", phase=StepPhase.INGESTION, deps=())
    registry = build_registry({"test": step})

    result = registry.as_dict()

    expect_is_instance(result, dict, label="registry as dict")
    expect_true(result["test"] is step, message="step stored in dict")


def test_iteration() -> None:
    """Test iterating over registry."""
    step_a = MockStep(name="a", description="A", phase=StepPhase.INGESTION, deps=())
    step_b = MockStep(name="b", description="B", phase=StepPhase.INGESTION, deps=())
    registry = build_registry(_registry_with_steps([step_a, step_b]))

    names = list(registry)

    expect_equal(names, ["a", "b"], label="iteration order")


def test_run_records_context_repo() -> None:
    """Test run captures context repository for diagnostics."""
    step = MockStep(name="a", description="A", phase=StepPhase.INGESTION, deps=())
    registry = build_registry({"a": step})
    ctx_stub = cast("PipelineContext", SimpleNamespace(repo="example/repo"))

    registry["a"].run(ctx_stub)

    expect_equal(step.run_count, 1, label="run count")
    expect_equal(step.observed_repos, ["example/repo"], label="observed repo")


def test_registry_contains_all_steps() -> None:
    """Test that REGISTRY contains all expected pipeline steps."""
    expect_true(len(REGISTRY) > 0, message="registry not empty")
    expect_in("repo_scan", REGISTRY, label="repo_scan present")
    expect_in("ast_extract", REGISTRY, label="ast_extract present")
    expect_in("export_docs", REGISTRY, label="export_docs present")


def test_backward_compat_exports() -> None:
    """Test backward-compatible exports."""
    expect_equal(REGISTRY.as_dict(), PIPELINE_STEPS, label="PIPELINE_STEPS")
    expect_equal(REGISTRY.dependency_graph(), PIPELINE_DEPS, label="PIPELINE_DEPS")
    expect_equal(REGISTRY.list_all_names(), PIPELINE_SEQUENCE, label="PIPELINE_SEQUENCE")


def test_all_steps_have_required_attributes() -> None:
    """Test that all registered steps have required attributes."""
    for name in REGISTRY:
        step = REGISTRY[name]
        expect_true(hasattr(step, "name"), message="name attr")
        expect_true(hasattr(step, "description"), message="description attr")
        expect_true(hasattr(step, "phase"), message="phase attr")
        expect_true(hasattr(step, "deps"), message="deps attr")
        expect_true(hasattr(step, "run"), message="run attr")

        expect_is_instance(step.name, str, label="name type")
        expect_is_instance(step.description, str, label="description type")
        expect_is_instance(step.phase, StepPhase, label="phase type")
        expect_equal(step.name, name, label="registry key matches step name")


def test_step_metadata() -> None:
    """Test that step metadata is correctly constructed."""
    all_metadata = REGISTRY.list_all()

    for meta in all_metadata:
        expect_is_instance(meta, StepMetadata, label="metadata type")
        expect_in(meta.name, REGISTRY, label="metadata name present")
        step = REGISTRY[meta.name]
        expect_equal(meta.description, step.description, label="metadata description")
        expect_equal(meta.phase, step.phase, label="metadata phase")
        expect_equal(meta.deps, tuple(step.deps), label="metadata deps")


def test_phase_coverage() -> None:
    """Test that all phases have at least one step."""
    for phase in StepPhase:
        steps = REGISTRY.list_by_phase(phase)
        expect_true(len(steps) > 0, message=f"Phase {phase.value} has no steps")
