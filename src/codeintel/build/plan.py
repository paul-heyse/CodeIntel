"""Plan data model and generator used by legacy tests and CLIs.

This module provides lightweight, Hamilton-agnostic planning types used by
unit tests for the minimal-work resolver and plan formatting.

The canonical Hamilton planner lives under ``codeintel.build.hamilton.planner``.
These types remain as a stable surface for the legacy resolver-driven plan flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.resolver import ResolutionResult
    from codeintel.build.targets import OutputTarget, TargetGraph

log = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND: int = 1000


class PlanStepDict(TypedDict):
    """Serialized representation of a PlanStep."""

    target: str
    module: str
    plugin: str
    estimated_duration_ms: int | None
    dependencies: list[str]
    reason: str


class PlanStageDict(TypedDict):
    """Serialized representation of a PlanStage."""

    module: str
    step_count: int
    estimated_duration_ms: int | None
    steps: list[PlanStepDict]


class BuildPlanDict(TypedDict):
    """Serialized representation of a BuildPlan."""

    requested_targets: list[str]
    total_steps: int
    estimated_duration_ms: int | None
    stages: list[PlanStageDict]
    skipped_targets: list[str]
    blocked_targets: list[str]


def format_duration(duration_ms: int | None) -> str:
    """Format a duration for human-readable plan summaries.

    Parameters
    ----------
    duration_ms
        Duration in milliseconds.

    Returns
    -------
    str
        A formatted suffix like ``", ~500ms"`` or ``", ~2s"``.
    """
    if duration_ms is None:
        return ""
    if duration_ms < MILLISECONDS_PER_SECOND:
        return f", ~{duration_ms}ms"
    return f", ~{duration_ms // MILLISECONDS_PER_SECOND}s"


@dataclass(frozen=True)
class PlanStep:
    """Single planned execution step."""

    target: str
    module: str
    plugin: str
    estimated_duration_ms: int | None
    dependencies: tuple[str, ...]
    reason: str

    def to_dict(self) -> PlanStepDict:
        """Serialize to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, object]
            JSON-friendly representation of the plan step.
        """
        return {
            "target": self.target,
            "module": self.module,
            "plugin": self.plugin,
            "estimated_duration_ms": self.estimated_duration_ms,
            "dependencies": list(self.dependencies),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PlanStage:
    """Stage grouping steps by module."""

    module: str
    steps: tuple[PlanStep, ...]

    @property
    def step_count(self) -> int:
        """Number of steps in this stage."""
        return len(self.steps)

    @property
    def estimated_duration_ms(self) -> int | None:
        """Estimated duration as the sum of step durations."""
        if not self.steps:
            return 0
        durations = [step.estimated_duration_ms for step in self.steps]
        if any(d is None for d in durations):
            return None
        return sum(d for d in durations if d is not None)

    def to_dict(self) -> PlanStageDict:
        """Serialize to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, object]
            JSON-friendly representation of the stage, including steps.
        """
        return {
            "module": self.module,
            "step_count": self.step_count,
            "estimated_duration_ms": self.estimated_duration_ms,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class BuildPlan:
    """Build plan for a requested set of targets."""

    requested_targets: tuple[str, ...]
    stages: tuple[PlanStage, ...]
    skipped_targets: tuple[str, ...]
    blocked_targets: tuple[str, ...]

    @property
    def total_steps(self) -> int:
        """Total number of steps across all stages."""
        return sum(stage.step_count for stage in self.stages)

    @property
    def estimated_duration_ms(self) -> int | None:
        """Estimated duration as sum of stage durations."""
        durations = [stage.estimated_duration_ms for stage in self.stages]
        if any(d is None for d in durations):
            return None
        return sum(d for d in durations if d is not None)

    def is_empty(self) -> bool:
        """Return True when there are no compute steps.

        Returns
        -------
        bool
            True when the plan contains no compute steps.
        """
        return self.total_steps == 0

    def to_dict(self) -> BuildPlanDict:
        """Serialize to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, object]
            JSON-friendly representation of the plan.
        """
        return {
            "requested_targets": list(self.requested_targets),
            "total_steps": self.total_steps,
            "estimated_duration_ms": self.estimated_duration_ms,
            "stages": [stage.to_dict() for stage in self.stages],
            "skipped_targets": list(self.skipped_targets),
            "blocked_targets": list(self.blocked_targets),
        }

    def format_summary(self) -> str:
        """Format a human-readable summary for CLI output.

        Returns
        -------
        str
            Multi-line, human-readable summary suitable for CLI output.
        """
        requested = ", ".join(self.requested_targets) if self.requested_targets else "(none)"
        lines: list[str] = [f"Build Plan for: {requested}"]
        lines.append(
            f"Total: {self.total_steps} steps{format_duration(self.estimated_duration_ms)}"
        )

        for idx, stage in enumerate(self.stages, start=1):
            label = stage.module.capitalize()
            lines.append(f"Stage {idx}: {label}{format_duration(stage.estimated_duration_ms)}")
            for step in stage.steps:
                lines.append(
                    f"- {step.target} ({step.plugin}){format_duration(step.estimated_duration_ms)}"
                )
                if step.reason:
                    lines.append(f"  {step.reason}")

        if self.skipped_targets:
            lines.append(f"Skipped: {len(self.skipped_targets)} targets")
        if self.blocked_targets:
            lines.append(f"Blocked: {len(self.blocked_targets)} targets")

        return "\n".join(lines)


PlanModule = Literal["ingestion", "graphs", "analytics", "export"]


class PlanGenerator:
    """Generate a BuildPlan from a ResolutionResult and target graph."""

    def __init__(self, graph: TargetGraph) -> None:
        self._graph = graph

    def generate(self, resolution: ResolutionResult) -> BuildPlan:
        """Generate a plan from a resolver output.

        Parameters
        ----------
        resolution
            Resolver output describing which targets to compute, skip, or block.

        Returns
        -------
        BuildPlan
            Build plan grouped by module stages.
        """
        steps = self._build_steps(resolution.to_compute, reasons=resolution.reasons)

        stages_by_module: dict[str, list[PlanStep]] = {}
        for step in steps:
            stages_by_module.setdefault(step.module, []).append(step)

        module_order: tuple[PlanModule, ...] = ("ingestion", "graphs", "analytics", "export")
        stages: list[PlanStage] = []
        for module in module_order:
            module_steps = stages_by_module.get(module)
            if module_steps:
                stages.append(PlanStage(module=module, steps=tuple(module_steps)))

        return BuildPlan(
            requested_targets=tuple(resolution.requested),
            stages=tuple(stages),
            skipped_targets=tuple(resolution.to_skip),
            blocked_targets=tuple(resolution.blocked),
        )

    def _build_steps(
        self,
        targets: Iterable[str],
        *,
        reasons: Mapping[str, object],
    ) -> list[PlanStep]:
        """Create PlanStep entries for a set of targets.

        Parameters
        ----------
        targets
            Target names to build plan steps for.
        reasons
            Mapping of target name to resolver reason object.

        Returns
        -------
        list[PlanStep]
            Planned steps ordered by the target iterable.
        """
        steps: list[PlanStep] = []
        for name in targets:
            target = self._graph.get(name)
            reason_obj = reasons.get(name)
            if reason_obj is None:
                log.warning("Target '%s' has no resolution reason; using empty reason", name)
                reason = ""
            else:
                reason = str(getattr(reason_obj, "details", ""))

            steps.append(_step_from_target(target, reason=reason))
        return steps


def _step_from_target(target: OutputTarget, *, reason: str) -> PlanStep:
    """Build a PlanStep from an OutputTarget definition.

    Parameters
    ----------
    target
        Target metadata describing the compute unit.
    reason
        Human-readable reason explaining why this target is included.

    Returns
    -------
    PlanStep
        Plan step populated from the target metadata.
    """
    return PlanStep(
        target=target.name,
        module=str(target.module),
        plugin=target.plugin,
        estimated_duration_ms=target.estimated_duration_ms,
        dependencies=tuple(target.dependencies),
        reason=reason,
    )


__all__ = [
    "BuildPlan",
    "PlanGenerator",
    "PlanStage",
    "PlanStep",
    "format_duration",
]
