"""Plan generation for the build system.

This module transforms resolution results (from Phase 3) into executable
build plans. A plan groups targets by target module, provides time
estimates, and can be serialized for dry-run display or async execution.

Key Concepts
------------
- **PlanStep**: A single target to compute with its metadata
- **PlanStage**: A group of steps in the same target module
- **BuildPlan**: Complete execution plan with stages, skipped, and blocked targets
- **PlanGenerator**: Transforms ResolutionResult into BuildPlan

The resolver tells us **what** to compute; the plan tells us **how**:
- **Ordering**: Respects module boundaries (ingestion -> graphs -> analytics)
- **Batching**: Groups targets that run in the same build stage
- **Estimation**: Provides time estimates for progress tracking
- **Serialization**: Supports JSON output for dry-run display

Integration Points
------------------
- Uses `TargetGraph` from Phase 1 for target metadata
- Uses `ResolutionResult` from Phase 3 for work to do
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codeintel.build.resolver import ResolutionResult
    from codeintel.build.targets import TargetGraph, TargetModule

log = logging.getLogger(__name__)

# =============================================================================
# Module-level Constants
# =============================================================================

MODULE_ORDER: tuple[TargetModule, ...] = ("ingestion", "graphs", "analytics", "export")
"""Canonical execution order for target modules."""

MS_PER_SECOND: int = 1000
"""Milliseconds per second for duration formatting."""


# =============================================================================
# Helper Functions
# =============================================================================


def format_duration(ms: float | None) -> str:
    """Format milliseconds as human-readable duration string.

    Parameters
    ----------
    ms
        Duration in milliseconds, or None if unknown.

    Returns
    -------
    str
        Formatted duration string (empty if None).

    Examples
    --------
    >>> format_duration(500)
    ', ~500ms'
    >>> format_duration(5000)
    ', ~5s'
    >>> format_duration(None)
    ''
    """
    if ms is None:
        return ""
    if ms < MS_PER_SECOND:
        return f", ~{ms}ms"
    return f", ~{ms // MS_PER_SECOND}s"


# =============================================================================
# Type Definitions
# =============================================================================


@dataclass(frozen=True)
class PlanStep:
    """A single step in the build plan.

    Represents one target that needs to be computed, along with metadata
    needed for execution and display.

    Attributes
    ----------
    target
        Target name to compute.
    module
        Target module (ingestion, graphs, or analytics).
    plugin
        Plugin name that produces this target.
    estimated_duration_ms
        Expected execution time in milliseconds, or None if unknown.
    dependencies
        Other targets this step depends on.
    reason
        Human-readable explanation of why this step is included.

    Examples
    --------
    >>> step = PlanStep(
    ...     target="ast",
    ...     module="ingestion",
    ...     plugin="ast_extract",
    ...     estimated_duration_ms=5000,
    ...     dependencies=("modules",),
    ...     reason="Target is stale: input hash changed",
    ... )
    >>> step.target
    'ast'
    """

    target: str
    module: TargetModule
    plugin: str
    estimated_duration_ms: int | None
    dependencies: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize step to dictionary for JSON output.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for JSON serialization.
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
    """A group of steps that execute together in one target module.

    Stages group targets by module (ingestion, graphs, analytics) and
    execute in a fixed order: ingestion first, then graphs, then analytics.

    Attributes
    ----------
    module
        Target module for this stage.
    steps
        Steps to execute in this stage.

    Examples
    --------
    >>> stage = PlanStage(
    ...     module="ingestion",
    ...     steps=(
    ...         PlanStep("modules", "ingestion", "repo_scan", 1000, (), "missing"),
    ...         PlanStep("ast", "ingestion", "ast_extract", 5000, ("modules",), "cascade"),
    ...     ),
    ... )
    >>> stage.step_count
    2
    >>> stage.estimated_duration_ms
    6000
    """

    module: TargetModule
    steps: tuple[PlanStep, ...]

    @property
    def step_count(self) -> int:
        """Return number of steps in this stage.

        Returns
        -------
        int
            Count of steps.
        """
        return len(self.steps)

    @property
    def estimated_duration_ms(self) -> int | None:
        """Calculate total estimated duration for this stage.

        Returns the sum of all step durations if all are known,
        or None if any step has unknown duration.

        Returns
        -------
        int | None
            Total milliseconds, or None if any step is unknown.
        """
        total = 0
        for step in self.steps:
            if step.estimated_duration_ms is None:
                return None
            total += step.estimated_duration_ms
        return total

    def to_dict(self) -> dict[str, Any]:
        """Serialize stage to dictionary for JSON output.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "module": self.module,
            "step_count": self.step_count,
            "estimated_duration_ms": self.estimated_duration_ms,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class BuildPlan:
    """Complete execution plan for a build request.

    Contains all information needed to execute a build: which targets
    to compute (grouped by stage), which to skip, and which are blocked.

    Attributes
    ----------
    requested_targets
        Original goal targets requested by the user.
    stages
        Execution stages in order (ingestion -> graphs -> analytics).
    skipped_targets
        Targets that are already up-to-date.
    blocked_targets
        Targets that cannot be computed due to external constraints.

    Examples
    --------
    >>> plan = BuildPlan(
    ...     requested_targets=("function_metrics",),
    ...     stages=(ingestion_stage, graphs_stage),
    ...     skipped_targets=("coverage",),
    ...     blocked_targets=(),
    ... )
    >>> plan.total_steps
    5
    >>> plan.is_empty()
    False
    """

    requested_targets: tuple[str, ...]
    stages: tuple[PlanStage, ...]
    skipped_targets: tuple[str, ...]
    blocked_targets: tuple[str, ...]

    @property
    def total_steps(self) -> int:
        """Return total number of steps across all stages.

        Returns
        -------
        int
            Sum of step counts from all stages.
        """
        return sum(stage.step_count for stage in self.stages)

    @property
    def estimated_duration_ms(self) -> int | None:
        """Calculate total estimated duration for the entire plan.

        Returns the sum of all stage durations if all are known,
        or None if any stage has unknown duration.

        Returns
        -------
        int | None
            Total milliseconds, or None if any stage is unknown.
        """
        total = 0
        for stage in self.stages:
            stage_duration = stage.estimated_duration_ms
            if stage_duration is None:
                return None
            total += stage_duration
        return total

    def is_empty(self) -> bool:
        """Check if there is no work to do.

        Returns
        -------
        bool
            True if no steps to execute.
        """
        return self.total_steps == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize plan to dictionary for JSON output.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for JSON serialization.
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
        """Format a human-readable summary string for CLI display.

        Returns
        -------
        str
            Multi-line summary suitable for terminal output.

        Examples
        --------
        >>> print(plan.format_summary())
        Build Plan for: function_metrics
        ============================================================
        <BLANKLINE>
        Stage 1: Ingestion (2 steps, ~6s)
          -> modules (No manifest exists)
          -> ast (Dependency cascade from: modules)
        ...
        """
        lines = [f"Build Plan for: {', '.join(self.requested_targets)}"]
        lines.append("=" * 60)

        for i, stage in enumerate(self.stages, 1):
            duration = format_duration(stage.estimated_duration_ms)
            lines.append(
                f"\nStage {i}: {stage.module.title()} ({stage.step_count} steps{duration})"
            )
            for step in stage.steps:
                reason_display = f" ({step.reason})" if step.reason else ""
                lines.append(f"  -> {step.target}{reason_display}")

        total_duration = format_duration(self.estimated_duration_ms)
        lines.append(f"\nTotal: {self.total_steps} steps{total_duration}")

        if self.skipped_targets:
            lines.append(f"Skipped: {len(self.skipped_targets)} targets (already current)")

        if self.blocked_targets:
            lines.append(f"Blocked: {len(self.blocked_targets)} targets")

        return "\n".join(lines)


# =============================================================================
# Plan Generator
# =============================================================================


class PlanGenerator:
    """Generate executable plans from resolution results.

    Transforms a ResolutionResult (what to compute) into a BuildPlan
    (how to compute it), grouping targets by module and preserving
    execution order.

    Parameters
    ----------
    graph
        Target graph for looking up target metadata.

    Examples
    --------
    >>> generator = PlanGenerator(graph)
    >>> plan = generator.generate(resolution)
    >>> print(plan.format_summary())
    Build Plan for: function_metrics
    ...
    """

    def __init__(self, graph: TargetGraph) -> None:
        """Initialize the plan generator.

        Parameters
        ----------
        graph
            Target graph containing all registered targets.
        """
        self._graph = graph

    def generate(self, resolution: ResolutionResult) -> BuildPlan:
        """Generate a build plan from resolution result.

        Groups targets by module and creates execution stages in the
        canonical order: ingestion -> graphs -> analytics.

        Parameters
        ----------
        resolution
            Resolution result specifying what needs to be computed.

        Returns
        -------
        BuildPlan
            Executable plan with stages, skipped, and blocked.

        Examples
        --------
        >>> plan = generator.generate(resolution)
        >>> plan.total_steps
        4
        """
        # Phase A: Group targets by module
        by_module: dict[TargetModule, list[str]] = {
            "ingestion": [],
            "graphs": [],
            "analytics": [],
            "export": [],
        }

        for target_name in resolution.to_compute:
            target = self._graph.get(target_name)
            by_module[target.module].append(target_name)

        # Phase B: Build stages in canonical order
        stages: list[PlanStage] = []
        for module in MODULE_ORDER:
            target_names = by_module[module]
            if target_names:
                stage = self._create_stage(module, target_names, resolution)
                stages.append(stage)

        return BuildPlan(
            requested_targets=resolution.requested,
            stages=tuple(stages),
            skipped_targets=resolution.to_skip,
            blocked_targets=resolution.blocked,
        )

    def _create_stage(
        self,
        module: TargetModule,
        target_names: list[str],
        resolution: ResolutionResult,
    ) -> PlanStage:
        """Create a PlanStage for a module.

        Parameters
        ----------
        module
            Target module for this stage.
        target_names
            Target names to include in this stage.
        resolution
            Resolution result for reason lookup.

        Returns
        -------
        PlanStage
            Stage containing steps for all targets.
        """
        steps = tuple(self._create_step(name, resolution) for name in target_names)
        return PlanStage(module=module, steps=steps)

    def _create_step(
        self,
        target_name: str,
        resolution: ResolutionResult,
    ) -> PlanStep:
        """Create a PlanStep for a target.

        Parameters
        ----------
        target_name
            Name of the target.
        resolution
            Resolution result for reason lookup.

        Returns
        -------
        PlanStep
            Step with target metadata and reason.
        """
        target = self._graph.get(target_name)
        reason = resolution.reasons.get(target_name)

        if reason is None:
            log.warning(
                "Target '%s' has no resolution reason, using empty string",
                target_name,
            )
            reason_str = ""
        else:
            reason_str = reason.details

        return PlanStep(
            target=target_name,
            module=target.module,
            plugin=target.plugin,
            estimated_duration_ms=target.estimated_duration_ms,
            dependencies=target.dependencies,
            reason=reason_str,
        )


__all__ = [
    "BuildPlan",
    "PlanGenerator",
    "PlanStage",
    "PlanStep",
    "format_duration",
]
