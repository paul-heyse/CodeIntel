"""Build plan and status helpers for tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.hamilton.planner import compute_plan
from codeintel.build.planning.model import BuildPlan, PlanRequest
from codeintel.build.state import StateValidationOptions, StateValidator
from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.state_types import BuildState
    from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness


@dataclass(frozen=True)
class PlanSummary:
    """Compact summary for plan or status results."""

    entries: Mapping[str, tuple[str, str | None]]


def compute_plan_summary(
    harness: HamiltonBuildHarness,
    targets: Iterable[str],
    *,
    runtime: HamiltonRuntimeBundle,
) -> PlanSummary:
    """Compute a summary of plan entries for requested targets.

    Returns
    -------
    PlanSummary
        Mapping of target -> (status, reason).
    """
    request = PlanRequest(
        requested_targets=tuple(targets),
        mode="predict",
        include_node_details=False,
        include_io_details=False,
        include_cache_details=False,
    )
    plan = compute_plan(
        env=harness.build_env(),
        plan_request=request,
        runtime=runtime,
        materialize=False,
    )
    return _summarize_plan(plan)


def compute_status_summary(
    harness: HamiltonBuildHarness,
    targets: Iterable[str],
    *,
    runtime: HamiltonRuntimeBundle,
) -> PlanSummary:
    """Compute a summary of build status for requested targets.

    Returns
    -------
    PlanSummary
        Mapping of target -> (status, reason).
    """
    env = harness.build_env()
    validator = StateValidator.from_runtime(
        runtime=runtime,
        env=env,
        options=StateValidationOptions(),
    )
    state = validator.validate()
    return _summarize_state(state, targets)


def format_plan_diff(expected: PlanSummary, actual: PlanSummary) -> str:
    """Format a readable diff between expected and actual summaries.

    Returns
    -------
    str
        Diff string describing missing, extra, and changed entries.
    """
    exp_keys = set(expected.entries)
    act_keys = set(actual.entries)
    missing = sorted(exp_keys - act_keys)
    extra = sorted(act_keys - exp_keys)
    common = sorted(exp_keys & act_keys)
    changed = [key for key in common if expected.entries[key] != actual.entries[key]]

    lines: list[str] = [
        f"plan diff expected={len(exp_keys)} actual={len(act_keys)}",
    ]
    if missing:
        lines.append(f"  missing: {missing}")
    if extra:
        lines.append(f"  extra: {extra}")
    if changed:
        lines.append("  changed:")
        for key in changed:
            exp_status, exp_reason = expected.entries[key]
            act_status, act_reason = actual.entries[key]
            lines.append(
                f"    - {key}: ({exp_status}, {exp_reason}) -> ({act_status}, {act_reason})"
            )
    if not (missing or extra or changed):
        lines.append("  (no differences)")
    return "\n".join(lines)


def _summarize_plan(plan: BuildPlan) -> PlanSummary:
    entries = {
        entry.target: (
            entry.predicted_action,
            ", ".join(entry.block_reasons) if entry.block_reasons else None,
        )
        for entry in plan.entries
    }
    return PlanSummary(entries=entries)


def _summarize_state(state: BuildState, targets: Iterable[str]) -> PlanSummary:
    entries: dict[str, tuple[str, str | None]] = {}
    for target in targets:
        target_state = state.get(target)
        if target_state is None:
            entries[target] = ("missing", None)
        else:
            entries[target] = (target_state.status, target_state.blocking_reason)
    return PlanSummary(entries=entries)


__all__ = [
    "PlanSummary",
    "compute_plan_summary",
    "compute_status_summary",
    "format_plan_diff",
]
