"""Minimal-work resolver for build targets.

This module computes the smallest set of targets to run for a set of goals,
given a precomputed database state (missing/computed/stale/blocked).

The resolver output is used by the legacy plan generator (`codeintel.build.plan`)
and unit tests; Hamilton-native planning lives in `codeintel.build.hamilton.planner`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from codeintel.build.state import TargetState

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.build.state import DatabaseState
    from codeintel.build.targets import TargetGraph, TargetModule

log = logging.getLogger(__name__)


ResolutionKind = Literal[
    "blocked_external",
    "cascade",
    "current",
    "dependency",
    "forced",
    "missing",
    "stale",
]


@dataclass(frozen=True)
class ResolutionReason:
    """Explain why a target is (re)computed, skipped, or blocked."""

    kind: ResolutionKind
    details: str


@dataclass(frozen=True)
class ResolutionResult:
    """Resolver output for a set of requested goals."""

    requested: tuple[str, ...]
    to_compute: tuple[str, ...]
    to_skip: tuple[str, ...]
    blocked: tuple[str, ...]
    reasons: Mapping[str, ResolutionReason]

    @property
    def total_work(self) -> int:
        """Number of targets to compute."""
        return len(self.to_compute)

    @property
    def total_skipped(self) -> int:
        """Number of targets skipped."""
        return len(self.to_skip)

    def is_empty(self) -> bool:
        """Return True when there is no work to do.

        Returns
        -------
        bool
            True when ``to_compute`` is empty.
        """
        return self.total_work == 0

    def get_reason(self, target: str) -> ResolutionReason:
        """Return the resolution reason for a target.

        Returns
        -------
        ResolutionReason
            Recorded reason for the target.

        Raises
        ------
        KeyError
            When the target has no recorded reason.
        """
        if target not in self.reasons:
            msg = f"Resolution reason for '{target}' not found"
            raise KeyError(msg)
        return self.reasons[target]


class BuildResolver:
    """Resolve the minimal computation set for build goals."""

    def __init__(self, graph: TargetGraph, state: DatabaseState) -> None:
        self._graph = graph
        self._state = state

    def resolve(
        self,
        goals: Sequence[str],
        *,
        force_recompute: Sequence[str] | None = None,
    ) -> ResolutionResult:
        """Resolve goals into to_compute/to_skip/blocked sets.

        Parameters
        ----------
        goals
            Requested target names.
        force_recompute
            Optional list of targets to force recomputation. Targets not in the
            transitive closure of goals are ignored with a warning.

        Returns
        -------
        ResolutionResult
            Resolver output containing compute/skip/blocked sets and reasons.

        Raises
        ------
        KeyError
            When requested goals include targets not registered in the graph.
        """
        if not goals:
            return ResolutionResult(
                requested=(),
                to_compute=(),
                to_skip=(),
                blocked=(),
                reasons={},
            )

        invalid = sorted({goal for goal in goals if goal not in self._graph})
        if invalid:
            msg = f"Unknown targets: {', '.join(invalid)}"
            raise KeyError(msg)

        closure = self._graph.topological_order(goals)
        force_set = self._filter_force_targets(force_recompute, closure=closure)
        requested_set = set(goals)

        to_compute: list[str] = []
        to_skip: list[str] = []
        blocked: list[str] = []
        reasons: dict[str, ResolutionReason] = {}

        compute_set: set[str] = set()

        for name in closure:
            state = self._state.targets.get(name)
            if state is None:
                state = TargetState(
                    name=name,
                    status="missing",
                    manifest=None,
                    staleness_reason=None,
                    blocking_deps=(),
                    current_input_hash=None,
                )

            deps = self._graph.get(name).dependencies
            dependency_computes = tuple(dep for dep in deps if dep in compute_set)
            dependency_blocked = tuple(dep for dep in deps if dep in blocked)

            if dependency_blocked:
                blocked.append(name)
                reasons[name] = self._annotate_goal(
                    name,
                    requested_set,
                    ResolutionReason(
                        kind="blocked_external",
                        details=(
                            "Blocked by external dependency: "
                            + ", ".join(sorted(dependency_blocked))
                        ),
                    ),
                )
                continue

            decision = self._decide(
                name,
                state,
                dependency_computes=dependency_computes,
                force_set=force_set,
            )

            reasons[name] = self._annotate_goal(name, requested_set, decision.reason)

            if decision.action == "compute":
                compute_set.add(name)
                to_compute.append(name)
            elif decision.action == "skip":
                to_skip.append(name)
            else:
                blocked.append(name)

        return ResolutionResult(
            requested=tuple(goals),
            to_compute=tuple(to_compute),
            to_skip=tuple(to_skip),
            blocked=tuple(blocked),
            reasons=reasons,
        )

    def resolve_all(self, *, module: TargetModule | None = None) -> ResolutionResult:
        """Resolve all targets, optionally filtering requested goals by module.

        Returns
        -------
        ResolutionResult
            Resolver output for the selected targets.
        """
        if module is None:
            goals = tuple(sorted(self._graph))
        else:
            goals = tuple(sorted(t.name for t in self._graph.targets_for_module(module)))
        return self.resolve(goals)

    def _filter_force_targets(
        self, force_recompute: Sequence[str] | None, *, closure: tuple[str, ...]
    ) -> set[str]:
        if not force_recompute:
            return set()

        closure_set = set(closure)
        force_set: set[str] = set()
        for name in force_recompute:
            if name not in self._graph:
                log.warning("Force target '%s' is not registered; ignoring", name)
                continue
            if name not in closure_set:
                log.warning(
                    "Force target '%s' is not in transitive deps of requested goals; ignoring", name
                )
                continue
            force_set.add(name)
        return force_set

    @staticmethod
    def _annotate_goal(
        name: str,
        requested: set[str],
        reason: ResolutionReason,
    ) -> ResolutionReason:
        if name not in requested:
            return reason
        if "goal" in reason.details.lower():
            return reason
        return ResolutionReason(kind=reason.kind, details=f"Requested goal: {reason.details}")

    def _decide(
        self,
        name: str,
        state: TargetState,
        *,
        dependency_computes: tuple[str, ...],
        force_set: set[str],
    ) -> _Decision:
        action: DecisionAction
        reason: ResolutionReason

        if name in force_set:
            action = "compute"
            reason = ResolutionReason(kind="forced", details="Forced recompute")
        elif state.status == "missing":
            action = "compute"
            reason = ResolutionReason(kind="missing", details="No manifest found; compute required")
        elif state.status == "stale":
            details = (
                state.staleness_reason.details
                if state.staleness_reason is not None
                else "Inputs changed"
            )
            action = "compute"
            reason = ResolutionReason(kind="stale", details=details)
        elif state.status == "blocked":
            external = tuple(dep for dep in state.blocking_deps if dep not in self._graph)
            if external:
                action = "blocked"
                reason = ResolutionReason(
                    kind="blocked_external",
                    details="Blocked by external dependencies: " + ", ".join(sorted(external)),
                )
            elif dependency_computes:
                action = "compute"
                reason = ResolutionReason(
                    kind="dependency",
                    details="Dependencies will run first: " + ", ".join(sorted(dependency_computes)),
                )
            else:
                action = "blocked"
                reason = ResolutionReason(
                    kind="blocked_external",
                    details="Blocked by dependencies: " + ", ".join(sorted(state.blocking_deps)),
                )
        elif dependency_computes:
            action = "compute"
            reason = ResolutionReason(
                kind="cascade",
                details="Dependency cascade from: " + ", ".join(sorted(dependency_computes)),
            )
        else:
            action = "skip"
            reason = ResolutionReason(kind="current", details="Up-to-date")

        return _Decision(action=action, reason=reason)


DecisionAction = Literal["blocked", "compute", "skip"]


@dataclass(frozen=True)
class _Decision:
    action: DecisionAction
    reason: ResolutionReason


__all__ = [
    "BuildResolver",
    "ResolutionReason",
    "ResolutionResult",
]
