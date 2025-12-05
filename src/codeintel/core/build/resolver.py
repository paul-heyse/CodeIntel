"""Minimal work resolver for the build system.

This module computes the minimal set of targets that must be (re)computed
to bring requested goals up-to-date. It takes user-requested goals and the
current database state (from Phase 2's StateValidator), then determines:

- **to_compute**: Targets that need (re)computation, in topological order
- **to_skip**: Targets already up-to-date that can be skipped
- **blocked**: Targets that cannot be computed due to external constraints
- **reasons**: Human-readable explanation for each decision

Key Concepts
------------
- **Cascade Invalidation**: When a dependency is recomputed, all downstream
  targets must also be recomputed, even if their own state shows "computed".
- **Topological Ordering**: Targets are processed in dependency order so that
  dependencies are always computed before their dependents.
- **Force Recompute**: Users can force specific targets to recompute, which
  triggers cascade invalidation to all dependents.

Algorithm Overview
------------------
The resolution proceeds in three phases:

1. **Phase A (Expand)**: Validate goals and expand to include all transitive
   dependencies.

2. **Phase B (Categorize)**: Walk targets in topological order, categorizing
   each as needing computation or being skippable based on state and cascade.

3. **Phase C (Build Result)**: Assemble the final result with proper ordering
   and reason tracking.

Integration Points
------------------
- Uses `TargetGraph` from Phase 1 for dependency traversal
- Uses `DatabaseState` from Phase 2 for current target states
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from codeintel.core.build.state import DatabaseState, TargetState
    from codeintel.core.build.targets import TargetGraph, TargetModule

log = logging.getLogger(__name__)

# =============================================================================
# Type Definitions
# =============================================================================

ResolutionKind = Literal[
    "requested",
    "dependency",
    "missing",
    "stale",
    "cascade",
    "forced",
    "blocked_external",
    "current",
]
"""Classification of why a target is in a particular resolution bucket.

- ``requested``: User explicitly requested this target as a goal
- ``dependency``: Required as a transitive dependency of a goal
- ``missing``: No manifest exists, must compute
- ``stale``: Manifest exists but input hash differs, must recompute
- ``cascade``: Dependency is being recomputed, must cascade
- ``forced``: User forced recomputation via force_recompute
- ``blocked_external``: Blocked by external constraint, cannot compute
- ``current``: Already up-to-date, can skip
"""


@dataclass(frozen=True)
class ResolutionReason:
    """Structured reason for a resolution decision.

    Provides machine-readable classification and human-readable details
    explaining why a target ended up in its particular bucket.

    Attributes
    ----------
    kind
        Classification of the resolution decision.
    details
        Human-readable explanation with specific context.

    Examples
    --------
    >>> reason = ResolutionReason(
    ...     kind="cascade",
    ...     details="Dependency 'ast' is being recomputed",
    ... )
    >>> reason.kind
    'cascade'
    """

    kind: ResolutionKind
    details: str


@dataclass(frozen=True)
class ResolutionResult:
    """Complete result of resolving what work needs to be done.

    Contains the categorization of all targets involved in satisfying
    the requested goals, with clear reasons for each decision.

    Attributes
    ----------
    requested
        Original goal target names requested by the user.
    to_compute
        Targets that need (re)computation, in topological order.
    to_skip
        Targets that are already up-to-date and can be skipped.
    blocked
        Targets blocked by external constraints (cannot be computed).
    reasons
        Mapping of target names to their resolution reasons.

    Examples
    --------
    >>> result = ResolutionResult(
    ...     requested=("function_metrics",),
    ...     to_compute=("modules", "ast", "goids", "function_metrics"),
    ...     to_skip=(),
    ...     blocked=(),
    ...     reasons={},
    ... )
    >>> result.total_work
    4
    """

    requested: tuple[str, ...]
    to_compute: tuple[str, ...]
    to_skip: tuple[str, ...]
    blocked: tuple[str, ...]
    reasons: Mapping[str, ResolutionReason]

    @property
    def total_work(self) -> int:
        """Return count of targets that need computation.

        Returns
        -------
        int
            Number of targets in to_compute.
        """
        return len(self.to_compute)

    @property
    def total_skipped(self) -> int:
        """Return count of targets that can be skipped.

        Returns
        -------
        int
            Number of targets in to_skip.
        """
        return len(self.to_skip)

    def is_empty(self) -> bool:
        """Check if there is no work to do.

        Returns
        -------
        bool
            True if to_compute is empty (all goals already satisfied).
        """
        return len(self.to_compute) == 0

    def get_reason(self, name: str) -> ResolutionReason:
        """Retrieve the reason for a specific target.

        Parameters
        ----------
        name
            Target name to look up.

        Returns
        -------
        ResolutionReason
            The reason for this target's categorization.

        Raises
        ------
        KeyError
            If target name is not in the resolution result.
        """
        if name not in self.reasons:
            msg = f"Target '{name}' not found in resolution result"
            raise KeyError(msg)
        return self.reasons[name]


# =============================================================================
# Build Resolver
# =============================================================================


class BuildResolver:
    """Resolve minimal work needed to achieve goal targets.

    Takes user-requested goals and current database state, then computes
    the minimal set of targets that must be (re)computed. Handles cascade
    invalidation when dependencies change.

    Parameters
    ----------
    graph
        Target graph defining all outputs and their dependencies.
    state
        Current database state from StateValidator.

    Examples
    --------
    >>> resolver = BuildResolver(graph, state)
    >>> result = resolver.resolve(["function_metrics"])
    >>> result.to_compute
    ('modules', 'ast', 'goids', 'function_metrics')
    """

    def __init__(
        self,
        graph: TargetGraph,
        state: DatabaseState,
    ) -> None:
        """Initialize the build resolver.

        Parameters
        ----------
        graph
            Target graph with all registered targets.
        state
            Current database state from validation.
        """
        self._graph = graph
        self._state = state

    def resolve(
        self,
        goals: Iterable[str],
        force_recompute: Iterable[str] | None = None,
    ) -> ResolutionResult:
        """Compute minimal work to make goals up-to-date.

        Expands goals to include transitive dependencies, then categorizes
        each target as needing computation or being skippable. Handles
        cascade invalidation when dependencies are recomputed.

        Parameters
        ----------
        goals
            Target names that must be up-to-date after execution.
        force_recompute
            Optional targets to recompute even if not stale.

        Returns
        -------
        ResolutionResult
            Resolution with targets categorized into to_compute, to_skip,
            and blocked, with reasons for each decision.

        Examples
        --------
        >>> result = resolver.resolve(["function_metrics"])
        >>> print(f"Need to compute {result.total_work} targets")
        Need to compute 4 targets
        """
        # Convert to tuples for consistency
        goal_list = tuple(goals)
        force_set = set(force_recompute) if force_recompute else set()

        # Handle empty goals case
        if not goal_list:
            return ResolutionResult(
                requested=(),
                to_compute=(),
                to_skip=(),
                blocked=(),
                reasons={},
            )

        # Phase A: Validate and expand
        self._validate_goals(goal_list)
        all_needed = self._expand_transitive(goal_list)
        force_set = self._filter_force_targets(force_set, all_needed)

        # Phase B: Categorize targets
        must_compute, to_skip, blocked, reasons = self._categorize_targets(
            goal_list, all_needed, force_set
        )

        # Phase C: Build result with proper ordering
        return self._build_result(goal_list, must_compute, to_skip, blocked, reasons)

    def resolve_all(
        self,
        module: TargetModule | None = None,
    ) -> ResolutionResult:
        """Resolve work needed for all targets.

        Convenience method that resolves all targets in the graph,
        optionally filtered to a specific module.

        Parameters
        ----------
        module
            Optional module filter ("ingestion", "graphs", or "analytics").
            If None, resolves all targets.

        Returns
        -------
        ResolutionResult
            Resolution for all (filtered) targets.

        Examples
        --------
        >>> result = resolver.resolve_all(module="analytics")
        >>> print(f"Analytics needs {result.total_work} targets computed")
        Analytics needs 15 targets computed
        """
        if module is not None:
            goals = [t.name for t in self._graph.targets_for_module(module)]
        else:
            goals = list(self._graph)
        return self.resolve(goals)

    # =========================================================================
    # Phase A: Validation and Expansion
    # =========================================================================

    def _validate_goals(self, goals: tuple[str, ...]) -> None:
        """Validate that all goals exist in the graph.

        Parameters
        ----------
        goals
            Target names to validate.

        Raises
        ------
        KeyError
            If any goal is not in the graph.
        """
        unknown = [g for g in goals if g not in self._graph]
        if unknown:
            msg = f"Unknown target(s): {', '.join(sorted(unknown))}"
            raise KeyError(msg)

    def _expand_transitive(self, goals: tuple[str, ...]) -> set[str]:
        """Expand goals to include all transitive dependencies.

        Parameters
        ----------
        goals
            Goal target names.

        Returns
        -------
        set[str]
            All target names needed (goals + transitive deps).
        """
        all_needed: set[str] = set()
        for goal in goals:
            all_needed.add(goal)
            all_needed.update(self._graph.transitive_deps(goal))
        return all_needed

    def _filter_force_targets(
        self,
        force_set: set[str],
        all_needed: set[str],
    ) -> set[str]:
        """Filter force_recompute to only include relevant targets.

        Logs warnings for force targets not in the needed set.

        Parameters
        ----------
        force_set
            Original force_recompute targets.
        all_needed
            All targets needed for the goals.

        Returns
        -------
        set[str]
            Filtered force set (intersection with all_needed).
        """
        graph_targets = set(self._graph)
        unknown_force = force_set - all_needed - graph_targets
        for target in unknown_force:
            log.warning("Force target '%s' is not in the graph, ignoring", target)

        irrelevant_force = force_set - all_needed
        for target in irrelevant_force - unknown_force:
            log.warning(
                "Force target '%s' is not in transitive deps of goals, ignoring",
                target,
            )

        return force_set & all_needed

    # =========================================================================
    # Phase B: Categorization
    # =========================================================================

    def _categorize_targets(
        self,
        goals: tuple[str, ...],
        all_needed: set[str],
        force_set: set[str],
    ) -> tuple[set[str], set[str], set[str], dict[str, ResolutionReason]]:
        """Categorize targets into compute/skip/blocked buckets.

        Walks targets in topological order, checking state and cascade
        invalidation to determine each target's fate.

        Parameters
        ----------
        goals
            Original goal targets (for reason tracking).
        all_needed
            All targets needed (goals + transitive deps).
        force_set
            Targets to force recompute.

        Returns
        -------
        tuple[set[str], set[str], set[str], dict[str, ResolutionReason]]
            (must_compute, to_skip, blocked, reasons) sets and mapping.
        """
        must_compute: set[str] = set()
        to_skip: set[str] = set()
        blocked: set[str] = set()
        reasons: dict[str, ResolutionReason] = {}

        goal_set = set(goals)

        # Process in topological order (deps before dependents)
        topo_order = self._graph.topological_order(all_needed)

        for target_name in topo_order:
            reason = self._categorize_single_target(target_name, goal_set, force_set, must_compute)
            reasons[target_name] = reason

            if reason.kind in {"missing", "stale", "cascade", "forced", "dependency"}:
                must_compute.add(target_name)
            elif reason.kind == "blocked_external":
                blocked.add(target_name)
            else:  # current
                to_skip.add(target_name)

        return must_compute, to_skip, blocked, reasons

    def _categorize_single_target(
        self,
        target_name: str,
        goal_set: set[str],
        force_set: set[str],
        must_compute: set[str],
    ) -> ResolutionReason:
        """Categorize a single target.

        Parameters
        ----------
        target_name
            Target to categorize.
        goal_set
            Set of original goal targets.
        force_set
            Set of forced recompute targets.
        must_compute
            Targets already marked for computation.

        Returns
        -------
        ResolutionReason
            Reason for this target's categorization.
        """
        target_state = self._state.get(target_name)
        is_goal = target_name in goal_set

        # Check if forced
        if target_name in force_set:
            return self._make_forced_reason(is_goal=is_goal)

        # Dispatch based on state status
        return self._categorize_by_state(
            target_name, target_state, is_goal=is_goal, must_compute=must_compute
        )

    @staticmethod
    def _make_forced_reason(*, is_goal: bool) -> ResolutionReason:
        """Create reason for forced recomputation.

        Parameters
        ----------
        is_goal
            Whether this target was explicitly requested.

        Returns
        -------
        ResolutionReason
            Forced reason with goal annotation if applicable.
        """
        detail = "User requested forced recomputation"
        if is_goal:
            detail += " (requested goal)"
        return ResolutionReason(kind="forced", details=detail)

    def _categorize_by_state(
        self,
        target_name: str,
        target_state: TargetState,
        *,
        is_goal: bool,
        must_compute: set[str],
    ) -> ResolutionReason:
        """Categorize target based on its state status.

        Parameters
        ----------
        target_name
            Target name for cascade checking.
        target_state
            Current state of the target.
        is_goal
            Whether this target was explicitly requested.
        must_compute
            Targets already marked for computation.

        Returns
        -------
        ResolutionReason
            Reason based on target state.
        """
        if target_state.status == "missing":
            return self._make_missing_reason(is_goal=is_goal)

        if target_state.status == "stale":
            return self._make_stale_reason(target_state, is_goal=is_goal)

        if target_state.status == "blocked":
            return self._make_blocked_reason(target_state, must_compute=must_compute)

        # Status is "computed" - check cascade invalidation
        return self._make_computed_reason(target_name, is_goal=is_goal, must_compute=must_compute)

    @staticmethod
    def _make_missing_reason(*, is_goal: bool) -> ResolutionReason:
        """Create reason for missing target.

        Parameters
        ----------
        is_goal
            Whether this target was explicitly requested.

        Returns
        -------
        ResolutionReason
            Missing reason with goal annotation if applicable.
        """
        detail = "No manifest exists, must compute"
        if is_goal:
            detail += " (requested goal)"
        return ResolutionReason(kind="missing", details=detail)

    @staticmethod
    def _make_stale_reason(
        target_state: TargetState,
        *,
        is_goal: bool,
    ) -> ResolutionReason:
        """Create reason for stale target.

        Parameters
        ----------
        target_state
            State containing staleness details.
        is_goal
            Whether this target was explicitly requested.

        Returns
        -------
        ResolutionReason
            Stale reason with details and goal annotation.
        """
        staleness_detail = ""
        if target_state.staleness_reason:
            staleness_detail = f": {target_state.staleness_reason.details}"
        detail = f"Target is stale{staleness_detail}"
        if is_goal:
            detail += " (requested goal)"
        return ResolutionReason(kind="stale", details=detail)

    @staticmethod
    def _make_blocked_reason(
        target_state: TargetState,
        *,
        must_compute: set[str],
    ) -> ResolutionReason:
        """Create reason for blocked target.

        Parameters
        ----------
        target_state
            State containing blocking dependencies.
        must_compute
            Targets already marked for computation.

        Returns
        -------
        ResolutionReason
            Dependency or blocked_external reason.
        """
        blocking_deps_will_compute = all(dep in must_compute for dep in target_state.blocking_deps)
        if blocking_deps_will_compute:
            detail = f"Dependencies {target_state.blocking_deps} will be computed first"
            return ResolutionReason(kind="dependency", details=detail)
        # Truly blocked by external constraint
        unresolved = [d for d in target_state.blocking_deps if d not in must_compute]
        detail = f"Blocked by unresolved dependencies: {tuple(unresolved)}"
        return ResolutionReason(kind="blocked_external", details=detail)

    def _make_computed_reason(
        self,
        target_name: str,
        *,
        is_goal: bool,
        must_compute: set[str],
    ) -> ResolutionReason:
        """Create reason for computed target (cascade or current).

        Parameters
        ----------
        target_name
            Target to check for cascade.
        is_goal
            Whether this target was explicitly requested.
        must_compute
            Targets already marked for computation.

        Returns
        -------
        ResolutionReason
            Cascade or current reason.
        """
        if self._needs_cascade_recompute(target_name, must_compute):
            cascading_deps = self._get_cascading_deps(target_name, must_compute)
            detail = f"Dependency cascade from: {tuple(cascading_deps)}"
            return ResolutionReason(kind="cascade", details=detail)

        # Target is current and can be skipped
        detail = "Already computed and up-to-date"
        if is_goal:
            detail += " (requested goal)"
        return ResolutionReason(kind="current", details=detail)

    def _needs_cascade_recompute(
        self,
        target_name: str,
        must_compute: set[str],
    ) -> bool:
        """Check if target needs recompute due to dependency cascade.

        If any direct dependency is being recomputed, this target must also
        be recomputed even if its own state shows 'computed'.

        Parameters
        ----------
        target_name
            Target to check.
        must_compute
            Set of targets already marked for computation.

        Returns
        -------
        bool
            True if cascade recompute is needed.
        """
        target = self._graph.get(target_name)
        return any(dep in must_compute for dep in target.dependencies)

    def _get_cascading_deps(
        self,
        target_name: str,
        must_compute: set[str],
    ) -> list[str]:
        """Get the dependencies causing cascade invalidation.

        Parameters
        ----------
        target_name
            Target being cascaded.
        must_compute
            Set of targets marked for computation.

        Returns
        -------
        list[str]
            Dependencies that are in must_compute (causing cascade).
        """
        target = self._graph.get(target_name)
        return [dep for dep in target.dependencies if dep in must_compute]

    # =========================================================================
    # Phase C: Build Result
    # =========================================================================

    def _build_result(
        self,
        goals: tuple[str, ...],
        must_compute: set[str],
        to_skip: set[str],
        blocked: set[str],
        reasons: dict[str, ResolutionReason],
    ) -> ResolutionResult:
        """Build the final resolution result with proper ordering.

        Parameters
        ----------
        goals
            Original goal targets.
        must_compute
            Targets to compute.
        to_skip
            Targets to skip.
        blocked
            Blocked targets.
        reasons
            Reason mapping.

        Returns
        -------
        ResolutionResult
            Final structured result.
        """
        # Order to_compute in topological order
        ordered_compute = self._graph.topological_order(must_compute) if must_compute else ()

        return ResolutionResult(
            requested=goals,
            to_compute=ordered_compute,
            to_skip=tuple(sorted(to_skip)),
            blocked=tuple(sorted(blocked)),
            reasons=reasons,
        )


__all__ = [
    "BuildResolver",
    "ResolutionKind",
    "ResolutionReason",
    "ResolutionResult",
]
