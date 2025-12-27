"""Unified state types for the build system.

This module defines canonical types for representing build target state,
unifying concepts from both state.py and readiness.py into a coherent
type hierarchy.

The key insight is that state.py and readiness.py compute the same
underlying facts but expose them through different lenses:

- **state.py**: "What is the current status?" (validation perspective)
- **readiness.py**: "What needs to happen?" (planning perspective)

These unified types capture the underlying facts once, allowing both
views to be derived without redundant computation.

Type Hierarchy
--------------
- ``TargetStatus``: Canonical status enumeration
- ``BlockingReason``: Why a target cannot be computed
- ``TargetState``: Complete state for a single target
- ``BuildState``: Aggregate state for all targets in a snapshot
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.build_manifest import OutputManifest

__all__ = [
    "BlockingReason",
    "BuildState",
    "TargetState",
    "TargetStatus",
]


TargetStatus = Literal["current", "stale", "missing", "blocked"]
"""Unified status for a build target.

- ``current``: Manifest exists, input hash matches, data exists
- ``stale``: Manifest exists but input hash differs
- ``missing``: No manifest exists for this target
- ``blocked``: Dependencies are not ready (missing, stale, or blocked)

The status is computed purely from observable facts:
1. Does a manifest exist?
2. Does the stored input_hash match the current computation?
3. Are all dependencies in "current" status?
"""


BlockingReason = Literal[
    "input_hash_mismatch",
    "dependency_missing",
    "dependency_stale",
    "dependency_blocked",
    "options_hash_mismatch",
    "data_missing",
]
"""Reason why a target is not in "current" status.

- ``input_hash_mismatch``: Stored hash differs from current computation
- ``dependency_missing``: A dependency has never been computed
- ``dependency_stale``: A dependency needs recomputation
- ``dependency_blocked``: A dependency is itself blocked
- ``options_hash_mismatch``: Plugin options have changed
- ``data_missing``: Output tables are empty despite manifest existing
"""


@dataclass(frozen=True)
class TargetState:
    """Unified state for a single target.

    Captures all observable facts about a target's current state,
    from which both status validation and action planning can be derived.

    Attributes
    ----------
    name
        Target identifier matching TargetDescriptor.name.
    status
        Computed status: current, stale, missing, or blocked.
    manifest
        Stored manifest if one exists, None otherwise.
    current_hash
        Computed input hash for current snapshot (None if not computed).
    blocking_reason
        Primary reason for non-current status (None if current).
    blocking_deps
        Dependencies causing blocked status (empty if not blocked).
    stored_hash
        Input hash from manifest (None if no manifest).

    Examples
    --------
    >>> state = TargetState(
    ...     name="ast",
    ...     status="current",
    ...     manifest=manifest,
    ...     current_hash="abc123def456",
    ... )
    >>> state.is_current
    True

    >>> state = TargetState(
    ...     name="call_graph",
    ...     status="blocked",
    ...     manifest=None,
    ...     current_hash=None,
    ...     blocking_reason="dependency_missing",
    ...     blocking_deps=("ast",),
    ... )
    >>> state.needs_computation
    False  # Can't run until deps are ready
    """

    name: str
    status: TargetStatus
    manifest: OutputManifest | None
    current_hash: str | None = None
    blocking_reason: BlockingReason | None = None
    blocking_deps: tuple[str, ...] = ()
    stored_hash: str | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate state consistency."""
        # If we have a manifest, extract stored_hash if not provided
        if self.manifest is not None and self.stored_hash is None:
            object.__setattr__(self, "stored_hash", self.manifest.input_hash)

    @property
    def is_current(self) -> bool:
        """Check if target is up-to-date.

        Returns
        -------
        bool
            True if status is "current".
        """
        return self.status == "current"

    @property
    def is_stale(self) -> bool:
        """Check if target needs recomputation.

        Returns
        -------
        bool
            True if status is "stale".
        """
        return self.status == "stale"

    @property
    def is_missing(self) -> bool:
        """Check if target has never been computed.

        Returns
        -------
        bool
            True if status is "missing".
        """
        return self.status == "missing"

    @property
    def is_blocked(self) -> bool:
        """Check if target is blocked by dependencies.

        Returns
        -------
        bool
            True if status is "blocked".
        """
        return self.status == "blocked"

    @property
    def needs_computation(self) -> bool:
        """Check if target needs to be computed.

        A target needs computation if it's missing or stale.
        Blocked targets cannot be computed until deps are ready.

        Returns
        -------
        bool
            True if target is missing or stale.
        """
        return self.status in {"missing", "stale"}

    @property
    def can_run(self) -> bool:
        """Check if target can be executed now.

        A target can run if it needs computation and is not blocked.

        Returns
        -------
        bool
            True if target is runnable.
        """
        return self.needs_computation and not self.is_blocked

    @property
    def hash_mismatch(self) -> bool:
        """Check if input hashes differ.

        Returns
        -------
        bool
            True if stored hash differs from current hash.
        """
        if self.stored_hash is None or self.current_hash is None:
            return False
        return self.stored_hash != self.current_hash


@dataclass(frozen=True)
class BuildState:
    """Complete state for all targets in a snapshot.

    Provides query methods to filter and aggregate target states,
    enabling both status reporting and execution planning.

    Attributes
    ----------
    repo
        Repository slug.
    commit
        Commit SHA.
    targets
        Mapping of target names to their states.

    Examples
    --------
    >>> state = BuildState(repo="org/repo", commit="abc123", targets={...})
    >>> state.by_status("missing")
    ('ast', 'modules', 'goids')
    >>> state.is_current("call_graph")
    False
    """

    repo: str
    commit: str
    targets: Mapping[str, TargetState]

    def get(self, name: str) -> TargetState:
        """Retrieve state for a specific target.

        Parameters
        ----------
        name
            Target name to look up.

        Returns
        -------
        TargetState
            State of the requested target.

        Raises
        ------
        KeyError
            If target name is not found.
        """
        if name not in self.targets:
            msg = f"Target '{name}' not found in build state"
            raise KeyError(msg)
        return self.targets[name]

    def by_status(self, status: TargetStatus) -> tuple[str, ...]:
        """Return target names with the specified status.

        Parameters
        ----------
        status
            Status to filter by.

        Returns
        -------
        tuple[str, ...]
            Sorted target names with matching status.
        """
        return tuple(sorted(name for name, state in self.targets.items() if state.status == status))

    def is_current(self, name: str) -> bool:
        """Check if a target is up-to-date.

        Parameters
        ----------
        name
            Target name to check.

        Returns
        -------
        bool
            True if target exists and has status "current".
        """
        if name not in self.targets:
            return False
        return self.targets[name].is_current

    def get_blockers(self, name: str) -> tuple[str, ...]:
        """Return blocking dependencies for a target.

        Parameters
        ----------
        name
            Target name to check.

        Returns
        -------
        tuple[str, ...]
            Names of blocking dependencies (empty if not blocked).
        """
        if name not in self.targets:
            return ()
        return self.targets[name].blocking_deps

    def runnable_targets(self) -> tuple[str, ...]:
        """Return targets that can be executed now.

        Returns
        -------
        tuple[str, ...]
            Sorted names of targets that need computation and aren't blocked.
        """
        return tuple(sorted(name for name, state in self.targets.items() if state.can_run))

    def current_targets(self) -> tuple[str, ...]:
        """Return up-to-date targets.

        Returns
        -------
        tuple[str, ...]
            Sorted names of targets with status "current".
        """
        return self.by_status("current")

    def missing_targets(self) -> tuple[str, ...]:
        """Return targets that have never been computed.

        Returns
        -------
        tuple[str, ...]
            Sorted names of targets with status "missing".
        """
        return self.by_status("missing")

    def stale_targets(self) -> tuple[str, ...]:
        """Return targets that need recomputation.

        Returns
        -------
        tuple[str, ...]
            Sorted names of targets with status "stale".
        """
        return self.by_status("stale")

    def blocked_targets(self) -> tuple[str, ...]:
        """Return targets blocked by dependencies.

        Returns
        -------
        tuple[str, ...]
            Sorted names of targets with status "blocked".
        """
        return self.by_status("blocked")

    @property
    def all_current(self) -> bool:
        """Check if all targets are up-to-date.

        Returns
        -------
        bool
            True if every target has status "current".
        """
        return all(state.is_current for state in self.targets.values())

    @property
    def summary(self) -> dict[TargetStatus, int]:
        """Return count of targets by status.

        Returns
        -------
        dict[TargetStatus, int]
            Mapping of status to count.
        """
        counts: dict[TargetStatus, int] = {
            "current": 0,
            "stale": 0,
            "missing": 0,
            "blocked": 0,
        }
        for state in self.targets.values():
            counts[state.status] += 1
        return counts
