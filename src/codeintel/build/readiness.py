"""Implicit readiness computation from state variables.

This module provides a declarative view of system state where derived properties
are automatically computed from primary facts (manifests, data existence, hashes).
Instead of asking "what's the state?" and then deciding what to do, consumers
can ask "what's ready?" and "what would make X ready?" - the reasoning is implicit.

Key Concepts
------------
- **ReadinessStatus**: Am I ready? If not, why not?
- **BlockerInfo**: What's blocking me and why?
- **TargetReadinessView**: A single target's computed readiness state
- **DatabaseReadinessView**: System-wide readiness queries

The readiness model is **purely computed** - nothing is stored. All properties
derive from three primary facts:

1. Manifest exists (was this target ever computed for this snapshot?)
2. Input hash matches (is the computation still valid?)
3. Data exists (do the output tables have rows?)

From these facts plus the dependency graph, we derive:
- Self status (my own state, ignoring dependencies)
- Dependency status (are my deps satisfied?)
- Blocker chain (full path from me to the ultimate bottleneck)
- Action needed (what to do to make me ready)

This module delegates to BuildSession for efficient caching of manifests
and input hashes, avoiding redundant computation.

Usage
-----
>>> from codeintel.build.readiness import DatabaseReadinessView
>>> view = DatabaseReadinessView(graph, gateway, snapshot)
>>>
>>> view["function_profile"].is_ready
False
>>> view["function_profile"].action_needed
ActionNeeded(kind='run_first', target='ast', reason='data missing')
>>>
>>>
>>> view.ready_targets()
('ast', 'typing', 'coverage')
>>>
>>>
>>> for name, readiness in view.blocked_targets():
...     print(f"{name}: {readiness.fix_command}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from codeintel.build.session import BuildSession

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from codeintel.build.manifest import OutputManifest
    from codeintel.build.targets import OutputTarget, TargetGraph, TargetModule
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


SelfStatus = Literal["current", "stale", "never_computed", "data_missing"]
"""Status of a target considering only its own state, ignoring dependencies.

- ``current``: Manifest exists, hash matches, data exists
- ``stale``: Manifest exists but hash doesn't match
- ``never_computed``: No manifest exists
- ``data_missing``: Output tables have no data
"""

DependencyStatusKind = Literal["satisfied", "blocked"]
"""Whether dependencies are satisfied or blocked."""

ActionKind = Literal["none", "run", "run_first", "blocked_external"]
"""What action is needed to make a target ready.

- ``none``: Target is already ready
- ``run``: Target can run now (all deps satisfied)
- ``run_first``: Another target must run first
- ``blocked_external``: Blocked by something outside the build system
"""


@dataclass(frozen=True)
class BlockerInfo:
    """Information about a single blocker in the chain.

    Attributes
    ----------
    target
        Name of the blocked target.
    blocked_by
        Name of the blocking target (None if this is the bottleneck).
    reason
        Why this target is blocked or needs computation.
    """

    target: str
    blocked_by: str | None
    reason: str


@dataclass(frozen=True)
class DependencyStatus:
    """Status of a target's dependencies.

    Attributes
    ----------
    kind
        Whether dependencies are satisfied or blocked.
    blockers
        Names of blocking dependencies (empty if satisfied).
    first_blocker
        Name of the first blocking dependency (None if satisfied).
    """

    kind: DependencyStatusKind
    blockers: tuple[str, ...] = ()
    first_blocker: str | None = None

    @property
    def is_satisfied(self) -> bool:
        """True if all dependencies are satisfied."""
        return self.kind == "satisfied"

    @property
    def is_blocked(self) -> bool:
        """True if any dependency is blocking."""
        return self.kind == "blocked"


@dataclass(frozen=True)
class ActionNeeded:
    """What action is needed to make a target ready.

    Attributes
    ----------
    kind
        Type of action needed.
    target
        Target to run (for 'run' or 'run_first').
    reason
        Human-readable explanation.
    command
        Suggested CLI command to fix.
    """

    kind: ActionKind
    target: str | None = None
    reason: str | None = None
    command: str | None = None

    @property
    def is_ready(self) -> bool:
        """True if no action is needed (target is ready)."""
        return self.kind == "none"


@dataclass(frozen=True)
class TargetReadiness:
    """Complete readiness state for a target.

    This is the main output of readiness computation, containing all
    derived properties for a single target.

    Attributes
    ----------
    name
        Target name.
    self_status
        Status considering only this target's own state.
    dependency_status
        Status of this target's dependencies.
    blocker_chain
        Full chain from this target to the ultimate bottleneck.
    action_needed
        What action would make this target ready.
    is_ready
        True if target is ready (no action needed).
    ultimate_bottleneck
        The deepest target that can actually run (None if ready).
    estimated_time_to_ready_ms
        Estimated time to make this target ready.
    """

    name: str
    self_status: SelfStatus
    dependency_status: DependencyStatus
    blocker_chain: tuple[BlockerInfo, ...] = ()
    action_needed: ActionNeeded = field(default_factory=lambda: ActionNeeded(kind="none"))
    ultimate_bottleneck: str | None = None
    estimated_time_to_ready_ms: int | None = None

    @property
    def is_ready(self) -> bool:
        """True if target is ready (computed, current, data exists)."""
        return self.action_needed.is_ready

    @property
    def is_blocked(self) -> bool:
        """True if target is blocked by dependencies."""
        return self.dependency_status.is_blocked

    @property
    def can_run(self) -> bool:
        """True if target can run now (deps satisfied, but needs computation)."""
        return self.action_needed.kind == "run" and self.action_needed.target == self.name

    @property
    def fix_command(self) -> str | None:
        """Suggested CLI command to make this target ready."""
        return self.action_needed.command


class TargetReadinessView:
    """Computed readiness view for a single target.

    This class wraps an OutputTarget and provides lazily-computed
    readiness properties derived from the current system state.

    All properties are computed on access from primary facts:
    - Manifest existence and content
    - Input hash computation
    - Dependency graph traversal

    Uses BuildSession for efficient caching of manifests and hashes.

    Parameters
    ----------
    target
        The target to compute readiness for.
    graph
        Complete target graph for dependency lookups.
    session
        Build session for caching and storage access.
    manifest_cache
        Optional pre-loaded manifests (deprecated, use session).
    """

    def __init__(
        self,
        target: OutputTarget,
        graph: TargetGraph,
        session: BuildSession,
        manifest_cache: Mapping[str, OutputManifest] | None = None,
    ) -> None:
        self._target = target
        self._graph = graph
        self._session = session
        # Legacy manifest_cache support for backward compatibility
        if manifest_cache:
            for name, manifest in manifest_cache.items():
                if name not in self._session._manifest_cache:
                    self._session._manifest_cache[name] = manifest

        self._readiness: TargetReadiness | None = None

    @property
    def name(self) -> str:
        """Target name."""
        return self._target.name

    @property
    def module(self) -> TargetModule:
        """Target module."""
        return self._target.module

    @property
    def readiness(self) -> TargetReadiness:
        """Compute and return full readiness state."""
        if self._readiness is None:
            self._readiness = self._compute_readiness()
        return self._readiness

    @property
    def is_ready(self) -> bool:
        """True if target is ready."""
        return self.readiness.is_ready

    @property
    def is_blocked(self) -> bool:
        """True if target is blocked."""
        return self.readiness.is_blocked

    @property
    def can_run(self) -> bool:
        """True if target can run now."""
        return self.readiness.can_run

    @property
    def self_status(self) -> SelfStatus:
        """Status considering only this target's own state."""
        return self.readiness.self_status

    @property
    def action_needed(self) -> ActionNeeded:
        """What action is needed to make this ready."""
        return self.readiness.action_needed

    @property
    def blocker_chain(self) -> tuple[BlockerInfo, ...]:
        """Full chain to ultimate bottleneck."""
        return self.readiness.blocker_chain

    @property
    def ultimate_bottleneck(self) -> str | None:
        """The deepest target that can actually run."""
        return self.readiness.ultimate_bottleneck

    @property
    def fix_command(self) -> str | None:
        """Suggested CLI command."""
        return self.readiness.fix_command

    def _compute_readiness(self) -> TargetReadiness:
        """Compute full readiness state from primary facts.

        Returns
        -------
        TargetReadiness
            Complete readiness state for this target.
        """
        self_status = self._compute_self_status()

        dep_status = self._compute_dependency_status()

        blocker_chain, ultimate_bottleneck = self._compute_blocker_chain(self_status, dep_status)

        action_needed = self._compute_action_needed(self_status, dep_status, ultimate_bottleneck)

        estimated_time = self._estimate_time_to_ready(blocker_chain)

        return TargetReadiness(
            name=self.name,
            self_status=self_status,
            dependency_status=dep_status,
            blocker_chain=blocker_chain,
            action_needed=action_needed,
            ultimate_bottleneck=ultimate_bottleneck,
            estimated_time_to_ready_ms=estimated_time,
        )

    def _compute_self_status(self) -> SelfStatus:
        """Compute status from this target's own state.

        Returns
        -------
        SelfStatus
            The target's self status (current, stale, never_computed, etc.).
        """
        manifest = self._get_manifest()

        if manifest is None:
            return "never_computed"

        current_hash = self._session.get_input_hash(self._target, manifest.options_hash)
        if manifest.input_hash != current_hash:
            return "stale"

        return "current"

    def _compute_dependency_status(self) -> DependencyStatus:
        """Compute status of this target's dependencies.

        Returns
        -------
        DependencyStatus
            Status indicating if dependencies are satisfied or blocked.
        """
        blockers: list[str] = []

        for dep_name in self._target.dependencies:
            dep_target = self._graph.get(dep_name)
            dep_view = TargetReadinessView(
                dep_target,
                self._graph,
                self._session,
            )

            if (
                dep_view.self_status != "current"
                or dep_view._compute_dependency_status().is_blocked
            ):
                blockers.append(dep_name)

        if not blockers:
            return DependencyStatus(kind="satisfied")

        return DependencyStatus(
            kind="blocked",
            blockers=tuple(blockers),
            first_blocker=blockers[0] if blockers else None,
        )

    def _compute_blocker_chain(
        self,
        self_status: SelfStatus,
        dep_status: DependencyStatus,
    ) -> tuple[tuple[BlockerInfo, ...], str | None]:
        """Build the blocker chain to the ultimate bottleneck.

        Returns
        -------
        tuple[tuple[BlockerInfo, ...], str | None]
            A tuple of (blocker_chain, ultimate_bottleneck).
        """
        if dep_status.is_satisfied and self_status == "current":
            return (), None

        chain: list[BlockerInfo] = []
        ultimate_bottleneck: str | None = None

        if self_status != "current":
            if dep_status.is_satisfied:
                chain.append(
                    BlockerInfo(
                        target=self.name,
                        blocked_by=None,
                        reason=self._status_to_reason(self_status),
                    )
                )
                ultimate_bottleneck = self.name
            else:
                chain.append(
                    BlockerInfo(
                        target=self.name,
                        blocked_by=dep_status.first_blocker,
                        reason="dependency not ready",
                    )
                )

                if dep_status.first_blocker:
                    dep_target = self._graph.get(dep_status.first_blocker)
                    dep_view = TargetReadinessView(
                        dep_target,
                        self._graph,
                        self._session,
                    )
                    dep_chain, dep_bottleneck = dep_view._compute_blocker_chain(
                        dep_view._compute_self_status(),
                        dep_view._compute_dependency_status(),
                    )
                    chain.extend(dep_chain)
                    ultimate_bottleneck = dep_bottleneck

        return tuple(chain), ultimate_bottleneck

    def _compute_action_needed(
        self,
        self_status: SelfStatus,
        dep_status: DependencyStatus,
        ultimate_bottleneck: str | None,
    ) -> ActionNeeded:
        """Determine what action would make this target ready.

        Returns
        -------
        ActionNeeded
            The action needed to make this target ready.
        """
        if self_status == "current" and dep_status.is_satisfied:
            return ActionNeeded(kind="none")

        if ultimate_bottleneck is None:
            return ActionNeeded(
                kind="blocked_external",
                reason="Unable to determine bottleneck",
            )

        if ultimate_bottleneck == self.name:
            return ActionNeeded(
                kind="run",
                target=self.name,
                reason=self._status_to_reason(self_status),
                command=f"codeintel build run {self.name}",
            )

        return ActionNeeded(
            kind="run_first",
            target=ultimate_bottleneck,
            reason=f"blocked by {ultimate_bottleneck}",
            command=f"codeintel build run {ultimate_bottleneck}",
        )

    def _estimate_time_to_ready(self, blocker_chain: tuple[BlockerInfo, ...]) -> int | None:
        """Estimate total time to make this target ready.

        Returns
        -------
        int | None
            Estimated milliseconds to ready, or None if unknown.
        """
        if not blocker_chain:
            return 0

        total_ms = 0
        include_self = len(blocker_chain) == 1
        for info in blocker_chain:
            if info.target == self.name and not include_self:
                continue
            target = self._graph.get(info.target)
            duration = target.estimated_duration_ms
            if duration is None:
                return None
            total_ms += duration

        return total_ms

    def _get_manifest(self) -> OutputManifest | None:
        """Get manifest from session cache or storage.

        Returns
        -------
        OutputManifest | None
            The manifest if it exists, or None.
        """
        return self._session.get_manifest(self.name)

    @staticmethod
    def _status_to_reason(status: SelfStatus) -> str:
        """Convert self status to human-readable reason.

        Returns
        -------
        str
            Human-readable reason string.
        """
        reasons: dict[SelfStatus, str] = {
            "current": "up to date",
            "stale": "input changed",
            "never_computed": "never computed",
            "data_missing": "data missing",
        }
        return reasons.get(status, str(status))


class DatabaseReadinessView:
    """System-wide readiness view for all targets.

    This class provides a convenient interface to query readiness across
    all targets in the build system.

    Uses BuildSession for efficient caching of manifests and hashes.

    Parameters
    ----------
    graph
        Complete target graph.
    gateway
        Storage gateway for manifest and data queries.
    snapshot
        Repository snapshot reference.
    """

    def __init__(
        self,
        graph: TargetGraph,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        self._graph = graph
        self._gateway = gateway
        self._snapshot = snapshot

        # Create session and preload manifests
        self._session = BuildSession(snapshot=snapshot, gateway=gateway)
        self._session.preload_manifests()

        self._views: dict[str, TargetReadinessView] = {}

    def __getitem__(self, name: str) -> TargetReadinessView:
        """Get readiness view for a specific target.

        Returns
        -------
        TargetReadinessView
            The readiness view for the named target.
        """
        if name not in self._views:
            target = self._graph.get(name)
            self._views[name] = TargetReadinessView(
                target,
                self._graph,
                self._session,
            )
        return self._views[name]

    def __contains__(self, name: str) -> bool:
        """Check if a target exists.

        Returns
        -------
        bool
            True if the target exists in the graph.
        """
        try:
            self._graph.get(name)
        except KeyError:
            return False
        else:
            return True

    def __iter__(self) -> Iterator[str]:
        """Iterate over all target names.

        Returns
        -------
        Iterator[str]
            Iterator of target names.
        """
        return iter(t.name for t in self._graph.all_targets)

    @property
    def repo(self) -> str:
        """Repository name."""
        return self._snapshot.repo

    @property
    def commit(self) -> str:
        """Commit SHA."""
        return self._snapshot.commit

    def all_readiness(self) -> dict[str, TargetReadiness]:
        """Get readiness for all targets.

        Returns
        -------
        dict[str, TargetReadiness]
            Mapping of target names to their readiness state.
        """
        return {name: self[name].readiness for name in self}

    def ready_targets(self) -> tuple[str, ...]:
        """Get names of all targets that are ready.

        Returns
        -------
        tuple[str, ...]
            Names of ready targets.
        """
        return tuple(name for name in self if self[name].is_ready)

    def runnable_targets(self) -> tuple[str, ...]:
        """Get names of targets that can run now (deps satisfied, needs compute).

        Returns
        -------
        tuple[str, ...]
            Names of runnable targets.
        """
        return tuple(name for name in self if self[name].can_run)

    def blocked_targets(self) -> tuple[tuple[str, TargetReadiness], ...]:
        """Get all blocked targets with their readiness info.

        Returns
        -------
        tuple[tuple[str, TargetReadiness], ...]
            Tuples of (name, readiness) for blocked targets.
        """
        return tuple((name, self[name].readiness) for name in self if self[name].is_blocked)

    def targets_for_module(self, module: TargetModule) -> tuple[str, ...]:
        """Get target names for a specific module.

        Returns
        -------
        tuple[str, ...]
            Names of targets in the specified module.
        """
        return tuple(t.name for t in self._graph.targets_for_module(module))

    def ready_for_module(self, module: TargetModule) -> tuple[str, ...]:
        """Get ready targets for a specific module.

        Returns
        -------
        tuple[str, ...]
            Names of ready targets in the specified module.
        """
        return tuple(name for name in self.targets_for_module(module) if self[name].is_ready)

    def runnable_for_module(self, module: TargetModule) -> tuple[str, ...]:
        """Get runnable targets for a specific module.

        Returns
        -------
        tuple[str, ...]
            Names of runnable targets in the specified module.
        """
        return tuple(name for name in self.targets_for_module(module) if self[name].can_run)

    def bottlenecks(self) -> tuple[str, ...]:
        """Get all ultimate bottlenecks across the system.

        These are targets that:
        1. Are not ready
        2. Have all their dependencies satisfied
        3. Can run now

        Running these would unblock other targets.

        Returns
        -------
        tuple[str, ...]
            Names of bottleneck targets.
        """
        seen: set[str] = set()
        for name in self:
            if self[name].is_blocked:
                bottleneck = self[name].ultimate_bottleneck
                if bottleneck:
                    seen.add(bottleneck)
        return tuple(sorted(seen))

    def summary(self) -> dict[str, int]:
        """Get summary counts of target statuses.

        Returns
        -------
        dict[str, int]
            Counts with keys: ready, runnable, blocked, total.
        """
        ready = 0
        runnable = 0
        blocked = 0

        for name in self:
            view = self[name]
            if view.is_ready:
                ready += 1
            elif view.can_run:
                runnable += 1
            else:
                blocked += 1

        return {
            "ready": ready,
            "runnable": runnable,
            "blocked": blocked,
            "total": ready + runnable + blocked,
        }

    def format_summary(self) -> str:
        """Format a human-readable summary of system readiness.

        Returns
        -------
        str
            Multi-line formatted summary text.
        """
        lines: list[str] = []
        lines.append(f"Readiness for {self.repo} @ {self.commit[:8]}")
        lines.append("=" * 50)
        lines.append("")

        summary = self.summary()
        lines.append(f"Ready: {summary['ready']} targets")
        lines.append(f"Runnable: {summary['runnable']} targets (can run now)")
        lines.append(f"Blocked: {summary['blocked']} targets")
        lines.append("")

        bottlenecks = self.bottlenecks()
        if bottlenecks:
            lines.append("Bottlenecks (run these to unblock):")
            for name in bottlenecks:
                view = self[name]
                lines.append(f"  • {name}: {view.action_needed.reason}")
            lines.append("")

        return "\n".join(lines)

__all__ = [
    "ActionKind",
    "ActionNeeded",
    "BlockerInfo",
    "DatabaseReadinessView",
    "DependencyStatus",
    "DependencyStatusKind",
    "SelfStatus",
    "TargetReadiness",
    "TargetReadinessView",
]
