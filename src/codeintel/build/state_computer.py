"""Unified state computation for build targets.

This module provides the StateComputer class which serves as the single
source of truth for computing target state. It replaces the duplicated
computation logic previously spread across state.py and readiness.py.

The StateComputer:
1. Computes individual target states from manifests and hashes
2. Propagates blocking status through the dependency graph
3. Uses session-scoped caching for efficiency

Both StateValidator and DatabaseReadinessView delegate to this computer
rather than implementing their own state computation.

Example
-------
>>> session = BuildSession(snapshot, gateway, settings)
>>> computer = StateComputer(graph, session)
>>> build_state = computer.compute_all()
>>> build_state.runnable_targets()
('ast', 'modules')
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from codeintel.build.config import BuildConfig
from codeintel.build.hash_evaluator import evaluate_hash_state
from codeintel.build.hashing import compute_target_options_hash
from codeintel.build.session import BuildSession
from codeintel.build.state_types import (
    BuildState,
    TargetState,
)
from codeintel.core.config.settings import BuildSettings

if TYPE_CHECKING:
    from codeintel.build.state_types import (
        BlockingReason,
        TargetStatus,
    )
    from codeintel.build.targets import OutputTarget, TargetGraph
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.build_manifest import OutputManifest
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

__all__ = [
    "StateComputer",
]


class StateComputer:
    """Single source of truth for target state computation.

    Computes the BuildState for all targets in a snapshot by:
    1. Bulk-loading manifests (single DB query)
    2. Computing individual states from manifests and hashes
    3. Propagating blocking status through the dependency graph

    The session provides caching to avoid redundant hash computations
    when the same target's state is queried multiple times.

    Parameters
    ----------
    graph
        Target graph defining all outputs and their dependencies.
    session
        Build session providing caching and gateway access.

    Examples
    --------
    >>> session = BuildSession(snapshot, gateway, settings)
    >>> computer = StateComputer(graph, session)
    >>> state = computer.compute_all()
    >>> state.by_status("current")
    ('modules', 'ast')
    """

    def __init__(
        self,
        graph: TargetGraph,
        session: BuildSession,
        *,
        config: BuildConfig | None = None,
    ) -> None:
        """Initialize the state computer.

        Parameters
        ----------
        graph
            Target graph with all registered targets.
        session
            Build session for caching and storage access.
        config
            Build configuration used to compute per-target options hashes.
        """
        self._graph = graph
        self._session = session
        self._config = config or BuildConfig.empty()

    @classmethod
    def create(
        cls,
        graph: TargetGraph,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        settings: BuildSettings,
    ) -> StateComputer:
        """Create a StateComputer with a new session.

        Convenience factory that creates the BuildSession automatically.

        Parameters
        ----------
        graph
            Target graph with all registered targets.
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        settings
            Build settings for input hash computation.

        Returns
        -------
        StateComputer
            New state computer instance.
        """
        session = BuildSession(snapshot=snapshot, gateway=gateway, settings=settings)
        return cls(graph=graph, session=session)

    def _options_hash_for_target(self, target: OutputTarget) -> str | None:
        params = self._config.parameters_for(target.name)
        return compute_target_options_hash(params)

    def compute_all(self) -> BuildState:
        """Compute state for all targets in topological order.

        Performs two-pass computation:
        1. Individual states from manifests and hashes
        2. Blocking propagation from dependencies

        Returns
        -------
        BuildState
            Complete state snapshot for all targets.
        """
        # Pass 1: Compute individual states (ignoring dependency status)
        preliminary = self._compute_preliminary_states()

        # Pass 2: Propagate blocking from dependencies to dependents
        final = self._propagate_blocking(preliminary)

        return BuildState(
            repo=self._session.snapshot.repo,
            commit=self._session.snapshot.commit,
            targets=final,
        )

    def compute_single(self, name: str) -> TargetState:
        """Compute state for a single target.

        Computes the full state including dependency blocking check.
        For bulk queries, prefer compute_all() to avoid redundant work.

        Parameters
        ----------
        name
            Target name to compute state for.

        Returns
        -------
        TargetState
            Current state of the target.

        Raises
        ------
        KeyError
            If target name is not in the graph.
        """
        if name not in self._graph:
            msg = f"Target '{name}' not found in graph"
            raise KeyError(msg)

        # Get the target and its manifest
        target = self._graph.get(name)
        manifest = self._session.get_manifest(name)

        # Compute preliminary state
        state = self._state_from_manifest(target, manifest)

        # Check for blocking dependencies
        if state.status != "missing":
            blocking_deps, blocking_reason = self._find_blocking_deps(target.dependencies)
            if blocking_deps:
                state = TargetState(
                    name=name,
                    status="blocked",
                    manifest=state.manifest,
                    current_hash=state.current_hash,
                    blocking_reason=blocking_reason,
                    blocking_deps=tuple(sorted(blocking_deps)),
                    stored_hash=state.stored_hash,
                )

        return state

    def _compute_preliminary_states(self) -> dict[str, TargetState]:
        """Compute individual states ignoring dependency blocking.

        Bulk-loads all manifests for efficiency, then computes state
        for each target based on manifest existence and hash comparison.

        Returns
        -------
        dict[str, TargetState]
            Mapping of target names to preliminary states.
        """
        # Bulk-load manifests in a single query
        self._session.preload_manifests()

        unknown_targets = sorted(
            name for name in self._session.cached_manifest_targets() if name not in self._graph
        )
        for name in unknown_targets:
            log.warning("Ignoring manifest for unknown target: %s", name)

        states: dict[str, TargetState] = {}
        for target_name in self._graph:
            target = self._graph.get(target_name)
            manifest = self._session.get_manifest(target_name)
            states[target_name] = self._state_from_manifest(target, manifest)

        return states

    def _state_from_manifest(
        self,
        target: OutputTarget,
        manifest: OutputManifest | None,
    ) -> TargetState:
        """Determine state for a target based on its manifest.

        Parameters
        ----------
        target
            Target to compute state for.
        manifest
            Stored manifest if one exists, None otherwise.

        Returns
        -------
        TargetState
            Preliminary state (may be upgraded to blocked in pass 2).
        """
        if manifest is None:
            return TargetState(
                name=target.name,
                status="missing",
                manifest=None,
                current_hash=None,
                blocking_reason=None,
                blocking_deps=(),
            )
        options_hash = self._options_hash_for_target(target)
        current_hash = self._session.get_input_hash(target, options_hash)
        evaluation = evaluate_hash_state(
            manifest=manifest,
            input_hash=current_hash,
            options_hash=options_hash,
        )

        if evaluation.status == "missing":
            return TargetState(
                name=target.name,
                status="missing",
                manifest=None,
                current_hash=current_hash,
                blocking_reason=None,
                blocking_deps=(),
            )

        if evaluation.status == "current":
            return TargetState(
                name=target.name,
                status="current",
                manifest=manifest,
                current_hash=current_hash,
                blocking_reason=None,
                blocking_deps=(),
                stored_hash=evaluation.stored_hash,
            )

        return TargetState(
            name=target.name,
            status="stale",
            manifest=manifest,
            current_hash=current_hash,
            blocking_reason=cast("BlockingReason", evaluation.reason),
            blocking_deps=(),
            stored_hash=evaluation.stored_hash,
        )

    def _propagate_blocking(
        self,
        preliminary: dict[str, TargetState],
    ) -> dict[str, TargetState]:
        """Propagate blocking status through dependency graph.

        A target is blocked if any of its dependencies is not current.
        Processes in topological order so dependencies are resolved first.

        Parameters
        ----------
        preliminary
            States computed in pass 1.

        Returns
        -------
        dict[str, TargetState]
            Final states with blocking propagated.
        """
        final = dict(preliminary)

        # Get topological order to process deps before dependents
        topo_order = self._graph.topological_order(list(self._graph))

        for target_name in topo_order:
            current_state = final[target_name]

            # Missing targets stay missing (no deps to check)
            if current_state.status == "missing":
                continue

            target = self._graph.get(target_name)
            blocking_deps, blocking_reason = self._find_blocking_deps_from_states(
                target.dependencies, final
            )

            if blocking_deps:
                final[target_name] = TargetState(
                    name=target_name,
                    status="blocked",
                    manifest=current_state.manifest,
                    current_hash=current_state.current_hash,
                    blocking_reason=blocking_reason,
                    blocking_deps=tuple(sorted(blocking_deps)),
                    stored_hash=current_state.stored_hash,
                )

        return final

    def _find_blocking_deps(
        self,
        dependencies: tuple[str, ...],
    ) -> tuple[list[str], BlockingReason | None]:
        """Find blocking dependencies for a single target lookup.

        Used by compute_single() for on-demand computation.

        Parameters
        ----------
        dependencies
            Names of dependencies to check.

        Returns
        -------
        tuple[list[str], BlockingReason | None]
            List of blocking dep names and reason for first blocker.
        """
        blocking_deps: list[str] = []
        first_reason: BlockingReason | None = None

        for dep_name in dependencies:
            dep_target = self._graph.get(dep_name)
            dep_manifest = self._session.get_manifest(dep_name)
            dep_state = self._state_from_manifest(dep_target, dep_manifest)
            reason = self._check_dep_blocking(dep_state)
            if reason is not None:
                blocking_deps.append(dep_name)
                if first_reason is None:
                    first_reason = reason

        return blocking_deps, cast("BlockingReason | None", first_reason)

    def _find_blocking_deps_from_states(
        self,
        dependencies: tuple[str, ...],
        states: dict[str, TargetState],
    ) -> tuple[list[str], BlockingReason | None]:
        """Find blocking dependencies from pre-computed states.

        Used by _propagate_blocking() during full computation.

        Parameters
        ----------
        dependencies
            Names of dependencies to check.
        states
            Current state of all targets.

        Returns
        -------
        tuple[list[str], BlockingReason | None]
            List of blocking dep names and reason for first blocker.
        """
        blocking_deps: list[str] = []
        first_reason: BlockingReason | None = None

        for dep_name in dependencies:
            dep_state = states[dep_name]
            reason = self._check_dep_blocking(dep_state)
            if reason is not None:
                blocking_deps.append(dep_name)
                if first_reason is None:
                    first_reason = reason

        return blocking_deps, cast("BlockingReason | None", first_reason)

    @staticmethod
    def _check_dep_blocking(dep_state: TargetState) -> BlockingReason | None:
        """Check if a dependency causes blocking.

        Parameters
        ----------
        dep_state
            Current state of the dependency.

        Returns
        -------
        BlockingReason | None
            Reason if dependency causes blocking, None otherwise.
        """
        status_to_reason: dict[TargetStatus, BlockingReason | None] = {
            "current": None,
            "missing": "dependency_missing",
            "stale": "dependency_stale",
            "blocked": "dependency_blocked",
        }
        return status_to_reason[dep_state.status]
