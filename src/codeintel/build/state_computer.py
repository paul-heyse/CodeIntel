"""Unified state computation for build targets.

This module provides the StateComputer class which serves as the single
source of truth for computing target state. It replaces the duplicated
computation logic previously spread across state.py and readiness.py.

The StateComputer:
1. Computes individual target states from cache presence
2. Propagates blocking status through the dependency graph
3. Uses session-scoped caching for efficiency

Both StateValidator and DatabaseReadinessView delegate to this computer
rather than implementing their own state computation.

Example
-------
>>> session = BuildSession(snapshot, cache_index, cache_key_resolver, {})
>>> computer = StateComputer(catalog, session)
>>> build_state = computer.compute_all()
>>> build_state.runnable_targets()
('ast', 'modules')
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.build.session import BuildSession
from codeintel.build.state_types import (
    BuildState,
    TargetState,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.hamilton.cache_index import CacheIndex
    from codeintel.build.hamilton.cache_key_resolver import CacheKeyResolver
    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.build.state_types import BlockingReason, TargetStatus
    from codeintel.config.primitives import SnapshotRef

__all__ = [
    "StateComputer",
]


class StateComputer:
    """Single source of truth for target state computation.

    Computes the BuildState for all targets in a snapshot by:
    1. Computing cache keys for nodes
    2. Computing individual states from cache presence
    3. Propagating blocking status through the dependency graph

    The session provides caching to avoid redundant cache key resolution.

    Parameters
    ----------
    catalog
        DAG catalog defining all outputs and their dependencies.
    session
        Build session providing cache index and cache key resolution.

    Examples
    --------
    >>> session = BuildSession(snapshot, cache_index, cache_key_resolver, {})
    >>> computer = StateComputer(catalog, session)
    >>> state = computer.compute_all()
    >>> state.by_status("current")
    ('modules', 'ast')
    """

    def __init__(
        self,
        catalog: DagCatalog,
        session: BuildSession,
    ) -> None:
        """Initialize the state computer.

        Parameters
        ----------
        catalog
            DAG catalog with all registered targets.
        session
            Build session for caching and storage access.
        """
        self._catalog = catalog
        self._session = session

    @classmethod
    def create(
        cls,
        catalog: DagCatalog,
        snapshot: SnapshotRef,
        cache_index: CacheIndex | None,
        cache_key_resolver: CacheKeyResolver | None,
        input_values: Mapping[str, object],
    ) -> StateComputer:
        """Create a StateComputer with a new session.

        Convenience factory that creates the BuildSession automatically.

        Parameters
        ----------
        catalog
            DAG catalog with all registered targets.
        snapshot
            Repository snapshot reference.
        cache_index
            Cache index used for cache probes.
        cache_key_resolver
            Cache key resolver for computing cache keys.
        input_values
            External input values used for cache hashing.

        Returns
        -------
        StateComputer
            New state computer instance.
        """
        session = BuildSession(
            snapshot=snapshot,
            cache_index=cache_index,
            cache_key_resolver=cache_key_resolver,
            input_values=input_values,
        )
        return cls(catalog=catalog, session=session)

    def compute_all(self) -> BuildState:
        """Compute state for all targets in topological order.

        Performs two-pass computation:
        1. Individual states from cache presence
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
            If target name is not in the catalog.
        """
        if name not in self._catalog:
            msg = f"Target '{name}' not found in catalog"
            raise KeyError(msg)

        # Get the target and its cache state
        target = self._catalog.targets[name]
        node_name = self._catalog.target_nodes.get(name)
        if node_name is None:
            msg = f"Target '{name}' missing anchor node in catalog"
            raise KeyError(msg)

        # Compute preliminary state
        state = self._state_from_cache(node_name)

        # Check for blocking dependencies
        blocking_deps, blocking_reason = self._find_blocking_deps(target.dependencies)
        if blocking_deps:
            state = TargetState(
                name=name,
                status="blocked",
                current_hash=state.current_hash,
                blocking_reason=blocking_reason,
                blocking_deps=tuple(sorted(blocking_deps)),
                stored_hash=state.stored_hash,
            )

        return state

    def _compute_preliminary_states(self) -> dict[str, TargetState]:
        """Compute individual states ignoring dependency blocking.

        Computes cache keys and then computes state for each target
        based on cache presence.

        Returns
        -------
        dict[str, TargetState]
            Mapping of target names to preliminary states.
        """
        self._session.preload_cache_keys()

        states: dict[str, TargetState] = {}
        for target_name in self._catalog:
            node_name = self._catalog.target_nodes.get(target_name)
            if node_name is None:
                msg = f"Target '{target_name}' missing anchor node in catalog"
                raise KeyError(msg)
            states[target_name] = self._state_from_cache(node_name)

        return states

    @staticmethod
    def _state_from_cache(
        self,
        node_name: str,
    ) -> TargetState:
        """Determine state for a target based on cache presence.

        Parameters
        ----------
        node_name
            Anchor node name for the target.

        Returns
        -------
        TargetState
            Preliminary state (may be upgraded to blocked in pass 2).
        """
        cache_key = self._session.cache_key_for_node(node_name)
        is_cached = self._session.cache_hit(node_name)
        if not is_cached:
            return TargetState(
                name=self._catalog.node_to_target[node_name],
                status="missing",
                current_hash=cache_key,
                blocking_reason=None,
                blocking_deps=(),
            )
        return TargetState(
            name=self._catalog.node_to_target[node_name],
            status="current",
            current_hash=cache_key,
            blocking_reason=None,
            blocking_deps=(),
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
        topo_order = self._catalog.closure(tuple(self._catalog.targets))

        for target_name in topo_order:
            current_state = final[target_name]

            target = self._catalog.targets[target_name]
            blocking_deps, blocking_reason = self._find_blocking_deps_from_states(
                target.dependencies, final
            )

            if blocking_deps:
                final[target_name] = TargetState(
                    name=target_name,
                    status="blocked",
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
            node_name = self._catalog.target_nodes.get(dep_name)
            if node_name is None:
                msg = f"Target '{dep_name}' missing anchor node in catalog"
                raise KeyError(msg)
            dep_state = self._state_from_cache(node_name)
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
            "blocked": "dependency_blocked",
        }
        return status_to_reason[dep_state.status]
