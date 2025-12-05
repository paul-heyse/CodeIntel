"""State validation for the build system.

This module provides state validation infrastructure that bridges the target
graph (Phase 1) with manifest storage (Phase 7). The StateValidator determines
the current state of all build targets by examining stored manifests and
comparing input hashes.

Key Concepts
------------
- **TargetState**: Current state of a single target (missing, computed, stale, blocked)
- **DatabaseState**: Aggregate state for all targets in a repo/commit snapshot
- **StateValidator**: Validates database state against the target graph

The validator answers: "What is the current state of all targets for this repo/commit?"
This enables the build system to identify what work needs to be done.

Integration Points
------------------
- Uses `TargetGraph` from Phase 1 for dependency traversal
- Uses `BuildTracking` from Phase 7 for manifest storage
- Uses `compute_input_hash` for cache invalidation detection
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from codeintel.build.hashing import compute_input_hash
from codeintel.build.manifest import OutputManifest

if TYPE_CHECKING:
    from codeintel.build.targets import OutputTarget, TargetGraph
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

# =============================================================================
# Type Definitions
# =============================================================================

TargetStatus = Literal["missing", "computed", "stale", "blocked"]
"""Status of a build target.

- ``missing``: No manifest exists for this target
- ``computed``: Manifest exists with matching input hash (up-to-date)
- ``stale``: Manifest exists but input hash differs (needs recomputation)
- ``blocked``: Dependencies are missing, stale, or blocked
"""

StalenessKind = Literal[
    "input_hash_mismatch",
    "dependency_stale",
    "dependency_missing",
    "dependency_blocked",
    "options_hash_mismatch",
]
"""Enumeration of reasons why a target may be stale or blocked."""


@dataclass(frozen=True)
class StalenessReason:
    """Structured explanation for why a target is stale or blocked.

    Provides machine-readable classification and human-readable details
    for debugging and reporting purposes.

    Attributes
    ----------
    kind
        Classification of the staleness cause.
    details
        Human-readable explanation with specific context.

    Examples
    --------
    >>> reason = StalenessReason(
    ...     kind="input_hash_mismatch",
    ...     details="Expected abc123, got def456",
    ... )
    >>> reason.kind
    'input_hash_mismatch'
    """

    kind: StalenessKind
    details: str


@dataclass(frozen=True)
class TargetState:
    """Current state of a single build target.

    Represents the validation result for one target, including its status,
    the stored manifest (if any), and details about why it may be stale
    or blocked.

    Attributes
    ----------
    name
        Target identifier matching the OutputTarget name.
    status
        Current status: missing, computed, stale, or blocked.
    manifest
        Stored manifest if one exists, None otherwise.
    staleness_reason
        Explanation if status is stale or blocked, None otherwise.
    blocking_deps
        Names of dependencies causing blocked status, empty tuple otherwise.
    current_input_hash
        Computed input hash for this snapshot (for debugging/comparison).

    Examples
    --------
    >>> state = TargetState(
    ...     name="ast",
    ...     status="computed",
    ...     manifest=manifest,
    ...     staleness_reason=None,
    ...     blocking_deps=(),
    ...     current_input_hash="abc123def456",
    ... )
    >>> state.status
    'computed'
    """

    name: str
    status: TargetStatus
    manifest: OutputManifest | None
    staleness_reason: StalenessReason | None
    blocking_deps: tuple[str, ...]
    current_input_hash: str | None


@dataclass(frozen=True)
class DatabaseState:
    """Aggregate state of all targets for a repo/commit snapshot.

    Provides query methods to filter targets by status, enabling the
    build resolver to determine what work needs to be done.

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
    >>> db_state = DatabaseState(repo="org/repo", commit="abc123", targets={})
    >>> db_state.missing_targets()
    ()
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
            If target name is not found in the state.
        """
        if name not in self.targets:
            msg = f"Target '{name}' not found in database state"
            raise KeyError(msg)
        return self.targets[name]

    def missing_targets(self) -> tuple[str, ...]:
        """Return names of targets with no manifest.

        Returns
        -------
        tuple[str, ...]
            Target names with status "missing", sorted alphabetically.
        """
        return tuple(
            sorted(name for name, state in self.targets.items() if state.status == "missing")
        )

    def stale_targets(self) -> tuple[str, ...]:
        """Return names of targets that need recomputation.

        Returns
        -------
        tuple[str, ...]
            Target names with status "stale", sorted alphabetically.
        """
        return tuple(
            sorted(name for name, state in self.targets.items() if state.status == "stale")
        )

    def computed_targets(self) -> tuple[str, ...]:
        """Return names of targets that are up-to-date.

        Returns
        -------
        tuple[str, ...]
            Target names with status "computed", sorted alphabetically.
        """
        return tuple(
            sorted(name for name, state in self.targets.items() if state.status == "computed")
        )

    def blocked_targets(self) -> tuple[str, ...]:
        """Return names of targets blocked by dependencies.

        Returns
        -------
        tuple[str, ...]
            Target names with status "blocked", sorted alphabetically.
        """
        return tuple(
            sorted(name for name, state in self.targets.items() if state.status == "blocked")
        )

    def is_target_current(self, name: str) -> bool:
        """Check if a target is up-to-date.

        Parameters
        ----------
        name
            Target name to check.

        Returns
        -------
        bool
            True if target exists and has status "computed".
        """
        if name not in self.targets:
            return False
        return self.targets[name].status == "computed"


# =============================================================================
# State Validator
# =============================================================================


class StateValidator:
    """Validate database state against the target graph.

    Examines stored manifests and computes current input hashes to determine
    which targets are missing, stale, computed, or blocked. This is the
    foundation for computing minimal execution plans.

    The validation proceeds in two passes:

    1. **Pass 1**: Compute individual target states by comparing manifests
       against current input hashes.
    2. **Pass 2**: Propagate blocking status from dependencies to dependents.

    Parameters
    ----------
    graph
        Target graph defining all outputs and their dependencies.
    gateway
        Storage gateway for accessing manifests.
    snapshot
        Repository snapshot reference (repo, commit, repo_root).

    Examples
    --------
    >>> validator = StateValidator(graph, gateway, snapshot)
    >>> state = validator.validate()
    >>> state.missing_targets()
    ('ast', 'modules', ...)
    """

    def __init__(
        self,
        graph: TargetGraph,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Initialize the state validator.

        Parameters
        ----------
        graph
            Target graph with all registered targets.
        gateway
            Storage gateway for manifest access.
        snapshot
            Repository snapshot reference.

        Raises
        ------
        ValueError
            If the target graph has validation errors.
        """
        self._graph = graph
        self._gateway = gateway
        self._snapshot = snapshot

        # Validate graph integrity before proceeding
        errors = graph.validate()
        if errors:
            error_msg = "\n".join(errors)
            msg = f"Target graph validation failed:\n{error_msg}"
            raise ValueError(msg)

    def validate(self) -> DatabaseState:
        """Validate state of all targets in the graph.

        Performs two-pass validation:
        1. Compute individual states from manifests and hashes
        2. Propagate blocking status from dependencies

        Returns
        -------
        DatabaseState
            Complete state snapshot for all targets.
        """
        # Load all manifests for this repo/commit
        manifests = self._gateway.build.list_manifests(
            repo=self._snapshot.repo,
            commit=self._snapshot.commit,
        )
        manifest_lookup: dict[str, OutputManifest] = {m.target: m for m in manifests}

        # Log manifests for unknown targets (graceful handling of removed targets)
        known_targets = set(self._graph)
        for target_name in manifest_lookup:
            if target_name not in known_targets:
                log.warning(
                    "Found manifest for unknown target '%s' (may have been removed)",
                    target_name,
                )

        # Pass 1: Compute individual target states
        preliminary_states: dict[str, TargetState] = {}
        for target_name in self._graph:
            target = self._graph.get(target_name)
            manifest = manifest_lookup.get(target_name)
            preliminary_states[target_name] = self._compute_individual_state(target, manifest)

        # Pass 2: Propagate blocking status
        final_states = self._propagate_blocking(preliminary_states)

        return DatabaseState(
            repo=self._snapshot.repo,
            commit=self._snapshot.commit,
            targets=final_states,
        )

    def validate_target(self, name: str) -> TargetState:
        """Validate state of a single target.

        This is a convenience method that validates the entire graph and
        returns the state for the specified target. For repeated single-target
        queries, prefer calling `validate()` once and using `DatabaseState.get()`.

        Parameters
        ----------
        name
            Target name to validate.

        Returns
        -------
        TargetState
            Current state of the specified target.

        Raises
        ------
        KeyError
            If target name is not in the graph.
        """
        if name not in self._graph:
            msg = f"Target '{name}' not found in graph"
            raise KeyError(msg)
        return self.validate().get(name)

    def _compute_individual_state(
        self,
        target: OutputTarget,
        manifest: OutputManifest | None,
    ) -> TargetState:
        """Compute state for a single target without considering dependencies.

        Parameters
        ----------
        target
            Target to compute state for.
        manifest
            Stored manifest if one exists, None otherwise.

        Returns
        -------
        TargetState
            Preliminary state (may be upgraded to blocked in Pass 2).
        """
        # No manifest means target is missing
        if manifest is None:
            return TargetState(
                name=target.name,
                status="missing",
                manifest=None,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash=None,
            )

        # Compare input hashes
        is_current, current_hash = self._check_input_hash(target, manifest)

        if is_current:
            return TargetState(
                name=target.name,
                status="computed",
                manifest=manifest,
                staleness_reason=None,
                blocking_deps=(),
                current_input_hash=current_hash,
            )

        # Determine staleness reason
        reason = self._determine_staleness_reason(manifest, current_hash)
        return TargetState(
            name=target.name,
            status="stale",
            manifest=manifest,
            staleness_reason=reason,
            blocking_deps=(),
            current_input_hash=current_hash,
        )

    def _check_input_hash(
        self,
        target: OutputTarget,
        manifest: OutputManifest,
    ) -> tuple[bool, str]:
        """Compare stored input_hash against current computation.

        Parameters
        ----------
        target
            Target to check.
        manifest
            Stored manifest with input_hash to compare.

        Returns
        -------
        tuple[bool, str]
            (is_current, current_hash) where is_current is True if hashes match.
        """
        current_hash = compute_input_hash(
            target=target,
            snapshot=self._snapshot,
            gateway=self._gateway,
            options_hash=manifest.options_hash,
        )
        return (manifest.input_hash == current_hash, current_hash)

    @staticmethod
    def _determine_staleness_reason(
        manifest: OutputManifest,
        current_hash: str,
    ) -> StalenessReason:
        """Determine why a target is stale based on hash comparison.

        Parameters
        ----------
        manifest
            Stored manifest with old hash.
        current_hash
            Newly computed hash.

        Returns
        -------
        StalenessReason
            Explanation of why hashes differ.
        """
        # The hash mismatch could be due to input changes or options changes
        # Since compute_input_hash includes options_hash, we report as input mismatch
        return StalenessReason(
            kind="input_hash_mismatch",
            details=f"Stored hash '{manifest.input_hash}' != current hash '{current_hash}'",
        )

    def _propagate_blocking(
        self,
        preliminary_states: dict[str, TargetState],
    ) -> dict[str, TargetState]:
        """Propagate blocking status from dependencies to dependents.

        A target is blocked if any of its dependencies is missing, stale,
        or blocked. This propagates transitively through the dependency graph.

        Parameters
        ----------
        preliminary_states
            States computed in Pass 1.

        Returns
        -------
        dict[str, TargetState]
            Final states with blocking propagated.
        """
        final_states = dict(preliminary_states)

        # Process in topological order so dependencies are finalized first
        topo_order = self._graph.topological_order(list(self._graph))

        for target_name in topo_order:
            current_state = final_states[target_name]

            # Skip targets that are already missing (no manifest to protect)
            if current_state.status == "missing":
                continue

            # Check all dependencies for blocking conditions
            target = self._graph.get(target_name)
            blocking_deps, blocking_reason = self._find_blocking_deps(
                target.dependencies, final_states
            )

            # If we have blocking deps, upgrade status to blocked
            if blocking_deps:
                final_states[target_name] = TargetState(
                    name=target_name,
                    status="blocked",
                    manifest=current_state.manifest,
                    staleness_reason=blocking_reason,
                    blocking_deps=tuple(sorted(blocking_deps)),
                    current_input_hash=current_state.current_input_hash,
                )

        return final_states

    def _find_blocking_deps(
        self,
        dependencies: tuple[str, ...],
        states: dict[str, TargetState],
    ) -> tuple[list[str], StalenessReason | None]:
        """Find dependencies that block a target.

        Parameters
        ----------
        dependencies
            Names of dependencies to check.
        states
            Current state of all targets.

        Returns
        -------
        tuple[list[str], StalenessReason | None]
            List of blocking dependency names and reason for first blocker.
        """
        blocking_deps: list[str] = []
        blocking_reason: StalenessReason | None = None

        for dep_name in dependencies:
            dep_state = states[dep_name]
            reason = self._check_dependency_blocking(dep_name, dep_state)
            if reason is not None:
                blocking_deps.append(dep_name)
                if blocking_reason is None:
                    blocking_reason = reason

        return blocking_deps, blocking_reason

    @staticmethod
    def _check_dependency_blocking(
        dep_name: str,
        dep_state: TargetState,
    ) -> StalenessReason | None:
        """Check if a dependency causes blocking.

        Parameters
        ----------
        dep_name
            Name of the dependency.
        dep_state
            Current state of the dependency.

        Returns
        -------
        StalenessReason | None
            Reason if dependency causes blocking, None otherwise.
        """
        if dep_state.status == "missing":
            return StalenessReason(
                kind="dependency_missing",
                details=f"Dependency '{dep_name}' has not been computed",
            )
        if dep_state.status == "stale":
            return StalenessReason(
                kind="dependency_stale",
                details=f"Dependency '{dep_name}' is stale and needs recomputation",
            )
        if dep_state.status == "blocked":
            return StalenessReason(
                kind="dependency_blocked",
                details=f"Dependency '{dep_name}' is blocked by its own dependencies",
            )
        return None


__all__ = [
    "DatabaseState",
    "StalenessKind",
    "StalenessReason",
    "StateValidator",
    "TargetState",
    "TargetStatus",
]
