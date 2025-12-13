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
- Delegates to `StateComputer` for unified state computation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from codeintel.build.session import BuildSession
from codeintel.build.state_computer import StateComputer
from codeintel.build.state_types import BuildState as UnifiedBuildState
from codeintel.build.state_types import TargetState as UnifiedTargetState

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.manifest import OutputManifest
    from codeintel.build.targets import TargetGraph
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


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


def _unified_to_legacy_status(status: str) -> TargetStatus:
    """Convert unified status to legacy status.

    Parameters
    ----------
    status
        Unified status string.

    Returns
    -------
    TargetStatus
        Legacy status.
    """
    status_map: dict[str, TargetStatus] = {
        "current": "computed",
        "stale": "stale",
        "missing": "missing",
        "blocked": "blocked",
    }
    return status_map.get(status, "missing")


def _unified_to_legacy_reason(
    unified: UnifiedTargetState,
) -> StalenessReason | None:
    """Convert unified blocking reason to legacy staleness reason.

    Parameters
    ----------
    unified
        Unified target state.

    Returns
    -------
    StalenessReason | None
        Legacy staleness reason.
    """
    if unified.blocking_reason is None:
        return None

    reason_map: dict[str, StalenessKind] = {
        "input_hash_mismatch": "input_hash_mismatch",
        "dependency_missing": "dependency_missing",
        "dependency_stale": "dependency_stale",
        "dependency_blocked": "dependency_blocked",
        "options_hash_mismatch": "options_hash_mismatch",
        "data_missing": "input_hash_mismatch",  # Map to closest equivalent
    }
    kind = reason_map.get(unified.blocking_reason, "input_hash_mismatch")

    # Build details message
    if unified.blocking_deps:
        details = f"Blocked by: {', '.join(unified.blocking_deps)}"
    elif unified.current_hash and unified.stored_hash:
        details = f"Stored hash '{unified.stored_hash}' != current hash '{unified.current_hash}'"
    else:
        details = f"Reason: {unified.blocking_reason}"

    return StalenessReason(kind=kind, details=details)


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

    @classmethod
    def from_unified(cls, unified: UnifiedTargetState) -> TargetState:
        """Create legacy TargetState from unified TargetState.

        Parameters
        ----------
        unified
            Unified target state from StateComputer.

        Returns
        -------
        TargetState
            Legacy target state.
        """
        return cls(
            name=unified.name,
            status=_unified_to_legacy_status(unified.status),
            manifest=unified.manifest,
            staleness_reason=_unified_to_legacy_reason(unified),
            blocking_deps=unified.blocking_deps,
            current_input_hash=unified.current_hash,
        )


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

    @classmethod
    def from_unified(cls, unified: UnifiedBuildState) -> DatabaseState:
        """Create legacy DatabaseState from unified BuildState.

        Parameters
        ----------
        unified
            Unified build state from StateComputer.

        Returns
        -------
        DatabaseState
            Legacy database state.
        """
        legacy_targets = {
            name: TargetState.from_unified(state) for name, state in unified.targets.items()
        }
        return cls(
            repo=unified.repo,
            commit=unified.commit,
            targets=legacy_targets,
        )

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


class StateValidator:
    """Validate database state against the target graph.

    Examines stored manifests and computes current input hashes to determine
    which targets are missing, stale, computed, or blocked. This is the
    foundation for computing minimal execution plans.

    The validation proceeds in two passes:

    1. **Pass 1**: Compute individual target states by comparing manifests
       against current input hashes.
    2. **Pass 2**: Propagate blocking status from dependencies to dependents.

    This class delegates to StateComputer for the actual computation,
    preserving the legacy API for backward compatibility.

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

        # Validate graph
        errors = graph.validate()
        if errors:
            error_msg = "\n".join(errors)
            msg = f"Target graph validation failed:\n{error_msg}"
            raise ValueError(msg)

        # Create session and computer for delegation
        self._session = BuildSession(snapshot=snapshot, gateway=gateway)
        self._computer = StateComputer(graph=graph, session=self._session)

    def validate(self) -> DatabaseState:
        """Validate state of all targets in the graph.

        Delegates to StateComputer and wraps result in legacy types.

        Returns
        -------
        DatabaseState
            Complete state snapshot for all targets.
        """
        unified_state = self._computer.compute_all()
        return DatabaseState.from_unified(unified_state)

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
        unified_state = self._computer.compute_single(name)
        return TargetState.from_unified(unified_state)


__all__ = [
    "DatabaseState",
    "StalenessKind",
    "StalenessReason",
    "StateValidator",
    "TargetState",
    "TargetStatus",
]
