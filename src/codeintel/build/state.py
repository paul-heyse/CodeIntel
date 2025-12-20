"""State validation for the build system.

This module provides the StateValidator class that determines the current
state of all build targets by examining stored manifests and comparing
input hashes.

Note: This module uses unified types from `codeintel.build.state_types`.
Import the unified types directly for new code.

Integration Points
------------------
- Uses `TargetGraph` from Phase 1 for dependency traversal
- Uses `BuildTracking` from Phase 7 for manifest storage
- Delegates to `StateComputer` for unified state computation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.config import BuildConfig, load_build_config
from codeintel.build.run_config import BuildRunConfig
from codeintel.build.run_context import BuildRunContext
from codeintel.build.session import BuildSession
from codeintel.build.state_computer import StateComputer
from codeintel.build.state_types import BuildState, TargetState
from codeintel.core.config.settings import BuildSettings

if TYPE_CHECKING:
    from codeintel.build.targets import TargetGraph
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StateValidationOptions:
    """Inputs required to compute target state."""

    settings: BuildSettings
    config: BuildConfig | None = None
    run_config: BuildRunConfig | None = None


class StateValidator:
    """Validate database state against the target graph.

    Examines stored manifests and computes current input hashes to determine
    which targets are missing, stale, current, or blocked. This is the
    foundation for computing minimal execution plans.

    The validation proceeds in two passes:

    1. **Pass 1**: Compute individual target states by comparing manifests
       against current input hashes.
    2. **Pass 2**: Propagate blocking status from dependencies to dependents.

    This class delegates to StateComputer for the actual computation.

    Parameters
    ----------
    graph
        Target graph defining all outputs and their dependencies.
    gateway
        Storage gateway for accessing manifests.
    snapshot
        Repository snapshot reference (repo, commit, repo_root).
    options
        State validation options (build settings + config overrides).

    Examples
    --------
    >>> options = StateValidationOptions(settings=build_settings)
    >>> validator = StateValidator(graph, gateway, snapshot, options=options)
    >>> state = validator.validate()
    >>> state.by_status("missing")
    ('ast', 'modules', ...)
    """

    def __init__(
        self,
        graph: TargetGraph,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        *,
        options: StateValidationOptions,
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
        options
            State validation options (build settings + config overrides).

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
        self._session = BuildSession(
            snapshot=snapshot,
            gateway=gateway,
            settings=options.settings,
        )
        base_config = options.config or load_build_config(snapshot.repo_root)
        effective_config = BuildRunContext.build_config_stack(base_config, options.run_config)
        self._computer = StateComputer(
            graph=graph,
            session=self._session,
            config=effective_config,
        )

    def validate(self) -> BuildState:
        """Validate state of all targets in the graph.

        Returns
        -------
        BuildState
            Complete state snapshot for all targets using unified types.
        """
        return self._computer.compute_all()

    def validate_target(self, name: str) -> TargetState:
        """Validate state of a single target.

        This is a convenience method that validates the entire graph and
        returns the state for the specified target. For repeated single-target
        queries, prefer calling `validate()` once and using `BuildState.get()`.

        Parameters
        ----------
        name
            Target name to validate.

        Returns
        -------
        TargetState
            Current state of the specified target using unified types.

        Raises
        ------
        KeyError
            If target name is not in the graph.
        """
        if name not in self._graph:
            msg = f"Target '{name}' not found in graph"
            raise KeyError(msg)
        return self._computer.compute_single(name)


__all__ = [
    "BuildState",
    "StateValidationOptions",
    "StateValidator",
    "TargetState",
]
