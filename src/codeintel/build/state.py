"""State validation for the build system.

This module provides the StateValidator class that determines the current
state of all build targets by examining cache presence.

Note: This module uses unified types from `codeintel.build.state_types`.
Import the unified types directly for new code.

Integration Points
------------------
- Uses `DagCatalog` from the Hamilton compiler for dependency traversal
- Uses the Hamilton cache index for cache presence
- Delegates to `StateComputer` for unified state computation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.session import BuildSession
from codeintel.build.state_computer import StateComputer
from codeintel.build.state_types import BuildState, TargetState
from codeintel.runtime.inputs import ExecutionInputs, execution_input_mapping

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle


@dataclass(frozen=True, slots=True)
class StateValidationOptions:
    """Inputs required to compute target state."""


class StateValidator:
    """Validate cache state against the DAG catalog.

    Examines cache entries to determine which targets are missing, current,
    or blocked. This is the foundation for inspecting readiness.

    The validation proceeds in two passes:

    1. **Pass 1**: Compute individual target states from cache presence.
    2. **Pass 2**: Propagate blocking status from dependencies to dependents.

    This class delegates to StateComputer for the actual computation.

    Parameters
    ----------
    catalog
        DAG catalog defining all outputs and their dependencies.
    session
        Build session for cache probing and snapshot identity.
    options
        State validation options.

    Examples
    --------
    >>> options = StateValidationOptions()
    >>> validator = StateValidator(catalog, session, options=options)
    >>> state = validator.validate()
    >>> state.by_status("missing")
    ('ast', 'modules', ...)
    """

    def __init__(
        self,
        catalog: DagCatalog,
        session: BuildSession,
        *,
        options: StateValidationOptions,
    ) -> None:
        """Initialize the state validator.

        Parameters
        ----------
        catalog
            DAG catalog with all registered targets.
        session
            Build session for cache access and snapshot metadata.
        options
            State validation options.

        Raises
        ------
        ValueError
            If the DAG catalog has validation errors.
        """
        self._catalog = catalog
        self._session = session
        self._options = options

        # Validate catalog
        errors = catalog.validate()
        if errors:
            error_msg = "\n".join(errors)
            msg = f"DAG catalog validation failed:\n{error_msg}"
            raise ValueError(msg)

        self._computer = StateComputer(
            catalog=catalog,
            session=session,
        )

    @classmethod
    def from_runtime(
        cls,
        *,
        runtime: HamiltonRuntimeBundle,
        env: BuildEnv,
        options: StateValidationOptions | None = None,
    ) -> StateValidator:
        """Create a StateValidator from a runtime bundle and environment.

        Parameters
        ----------
        runtime
            Runtime bundle providing cache index and catalog.
        env
            Build environment providing snapshot identity.
        options
            Optional state validation options.

        Returns
        -------
        StateValidator
            Validator wired to runtime cache inputs.
        """
        session = BuildSession(
            snapshot=env.snapshot,
            cache_index=runtime.cache_index,
            cache_key_resolver=runtime.cache_key_resolver,
            input_values=_state_input_values(runtime=runtime, env=env),
        )
        return cls(
            runtime.catalog,
            session,
            options=options or StateValidationOptions(),
        )

    def validate(self) -> BuildState:
        """Validate state of all targets in the catalog.

        Returns
        -------
        BuildState
            Complete state snapshot for all targets using unified types.
        """
        return self._computer.compute_all()

    def validate_target(self, name: str) -> TargetState:
        """Validate state of a single target.

        This is a convenience method that validates the entire catalog and
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
            If target name is not in the catalog.
        """
        if name not in self._catalog:
            msg = f"Target '{name}' not found in catalog"
            raise KeyError(msg)
        return self._computer.compute_single(name)


def _state_input_values(*, runtime: HamiltonRuntimeBundle, env: BuildEnv) -> Mapping[str, object]:
    inputs = ExecutionInputs(
        env=env,
        catalog=runtime.catalog,
        tag_query=runtime.tag_query,
        cache_index=runtime.cache_index,
        cache_key_resolver=runtime.cache_key_resolver,
        schema_index=runtime.schema_index,
        semantic_registry=runtime.semantic_registry,
        runtime_fingerprint=runtime.fingerprint,
    )
    return execution_input_mapping(inputs)


__all__ = [
    "BuildState",
    "StateValidationOptions",
    "StateValidator",
    "TargetState",
]
