"""Pipeline step registry with query and execution APIs."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.pipeline.orchestration.core import (
    PipelineStep,
    StepMetadata,
    StepPhase,
)

if TYPE_CHECKING:
    from codeintel.pipeline.orchestration.core import PipelineContext

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepRegistry:
    """
    Unified registry for pipeline steps with query and execution APIs.

    The registry provides a single source of truth for step metadata,
    dependencies, and execution order. It supports:
    - Querying steps by name or phase
    - Topological sorting of step dependencies
    - Expanding step selections to include required dependencies
    - Executing steps in correct order

    Parameters
    ----------
    _steps
        Mapping of step names to step instances.
    _sequence
        Ordered sequence of step names (insertion order defines default ordering).
    """

    _steps: Mapping[str, PipelineStep]
    _sequence: tuple[str, ...] = field(default_factory=tuple)

    def get(self, name: str) -> PipelineStep | None:
        """
        Retrieve a step by name.

        Parameters
        ----------
        name
            Step name to look up.

        Returns
        -------
        PipelineStep | None
            The step if found, None otherwise.
        """
        return self._steps.get(name)

    def __getitem__(self, name: str) -> PipelineStep:
        """
        Retrieve a step by name, raising KeyError if not found.

        Parameters
        ----------
        name
            Step name to look up.

        Returns
        -------
        PipelineStep
            The step instance.

        Raises
        ------
        KeyError
            If the step name is not registered.
        """
        step = self._steps.get(name)
        if step is None:
            message = f"Unknown pipeline step: {name}"
            raise KeyError(message)
        return step

    def __contains__(self, name: str) -> bool:
        """
        Check if a step name is registered.

        Returns
        -------
        bool
            True if the step is registered, False otherwise.
        """
        return name in self._steps

    def __len__(self) -> int:
        """
        Return the number of registered steps.

        Returns
        -------
        int
            Number of registered steps.
        """
        return len(self._steps)

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over step names in sequence order.

        Returns
        -------
        Iterator[str]
            Iterator over step names.
        """
        return iter(self._sequence)

    def list_all(self) -> list[StepMetadata]:
        """
        Return metadata for all registered steps in sequence order.

        Returns
        -------
        list[StepMetadata]
            List of step metadata in registration order.
        """
        return [self._step_metadata(name) for name in self._sequence]

    def list_all_names(self) -> tuple[str, ...]:
        """
        Return all step names in sequence order.

        Returns
        -------
        tuple[str, ...]
            Step names in registration order.
        """
        return self._sequence

    def list_by_phase(self, phase: StepPhase) -> list[StepMetadata]:
        """
        Return metadata for steps belonging to a specific phase.

        Parameters
        ----------
        phase
            The phase to filter by.

        Returns
        -------
        list[StepMetadata]
            List of step metadata for the given phase, in sequence order.
        """
        return [
            self._step_metadata(name) for name in self._sequence if self._steps[name].phase == phase
        ]

    def dependency_graph(self) -> dict[str, tuple[str, ...]]:
        """
        Return the full dependency graph as a mapping.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Mapping of step name to tuple of dependency names.
        """
        return {name: tuple(step.deps) for name, step in self._steps.items()}

    def get_deps(self, name: str) -> tuple[str, ...]:
        """
        Return the direct dependencies of a step.

        Parameters
        ----------
        name
            Step name to query.

        Returns
        -------
        tuple[str, ...]
            Tuple of direct dependency names.
        """
        return tuple(self[name].deps)

    def expand_with_deps(self, names: Sequence[str]) -> set[str]:
        """
        Expand a set of step names to include all transitive dependencies.

        Parameters
        ----------
        names
            Step names to expand.

        Returns
        -------
        set[str]
            Set of step names including all transitive dependencies.
        """
        expanded: set[str] = set()
        for name in names:
            self._expand_recursive(name, expanded)
        return expanded

    def _expand_recursive(self, name: str, expanded: set[str]) -> None:
        """Recursively expand dependencies for a step."""
        if name in expanded:
            return
        step = self[name]  # Raises KeyError if not found
        for dep in step.deps:
            self._expand_recursive(dep, expanded)
        expanded.add(name)

    def topological_order(self, names: Sequence[str]) -> list[str]:
        """
        Return a topological ordering of the requested steps.

        Parameters
        ----------
        names
            Step names to order.

        Returns
        -------
        list[str]
            Steps ordered to respect declared dependencies.

        Raises
        ------
        RuntimeError
            If a dependency cycle is detected.
        KeyError
            If any step name is not registered.
        """
        # Validate all names exist
        for name in names:
            if name not in self._steps:
                message = f"Unknown pipeline step: {name}"
                raise KeyError(message)

        deps = {name: set(self._steps[name].deps) & set(names) for name in names}
        remaining = set(names)
        ordered: list[str] = []
        no_deps = [name for name in names if not deps[name]]

        while no_deps:
            name = no_deps.pop()
            ordered.append(name)
            remaining.discard(name)
            for other in list(remaining):
                deps[other].discard(name)
                if not deps[other]:
                    no_deps.append(other)

        if remaining:
            message = f"Circular dependencies detected: {sorted(remaining)}"
            raise RuntimeError(message)
        return ordered

    def execute(
        self,
        ctx: PipelineContext,
        selected_steps: Sequence[str] | None = None,
    ) -> None:
        """
        Execute pipeline steps in topological order.

        Parameters
        ----------
        ctx
            PipelineContext containing configs and runtime services.
        selected_steps
            Optional subset of steps to execute; dependencies are included automatically.
        """
        step_names = tuple(selected_steps) if selected_steps is not None else self._sequence

        # Expand with dependencies
        expanded = self.expand_with_deps(step_names)

        # Preserve sequence order for expanded steps
        ordered_names = [name for name in self._sequence if name in expanded]

        # Topological sort
        ordered = self.topological_order(tuple(ordered_names))

        # Execute steps
        for name in ordered:
            step = self._steps[name]
            log.debug("Executing pipeline step: %s", name)
            step.run(ctx)

    def as_dict(self) -> dict[str, PipelineStep]:
        """
        Return the steps as a mutable dictionary.

        Returns
        -------
        dict[str, PipelineStep]
            Copy of the steps mapping.
        """
        return dict(self._steps)

    def _step_metadata(self, name: str) -> StepMetadata:
        """
        Build StepMetadata for a step by name.

        Returns
        -------
        StepMetadata
            Metadata for the step.
        """
        step = self._steps[name]
        return StepMetadata(
            name=step.name,
            description=step.description,
            phase=step.phase,
            deps=tuple(step.deps),
        )


def build_registry(*step_dicts: Mapping[str, PipelineStep]) -> StepRegistry:
    """
    Build a StepRegistry from one or more step dictionaries.

    Parameters
    ----------
    step_dicts
        One or more mappings of step name to step instance.

    Returns
    -------
    StepRegistry
        Registry containing all provided steps.

    Examples
    --------
    >>> registry = build_registry(INGESTION_STEPS, GRAPH_STEPS, ANALYTICS_STEPS)
    >>> registry.get("repo_scan")
    RepoScanStep(...)
    """
    merged: dict[str, PipelineStep] = {}
    sequence: list[str] = []
    for step_dict in step_dicts:
        for name, step in step_dict.items():
            if name in merged:
                log.warning("Duplicate step name '%s' in registry; overwriting.", name)
            merged[name] = step
            if name not in sequence:
                sequence.append(name)
    return StepRegistry(_steps=merged, _sequence=tuple(sequence))


__all__ = [
    "StepRegistry",
    "build_registry",
]
