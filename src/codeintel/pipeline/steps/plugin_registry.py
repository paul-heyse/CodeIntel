"""Pipeline step registry backed by BasePluginRegistry infrastructure.

This module provides a unified registry for pipeline steps that leverages
the core plugin registry infrastructure, enabling consistent dependency
resolution, topological sorting, and planning across all plugin types.

The StepPluginRegistry extends BasePluginRegistry to provide:
- Registration and lookup of pipeline steps
- Dependency resolution using the core topological sort
- Planning utilities for step execution
- Entry point discovery for external step plugins
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.core.execution.ids import new_run_id
from codeintel.core.plugins.registry.base import (
    BasePluginRegistry,
    PluginPlan,
    PluginSkip,
)
from codeintel.pipeline.steps.base import PipelineStep, StepPhase

if TYPE_CHECKING:
    from codeintel.pipeline.execution.context import PipelineContext

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepPlan:
    """Execution plan for pipeline steps.

    This is a specialized plan type that wraps PluginPlan with
    step-specific utilities.

    Attributes
    ----------
    steps
        Ordered tuple of steps to execute.
    plan_id
        Unique identifier for this plan.
    skipped
        Steps that were excluded from the plan.
    dep_graph
        Dependency graph (step name -> dependencies).
    """

    steps: tuple[PipelineStep, ...]
    plan_id: str = ""
    skipped: tuple[PluginSkip, ...] = ()
    dep_graph: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def ordered_names(self) -> tuple[str, ...]:
        """Return step names in execution order.

        Returns
        -------
        tuple[str, ...]
            Step names in execution order.
        """
        return tuple(s.name for s in self.steps)

    @classmethod
    def from_plugin_plan(cls, plan: PluginPlan[PipelineStep]) -> StepPlan:
        """Create a StepPlan from a generic PluginPlan.

        Parameters
        ----------
        plan
            Generic plugin plan to convert.

        Returns
        -------
        StepPlan
            Step-specific plan with the same contents.
        """
        return cls(
            steps=plan.plugins,
            plan_id=plan.plan_id,
            skipped=plan.skipped,
            dep_graph=dict(plan.dep_graph),
        )


class StepPluginRegistry(BasePluginRegistry[PipelineStep]):
    """Pipeline step registry backed by BasePluginRegistry infrastructure.

    This registry provides a unified way to manage pipeline steps using
    the same infrastructure as graph and analytics plugins. Steps are
    registered with their metadata and can be planned for execution
    with automatic dependency resolution.

    Examples
    --------
    >>> registry = StepPluginRegistry()
    >>> registry.register(my_step)
    >>> plan = registry.plan()
    >>> for step in plan.steps:
    ...     step.run(ctx)
    """

    def __init__(self) -> None:
        """Initialize an empty step registry."""
        super().__init__()

    def __len__(self) -> int:
        """Return the number of registered steps.

        Returns
        -------
        int
            Number of registered steps.
        """
        self._ensure_loaded()
        return len(self._plugins)

    def __iter__(self) -> Iterator[str]:
        """Iterate over step names.

        Returns
        -------
        Iterator[str]
            Iterator over step names.
        """
        self._ensure_loaded()
        return iter(self._plugins)

    def __contains__(self, name: object) -> bool:
        """Check if a step name is registered.

        Returns
        -------
        bool
            True if the step name is registered.
        """
        self._ensure_loaded()
        return name in self._plugins

    def __getitem__(self, name: str) -> PipelineStep:
        """Retrieve a step by name, raising KeyError if not found.

        Returns
        -------
        PipelineStep
            The step instance.

        Raises
        ------
        KeyError
            If the step name is not registered.
        """
        self._ensure_loaded()
        step = self._plugins.get(name)
        if step is None:
            message = f"Unknown pipeline step: {name}"
            raise KeyError(message)
        return step

    def as_dict(self) -> dict[str, PipelineStep]:
        """Return the steps as a mutable dictionary.

        Returns
        -------
        dict[str, PipelineStep]
            Copy of the steps mapping.
        """
        self._ensure_loaded()
        return dict(self._plugins)

    @property
    def _default_entrypoint_group(self) -> str:
        """Return the entry point group for step discovery.

        Returns
        -------
        str
            Entry point group name.
        """
        return "codeintel.pipeline.steps"

    @staticmethod
    def _get_default_plugins() -> Sequence[str]:
        """Return the default step names.

        Steps are registered explicitly, so this returns an empty sequence.

        Returns
        -------
        Sequence[str]
            Empty sequence (steps are registered explicitly).
        """
        return ()

    def _ensure_builtins_loaded(self) -> None:
        """Load built-in steps if not already done.

        Steps are registered explicitly via register(), so this is a no-op.
        """
        self._builtins_loaded = True

    def list_by_phase(self, phase: StepPhase) -> tuple[PipelineStep, ...]:
        """Return steps belonging to a specific phase.

        Parameters
        ----------
        phase
            Phase to filter by.

        Returns
        -------
        tuple[PipelineStep, ...]
            Steps in the specified phase.
        """
        self._ensure_loaded()
        return tuple(s for s in self._plugins.values() if s.phase == phase)

    def plan(
        self,
        *,
        step_names: Sequence[str] | None = None,
        enabled: Sequence[str] | None = None,
        disabled: Sequence[str] | None = None,
    ) -> StepPlan:
        """Build an execution plan for steps.

        Parameters
        ----------
        step_names
            Explicit list of steps to include.
        enabled
            Steps to enable (overrides defaults).
        disabled
            Steps to disable.

        Returns
        -------
        StepPlan
            Execution plan with steps in dependency order.
        """
        self._ensure_loaded()

        # Use all registered steps as defaults if none specified
        defaults = list(self._plugins.keys())

        # Resolve which steps to include
        selected, skipped = self._resolve_selection(
            plugin_names=step_names,
            enabled=enabled,
            disabled=disabled,
            defaults=defaults,
        )

        # Build dependency graph
        dependencies = self._resolve_dependencies(selected)

        # Topological sort
        ordered = self._topological_sort(selected, dependencies)

        # Build dependency graph for plan
        dep_graph = {name: tuple(deps) for name, deps in dependencies.items()}

        # Create the plan
        return StepPlan(
            steps=tuple(ordered),
            plan_id=new_run_id("step-plan"),
            skipped=skipped,
            dep_graph=dep_graph,
        )

    def expand_with_deps(self, names: Sequence[str]) -> set[str]:
        """Expand a set of step names to include all transitive dependencies.

        Parameters
        ----------
        names
            Step names to expand.

        Returns
        -------
        set[str]
            Set of step names including all transitive dependencies.
        """
        self._ensure_loaded()
        expanded: set[str] = set()
        for name in names:
            self._expand_recursive(name, expanded)
        return expanded

    def _expand_recursive(self, name: str, expanded: set[str]) -> None:
        """Recursively expand dependencies for a step."""
        if name in expanded:
            return
        step = self.get(name)
        for dep in step.metadata.depends_on:
            self._expand_recursive(dep, expanded)
        expanded.add(name)

    def topological_order(self, names: Sequence[str]) -> list[str]:
        """Return a topological ordering of the requested steps.

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
        KeyError
            If any step name is not registered.
        """
        self._ensure_loaded()

        # Validate all names exist
        for name in names:
            if name not in self._plugins:
                message = f"Unknown pipeline step: {name}"
                raise KeyError(message)

        # Build dependency set for sorting
        selected = {name: self._plugins[name] for name in names}
        dependencies = self._resolve_dependencies(selected)

        # Get ordered plugins and extract names
        ordered = self._topological_sort(selected, dependencies)
        return [s.name for s in ordered]

    def list_all_names(self) -> tuple[str, ...]:
        """Return all step names in registration order.

        Returns
        -------
        tuple[str, ...]
            Step names in registration order.
        """
        self._ensure_loaded()
        return tuple(self._plugins.keys())

    def get_deps(self, name: str) -> tuple[str, ...]:
        """Return direct dependencies for a step.

        Parameters
        ----------
        name
            Step name to get dependencies for.

        Returns
        -------
        tuple[str, ...]
            Direct dependency names for the step.

        Note
        ----
        Raises KeyError if the step name is not registered (via __getitem__).
        """
        return self[name].metadata.depends_on

    def execute(
        self,
        ctx: PipelineContext,
        selected_steps: Sequence[str] | None = None,
    ) -> None:
        """Execute pipeline steps in topological order.

        Parameters
        ----------
        ctx
            PipelineContext containing configs and runtime services.
        selected_steps
            Optional subset of steps to execute; dependencies are included automatically.
        """
        if selected_steps is not None:
            step_plan = self.plan(step_names=selected_steps)
        else:
            step_plan = self.plan()

        for step in step_plan.steps:
            log.debug("Executing pipeline step: %s", step.name)
            step.run(ctx)


def build_step_plugin_registry(
    steps: dict[str, PipelineStep],
) -> StepPluginRegistry:
    """Build a StepPluginRegistry from a dictionary of steps.

    Parameters
    ----------
    steps
        Mapping of step name to step instance.

    Returns
    -------
    StepPluginRegistry
        Registry containing all provided steps.

    Examples
    --------
    >>> registry = build_step_plugin_registry(
    ...     {
    ...         "repo_scan": repo_scan_step,
    ...         "graph_build": graph_build_step,
    ...     }
    ... )
    """
    registry = StepPluginRegistry()
    for step in steps.values():
        registry.register(step)
    return registry


__all__ = [
    "StepPlan",
    "StepPluginRegistry",
    "build_step_plugin_registry",
]
