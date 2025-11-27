"""Prefect task wrapper factory for pipeline steps."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from prefect import task

from codeintel.pipeline.orchestration.core import PipelineStep

if TYPE_CHECKING:
    from codeintel.pipeline.orchestration.core import PipelineContext
    from codeintel.pipeline.orchestration.registry import StepRegistry

log = logging.getLogger(__name__)


def make_prefect_task(
    step: PipelineStep,
    *,
    retries: int = 1,
    retry_delay_seconds: int = 2,
) -> Callable[[PipelineContext], None]:
    """
    Wrap a PipelineStep as a Prefect task with retries.

    Parameters
    ----------
    step
        The pipeline step to wrap.
    retries
        Number of retry attempts on failure.
    retry_delay_seconds
        Delay between retries in seconds.

    Returns
    -------
    Callable[[PipelineContext], None]
        A Prefect task function that executes the step.

    Examples
    --------
    >>> from codeintel.pipeline.orchestration.steps_ingestion import RepoScanStep
    >>> task_fn = make_prefect_task(RepoScanStep(), retries=2)
    >>> # task_fn(ctx)  # Execute as a Prefect task
    """

    @task(name=step.name, retries=retries, retry_delay_seconds=retry_delay_seconds)
    def _task(ctx: PipelineContext) -> None:
        log.debug("Running Prefect task for step: %s", step.name)
        step.run(ctx)

    # Preserve step metadata on the task function
    _task.step = step  # type: ignore[attr-defined]
    _task.step_name = step.name  # type: ignore[attr-defined]
    return _task


def build_task_map(
    registry: StepRegistry,
    *,
    retries: int = 1,
    retry_delay_seconds: int = 2,
) -> dict[str, Callable[[PipelineContext], None]]:
    """
    Build a mapping of step names to Prefect task functions.

    Parameters
    ----------
    registry
        The step registry to build tasks from.
    retries
        Number of retry attempts on failure for each task.
    retry_delay_seconds
        Delay between retries in seconds.

    Returns
    -------
    dict[str, Callable[[PipelineContext], None]]
        Mapping of step name to Prefect task function.
    """
    return {
        name: make_prefect_task(
            registry[name],
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
        )
        for name in registry
    }


def run_pipeline_as_tasks(
    ctx: PipelineContext,
    registry: StepRegistry,
    *,
    selected_steps: Sequence[str] | None = None,
    retries: int = 1,
    retry_delay_seconds: int = 2,
) -> None:
    """
    Execute pipeline steps as individual Prefect tasks.

    Each step is wrapped as a Prefect task with configurable retries.
    Steps are executed in topological order based on dependencies.

    Parameters
    ----------
    ctx
        PipelineContext containing configs and runtime services.
    registry
        The step registry containing all pipeline steps.
    selected_steps
        Optional subset of steps to execute; dependencies are included automatically.
    retries
        Number of retry attempts on failure for each task.
    retry_delay_seconds
        Delay between retries in seconds.
    """
    step_names = tuple(selected_steps) if selected_steps is not None else registry.list_all_names()

    # Expand with dependencies
    expanded = registry.expand_with_deps(step_names)

    # Preserve sequence order for expanded steps
    sequence = registry.list_all_names()
    ordered_names = [name for name in sequence if name in expanded]

    # Topological sort
    ordered = registry.topological_order(tuple(ordered_names))

    # Build tasks on demand
    task_cache: dict[str, Callable[[PipelineContext], None]] = {}

    def get_task(name: str) -> Callable[[PipelineContext], None]:
        if name not in task_cache:
            task_cache[name] = make_prefect_task(
                registry[name],
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
            )
        return task_cache[name]

    # Execute tasks in order
    for name in ordered:
        task_fn = get_task(name)
        log.info("Executing step as Prefect task: %s", name)
        task_fn(ctx)


__all__ = [
    "build_task_map",
    "make_prefect_task",
    "run_pipeline_as_tasks",
]
