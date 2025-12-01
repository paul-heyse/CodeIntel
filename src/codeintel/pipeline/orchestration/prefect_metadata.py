"""Helpers for attaching metadata to Prefect tasks without direct attr writes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    from codeintel.pipeline.orchestration.core import PipelineStep


@dataclass(frozen=True)
class TaskMetadata:
    """Metadata associated with a Prefect task or task-like callable."""

    step: PipelineStep | None
    step_name: str | None
    fn: Callable[..., object] | None


_TASK_METADATA: WeakKeyDictionary[Callable[..., object], TaskMetadata] = WeakKeyDictionary()


def attach_task_metadata(
    task_fn: Callable[..., object],
    *,
    step: PipelineStep | None = None,
    step_name: str | None = None,
    fn: Callable[..., object] | None = None,
) -> Callable[..., object]:
    """
    Record metadata for a Prefect task or callable.

    Uses a WeakKeyDictionary to avoid mutating the task object with ad-hoc attributes.

    Parameters
    ----------
    task_fn
        Task callable produced by Prefect or a plain function.
    step
        Optional pipeline step associated with the task.
    step_name
        Optional step name; defaults to ``step.name`` when provided.
    fn
        Optional underlying function (for task-like wrappers that expose ``fn``).

    Returns
    -------
    Callable[..., object]
        The original task callable.
    """
    name = step_name or getattr(step, "name", None)
    _TASK_METADATA[task_fn] = TaskMetadata(step=step, step_name=name, fn=fn)
    tags = getattr(task_fn, "tags", None)
    if name is not None and isinstance(tags, set):
        tags.add(f"step:{name}")
    return task_fn


def get_task_metadata(task_fn: Callable[..., object]) -> TaskMetadata | None:
    """
    Retrieve metadata for a Prefect task or callable.

    Returns
    -------
    TaskMetadata | None
        Metadata when previously attached; otherwise ``None``.
    """
    return _TASK_METADATA.get(task_fn)


__all__ = ["TaskMetadata", "attach_task_metadata", "get_task_metadata"]
