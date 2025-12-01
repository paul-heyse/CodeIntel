"""Tests for Prefect metadata helpers."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from prefect import task

from codeintel.pipeline.orchestration.core import PipelineStep, StepPhase
from codeintel.pipeline.orchestration.prefect_metadata import (
    attach_task_metadata,
    get_task_metadata,
)


class _StubStep(PipelineStep):
    name = "stub"
    description = "stub step"
    phase = StepPhase.INGESTION
    deps: tuple[str, ...] = ()

    @staticmethod
    def run(ctx: object) -> None:  # pragma: no cover - not used
        _ = ctx


def test_attach_task_metadata_records_step() -> None:
    """Metadata helper should record step and name for a Prefect task."""

    @task
    def sample() -> None:
        return None

    step = _StubStep()
    task_fn = attach_task_metadata(sample, step=step, fn=getattr(sample, "fn", None))
    metadata = get_task_metadata(task_fn)
    if metadata is None:
        pytest.fail("Expected metadata to be registered")
    if metadata.step is not step:
        pytest.fail("Step should be preserved in metadata")
    if metadata.step_name != step.name:
        pytest.fail("Step name should be preserved in metadata")
    if metadata.fn is not getattr(task_fn, "fn", None):
        pytest.fail("Underlying fn should be preserved in metadata")


def test_attach_task_metadata_handles_plain_callables() -> None:
    """Helper should accept plain callables and round-trip metadata."""
    called: list[bool] = []

    def fn() -> None:
        called.append(True)

    task_like: Callable[[], None] = attach_task_metadata(fn, step_name="plain", fn=fn)
    task_like()
    metadata = get_task_metadata(task_like)
    if metadata is None:
        pytest.fail("Metadata should be registered for plain callable")
    if metadata.step is not None:
        pytest.fail("Plain callable should not record a step")
    if metadata.step_name != "plain":
        pytest.fail("Expected explicit step_name to be preserved")
    if metadata.fn is not fn:
        pytest.fail("Underlying fn should be preserved")
    if not called:
        pytest.fail("Callable should be executed")
