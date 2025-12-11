"""Shared helpers and assertions for pipeline run tracking tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from codeintel.core.execution import RunContext
from codeintel.storage.tracking import PipelineRunTracking
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO, DEFAULT_RUN_ID
from tests._helpers.factories import make_snapshot

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from codeintel.core.execution import RunKind, TriggerKind
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.gateway.protocol import DuckDBConnection
    from codeintel.storage.tracking import PipelineRunRecord, PipelineStepRecord


@dataclass(frozen=True)
class ExpectedRun:
    """Optional expectations for a pipeline run record."""

    run_id: str | None = None
    repo: str | None = None
    status: str | None = None
    kind: str | None = None
    trigger: str | None = None
    pipeline_name: str | None = None
    error_summary: str | None = None
    requested_operation: str | None = None
    requested_datasets: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RunContextOptions:
    """Parameters for building RunContext objects in tests."""

    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    kind: RunKind = "analytics"
    trigger: TriggerKind = "cli"
    requested_operation: str | None = None
    requested_datasets: Sequence[str] = ()


def make_run_context(
    *,
    run_id: str = DEFAULT_RUN_ID,
    repo_root: Path | None = None,
    options: RunContextOptions | None = None,
) -> RunContext:
    """Build a RunContext with sensible defaults for tests.

    Returns
    -------
    RunContext
        Run context configured with provided options.
    """
    opts = options or RunContextOptions()
    snapshot = make_snapshot(repo=opts.repo, commit=opts.commit, repo_root=repo_root)
    return RunContext(
        run_id=run_id,
        kind=opts.kind,
        snapshot=snapshot,
        trigger=opts.trigger,
        requested_operation=opts.requested_operation,
        requested_datasets=tuple(opts.requested_datasets),
    )


def make_tracking(con: DuckDBConnection) -> PipelineRunTracking:
    """Create a PipelineRunTracking accessor for the given connection.

    Returns
    -------
    PipelineRunTracking
        Tracking accessor bound to the provided connection.
    """
    return PipelineRunTracking(con)


def expect_run(record: PipelineRunRecord | None, expected: ExpectedRun) -> PipelineRunRecord:
    """Validate that a run record matches expectations.

    Returns
    -------
    PipelineRunRecord
        The validated record.
    """
    if record is None:
        pytest.fail("Expected run record but got None")

    comparisons = [
        ("run_id", expected.run_id, record.run_id),
        ("repo", expected.repo, record.repo),
        ("status", expected.status, record.status),
        ("kind", expected.kind, record.kind),
        ("trigger", expected.trigger, record.trigger),
        ("pipeline_name", expected.pipeline_name, record.pipeline_name),
        ("error_summary", expected.error_summary, record.error_summary),
        ("requested_operation", expected.requested_operation, record.requested_operation),
        ("requested_datasets", expected.requested_datasets, record.requested_datasets),
    ]
    for label, expected_value, actual in comparisons:
        if expected_value is not None:
            expect_equal(actual, expected_value, label=label)
    return record


def expect_steps(
    steps: list[PipelineStepRecord],
    *,
    expected_count: int | None = None,
    expected_modules: set[str] | None = None,
) -> list[PipelineStepRecord]:
    """Validate step collection shape and optionally module membership.

    Returns
    -------
    list[PipelineStepRecord]
        The provided steps after validation.
    """
    if expected_count is not None:
        expect_equal(len(steps), expected_count, label="step count")
    if expected_modules is not None:
        modules = {step.module for step in steps}
        expect_equal(modules, expected_modules, label="step modules")
    return steps


def expect_step(
    step: PipelineStepRecord,
    *,
    name: str | None = None,
    status: str | None = None,
    row_counts: dict[str, int] | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    """Validate individual step attributes when expectations are provided."""
    if name is not None:
        expect_equal(step.name, name, label="step name")
    if status is not None:
        expect_equal(step.status, status, label="step status")
    if row_counts is not None:
        expect_equal(step.row_counts, row_counts, label="row_counts")
    if extra is not None:
        expect_equal(step.extra, extra, label="extra")
    expect_true(step.started_at is not None, message="Step must include started_at")


@dataclass(frozen=True)
class RunTrackingHarness:
    """Bundle gateway, tracking accessor, and helpers for run-tracking tests."""

    gateway: StorageGateway
    tracking: PipelineRunTracking
    repo_root: Path
    options: RunContextOptions = RunContextOptions()

    def make_context(self, run_id: str, options: RunContextOptions | None = None) -> RunContext:
        """Create a RunContext bound to the harness repo root.

        Returns
        -------
        RunContext
            Context configured for the given run id and options.
        """
        return make_run_context(
            run_id=run_id, repo_root=self.repo_root, options=options or self.options
        )

    def assert_run(
        self,
        run_id: str,
        expected: ExpectedRun,
    ) -> PipelineRunRecord:
        """Fetch and validate a run record.

        Returns
        -------
        PipelineRunRecord
            The validated run record.
        """
        return expect_run(self.tracking.fetch_run(run_id), expected)

    def assert_steps(
        self,
        run_id: str,
        *,
        expected_count: int | None = None,
        expected_row_counts: dict[str, int] | None = None,
    ) -> list[PipelineStepRecord]:
        """Fetch steps and validate shape/row counts.

        Returns
        -------
        list[PipelineStepRecord]
            Retrieved step records after validation.
        """
        steps = expect_steps(
            self.tracking.fetch_steps(run_id),
            expected_count=expected_count,
        )
        if expected_row_counts is not None and steps:
            expect_step(steps[-1], row_counts=expected_row_counts)
        return steps


__all__ = [
    "ExpectedRun",
    "RunContextOptions",
    "RunTrackingHarness",
    "expect_run",
    "expect_step",
    "expect_steps",
    "make_run_context",
    "make_tracking",
]
