"""Integration tests for run_registry wiring via gateway."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.core.execution import RunContext, new_run_context
from codeintel.storage.tracking import PipelineStepRecord
from tests._helpers.gateway import open_ingestion_gateway_with_macros

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.tracking.run_tracking import PipelineRunRecord


@dataclass(frozen=True)
class ExpectedRun:
    """Expected values for run assertions."""

    run_id: str | None = None
    repo: str | None = None
    status: str | None = None
    kind: str | None = None
    trigger: str | None = None
    error_summary: str | None = None


def _require_run(rec: PipelineRunRecord | None, expected: ExpectedRun) -> PipelineRunRecord:
    if rec is None:
        pytest.fail("Expected run record but got None")
    if expected.run_id is not None and rec.run_id != expected.run_id:
        pytest.fail(f"Expected run_id {expected.run_id} but got {rec.run_id}")
    if expected.repo is not None and rec.repo != expected.repo:
        pytest.fail(f"Expected repo {expected.repo} but got {rec.repo}")
    if expected.status is not None and rec.status != expected.status:
        pytest.fail(f"Expected status {expected.status} but got {rec.status}")
    if expected.kind is not None and rec.kind != expected.kind:
        pytest.fail(f"Expected kind {expected.kind} but got {rec.kind}")
    if expected.trigger is not None and rec.trigger != expected.trigger:
        pytest.fail(f"Expected trigger {expected.trigger} but got {rec.trigger}")
    if expected.error_summary is not None and rec.error_summary != expected.error_summary:
        pytest.fail(f"Expected error summary {expected.error_summary} but got {rec.error_summary}")
    return rec


def _require_steps(
    steps: list[PipelineStepRecord],
    *,
    expected_count: int | None = None,
    expected_modules: set[str] | None = None,
) -> list[PipelineStepRecord]:
    if expected_count is not None and len(steps) != expected_count:
        pytest.fail(f"Expected {expected_count} steps but got {len(steps)}")
    if expected_modules is not None:
        modules = {step.module for step in steps}
        if modules != expected_modules:
            pytest.fail(f"Expected modules {expected_modules} but got {modules}")
    return steps


def _require_step(
    step: PipelineStepRecord,
    *,
    name: str | None = None,
    status: str | None = None,
    row_counts: dict[str, int] | None = None,
    extra: dict[str, str] | None = None,
) -> None:
    if name is not None and step.name != name:
        pytest.fail(f"Expected step name {name} but got {step.name}")
    if status is not None and step.status != status:
        pytest.fail(f"Expected step status {status} but got {step.status}")
    if row_counts is not None and step.row_counts != row_counts:
        pytest.fail(f"Expected row_counts {row_counts} but got {step.row_counts}")
    if extra is not None and step.extra != extra:
        pytest.fail(f"Expected extra {extra} but got {step.extra}")


@pytest.fixture
def gateway() -> StorageGateway:
    """Create an in-memory gateway with full schema for integration tests.

    Returns
    -------
    StorageGateway
        In-memory gateway configured for integration tests.
    """
    return open_ingestion_gateway_with_macros(
        apply_schema=True,
        ensure_views=True,
        validate_schema=False,  # Allow missing tables in tests
        strict_schema=False,
    )


@pytest.fixture
def sample_snapshot(tmp_path: Path) -> SnapshotRef:
    """Create a sample snapshot for testing.

    Returns
    -------
    SnapshotRef
        Sample snapshot reference for integration tests.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    return SnapshotRef(
        repo="github.com/demo/repo",
        commit="deadbeef" * 5,
        repo_root=repo_root,
    )


class TestGatewayRunsApi:
    """Test that the gateway.runs API functions work correctly."""

    @staticmethod
    def test_start_and_fetch_run_with_gateway(
        gateway: StorageGateway,
        sample_snapshot: SnapshotRef,
    ) -> None:
        """Verify start_run and fetch_run work with gateway.runs."""
        ctx = RunContext(
            run_id="integ-123",
            kind="ingest",
            snapshot=sample_snapshot,
            trigger="cli",
        )

        gateway.runs.start_run(
            ctx,
            pipeline_name="ingest:default",
        )

        rec = gateway.runs.fetch_run("integ-123")
        _require_run(
            rec,
            ExpectedRun(
                run_id="integ-123",
                repo="github.com/demo/repo",
                status="running",
                kind="ingest",
                trigger="cli",
            ),
        )

    @staticmethod
    def test_complete_run_with_gateway(
        gateway: StorageGateway,
        sample_snapshot: SnapshotRef,
    ) -> None:
        """Verify complete_run works with gateway.runs."""
        ctx = RunContext(
            run_id="integ-456",
            kind="analytics",
            snapshot=sample_snapshot,
            trigger="http",
        )

        gateway.runs.start_run(ctx, pipeline_name="analytics")
        gateway.runs.complete_run(
            "integ-456",
            status="succeeded",
        )

        rec = gateway.runs.fetch_run("integ-456")
        _require_run(
            rec,
            ExpectedRun(status="succeeded", kind="analytics", trigger="http"),
        )

    @staticmethod
    def test_record_steps_with_gateway(
        gateway: StorageGateway,
        sample_snapshot: SnapshotRef,
    ) -> None:
        """Verify step recording works with gateway.runs."""
        ctx = RunContext(
            run_id="integ-789",
            kind="graphs",
            snapshot=sample_snapshot,
            trigger="api",
        )

        gateway.runs.start_run(ctx, pipeline_name="graphs")

        now = datetime.now(tz=UTC)
        gateway.runs.record_step(
            PipelineStepRecord(
                run_id="integ-789",
                module="graphs",
                stage="build",
                name="call_graph_builder",
                status="succeeded",
                started_at=now,
                completed_at=now,
                row_counts={"graph.call_graph_edges": 100},
            ),
        )

        steps = gateway.runs.fetch_steps("integ-789")
        steps = _require_steps(steps, expected_count=1)
        _require_step(
            steps[0],
            name="call_graph_builder",
            row_counts={"graph.call_graph_edges": 100},
        )


class TestNewRunContextIntegration:
    """Test new_run_context factory integration."""

    @staticmethod
    def test_new_run_context_for_ingest(sample_snapshot: SnapshotRef) -> None:
        """Verify new_run_context generates correct ingest context."""
        ctx = new_run_context(
            snapshot=sample_snapshot,
            kind="ingest",
            trigger="cli",
            requested_operation="scan",
        )

        if not ctx.run_id.startswith("ingest-"):
            pytest.fail(f"Run id should start with ingest- but was {ctx.run_id}")
        if ctx.kind != "ingest":
            pytest.fail(f"Expected kind ingest but got {ctx.kind}")
        if ctx.trigger != "cli":
            pytest.fail(f"Expected trigger cli but got {ctx.trigger}")
        if ctx.requested_operation != "scan":
            pytest.fail(f"Expected operation scan but got {ctx.requested_operation}")
        if ctx.snapshot.repo != sample_snapshot.repo:
            pytest.fail("Snapshot repo mismatch")

    @staticmethod
    def test_new_run_context_for_analytics(sample_snapshot: SnapshotRef) -> None:
        """Verify new_run_context generates correct analytics context."""
        ctx = new_run_context(
            snapshot=sample_snapshot,
            kind="analytics",
            trigger="http",
            requested_datasets=("analytics.function_metrics",),
        )

        if not ctx.run_id.startswith("analytics-"):
            pytest.fail(f"Run id should start with analytics- but was {ctx.run_id}")
        if ctx.kind != "analytics":
            pytest.fail(f"Expected kind analytics but got {ctx.kind}")
        if ctx.trigger != "http":
            pytest.fail(f"Expected trigger http but got {ctx.trigger}")
        if ctx.requested_datasets != ("analytics.function_metrics",):
            pytest.fail("Requested datasets mismatch")

    @staticmethod
    def test_new_run_context_for_graphs(sample_snapshot: SnapshotRef) -> None:
        """Verify new_run_context generates correct graphs context."""
        ctx = new_run_context(
            snapshot=sample_snapshot,
            kind="graphs",
            trigger="mcp",
        )

        if not ctx.run_id.startswith("graphs-"):
            pytest.fail(f"Run id should start with graphs- but was {ctx.run_id}")
        if ctx.kind != "graphs":
            pytest.fail(f"Expected kind graphs but got {ctx.kind}")
        if ctx.trigger != "mcp":
            pytest.fail(f"Expected trigger mcp but got {ctx.trigger}")


class TestFullRunLifecycle:
    """Test complete run lifecycle scenarios."""

    @staticmethod
    def test_multi_engine_run_lifecycle(
        gateway: StorageGateway,
        sample_snapshot: SnapshotRef,
    ) -> None:
        """Verify a full run lifecycle across multiple modules."""
        # Create a full pipeline run context
        ctx = new_run_context(
            snapshot=sample_snapshot,
            kind="full",
            trigger="cli",
        )

        # Start the run via gateway
        gateway.runs.start_run(
            ctx,
            pipeline_name="full:default",
        )

        now = datetime.now(tz=UTC)

        # Record ingestion step
        gateway.runs.record_step(
            PipelineStepRecord(
                run_id=ctx.run_id,
                module="ingestion",
                stage="scan",
                name="repo_scan",
                status="succeeded",
                started_at=now,
                completed_at=now,
                row_counts={"core.modules": 10},
            ),
        )

        # Record graphs step
        gateway.runs.record_step(
            PipelineStepRecord(
                run_id=ctx.run_id,
                module="graphs",
                stage="build",
                name="call_graph_builder",
                status="succeeded",
                started_at=now,
                completed_at=now,
                row_counts={"graph.call_graph_edges": 50},
            ),
        )

        # Record analytics step
        gateway.runs.record_step(
            PipelineStepRecord(
                run_id=ctx.run_id,
                module="analytics",
                stage="function",
                name="function_metrics",
                status="succeeded",
                started_at=now,
                completed_at=now,
                row_counts={"analytics.function_metrics": 10},
            ),
        )

        # Complete the run
        gateway.runs.complete_run(
            ctx.run_id,
            status="succeeded",
        )

        # Verify run record
        _require_run(
            gateway.runs.fetch_run(ctx.run_id),
            ExpectedRun(status="succeeded", kind="full"),
        )

        # Verify steps
        steps = _require_steps(
            gateway.runs.fetch_steps(ctx.run_id),
            expected_count=3,
            expected_modules={"ingestion", "graphs", "analytics"},
        )
        for step in steps:
            if step.module == "ingestion":
                _require_step(
                    step,
                    name="repo_scan",
                    status="succeeded",
                    row_counts={"core.modules": 10},
                )
            if step.module == "graphs":
                _require_step(
                    step,
                    name="call_graph_builder",
                    status="succeeded",
                    row_counts={"graph.call_graph_edges": 50},
                )
            if step.module == "analytics":
                _require_step(
                    step,
                    name="function_metrics",
                    status="succeeded",
                    row_counts={"analytics.function_metrics": 10},
                )

    @staticmethod
    def test_failed_run_lifecycle(
        gateway: StorageGateway,
        sample_snapshot: SnapshotRef,
    ) -> None:
        """Verify failed run is recorded correctly."""
        ctx = new_run_context(
            snapshot=sample_snapshot,
            kind="analytics",
            trigger="api",
        )

        gateway.runs.start_run(ctx, pipeline_name="analytics")

        now = datetime.now(tz=UTC)

        # Record a failed step
        gateway.runs.record_step(
            PipelineStepRecord(
                run_id=ctx.run_id,
                module="analytics",
                stage="function",
                name="function_metrics",
                status="failed",
                started_at=now,
                completed_at=now,
                extra={"error": "Database connection failed"},
            ),
        )

        # Complete the run as failed
        gateway.runs.complete_run(
            ctx.run_id,
            status="failed",
            error_summary="function_metrics plugin failed",
        )

        # Verify
        _require_run(
            gateway.runs.fetch_run(ctx.run_id),
            ExpectedRun(status="failed", error_summary="function_metrics plugin failed"),
        )

        steps = _require_steps(gateway.runs.fetch_steps(ctx.run_id), expected_count=1)
        _require_step(
            steps[0],
            status="failed",
            extra={"error": "Database connection failed"},
        )
