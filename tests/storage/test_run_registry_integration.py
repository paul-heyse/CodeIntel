"""Integration tests for run_registry wiring via gateway."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.core.execution import RunContext, new_run_context
from codeintel.storage.tracking import PipelineStepRecord
from tests._helpers.run_tracking import (
    ExpectedRun,
    expect_run,
    expect_step,
    expect_steps,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.run_tracking import (
        RunTrackingHarness,
    )


@pytest.fixture
def gateway(run_tracking_harness: RunTrackingHarness) -> StorageGateway:
    """Reuse the shared run-tracking gateway fixture for integration tests.

    Returns
    -------
    StorageGateway
        Gateway bound to the shared run-tracking harness.
    """
    return run_tracking_harness.gateway


@pytest.fixture
def sample_snapshot(run_tracking_harness: RunTrackingHarness) -> SnapshotRef:
    """Create a sample snapshot for testing.

    Returns
    -------
    SnapshotRef
        Sample snapshot reference for integration tests.
    """
    repo_root = run_tracking_harness.repo_root
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

        expect_run(
            gateway.runs.fetch_run("integ-123"),
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

        expect_run(
            gateway.runs.fetch_run("integ-456"),
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

        steps = expect_steps(gateway.runs.fetch_steps("integ-789"), expected_count=1)
        expect_step(
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
            requested_datasets=("analytics.function_types",),
        )

        if not ctx.run_id.startswith("analytics-"):
            pytest.fail(f"Run id should start with analytics- but was {ctx.run_id}")
        if ctx.kind != "analytics":
            pytest.fail(f"Expected kind analytics but got {ctx.kind}")
        if ctx.trigger != "http":
            pytest.fail(f"Expected trigger http but got {ctx.trigger}")
        if ctx.requested_datasets != ("analytics.function_types",):
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
        ctx = new_run_context(
            snapshot=sample_snapshot,
            kind="full",
            trigger="cli",
        )

        gateway.runs.start_run(
            ctx,
            pipeline_name="full:default",
        )

        now = datetime.now(tz=UTC)

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

        gateway.runs.record_step(
            PipelineStepRecord(
                run_id=ctx.run_id,
                module="analytics",
                stage="function",
                name="function_types",
                status="succeeded",
                started_at=now,
                completed_at=now,
                row_counts={"analytics.function_types": 10},
            ),
        )

        gateway.runs.complete_run(
            ctx.run_id,
            status="succeeded",
        )

        expect_run(
            gateway.runs.fetch_run(ctx.run_id),
            ExpectedRun(status="succeeded", kind="full"),
        )

        steps = expect_steps(
            gateway.runs.fetch_steps(ctx.run_id),
            expected_count=3,
            expected_modules={"ingestion", "graphs", "analytics"},
        )
        for step in steps:
            if step.module == "ingestion":
                expect_step(
                    step,
                    name="repo_scan",
                    status="succeeded",
                    row_counts={"core.modules": 10},
                )
            if step.module == "graphs":
                expect_step(
                    step,
                    name="call_graph_builder",
                    status="succeeded",
                    row_counts={"graph.call_graph_edges": 50},
                )
            if step.module == "analytics":
                expect_step(
                    step,
                    name="function_types",
                    status="succeeded",
                    row_counts={"analytics.function_types": 10},
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

        gateway.runs.record_step(
            PipelineStepRecord(
                run_id=ctx.run_id,
                module="analytics",
                stage="function",
                name="function_types",
                status="failed",
                started_at=now,
                completed_at=now,
                extra={"error": "Database connection failed"},
            ),
        )

        gateway.runs.complete_run(
            ctx.run_id,
            status="failed",
            error_summary="function_types plugin failed",
        )

        expect_run(
            gateway.runs.fetch_run(ctx.run_id),
            ExpectedRun(status="failed", error_summary="function_types plugin failed"),
        )

        steps = expect_steps(gateway.runs.fetch_steps(ctx.run_id), expected_count=1)
        expect_step(
            steps[0],
            status="failed",
            extra={"error": "Database connection failed"},
        )
