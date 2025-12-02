"""Integration tests for run_registry wiring via gateway."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.runtime import RunContext, new_run_context
from codeintel.storage.run_tracking import PipelineStepRecord
from tests._helpers.gateway import open_ingestion_gateway_with_macros

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


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
        assert rec is not None
        assert rec.run_id == "integ-123"
        assert rec.repo == "github.com/demo/repo"
        assert rec.status == "running"

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
        assert rec is not None
        assert rec.status == "succeeded"
        assert rec.completed_at is not None

    @staticmethod
    def test_record_steps_with_gateway(
        gateway: StorageGateway,
        sample_snapshot: SnapshotRef,
    ) -> None:
        """Verify step recording works with gateway.runs."""
        from datetime import UTC, datetime  # noqa: PLC0415

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
        assert len(steps) == 1
        assert steps[0].name == "call_graph_builder"
        assert steps[0].row_counts == {"graph.call_graph_edges": 100}


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

        assert ctx.run_id.startswith("ingest-")
        assert ctx.kind == "ingest"
        assert ctx.trigger == "cli"
        assert ctx.requested_operation == "scan"
        assert ctx.snapshot.repo == sample_snapshot.repo

    @staticmethod
    def test_new_run_context_for_analytics(sample_snapshot: SnapshotRef) -> None:
        """Verify new_run_context generates correct analytics context."""
        ctx = new_run_context(
            snapshot=sample_snapshot,
            kind="analytics",
            trigger="http",
            requested_datasets=("analytics.function_metrics",),
        )

        assert ctx.run_id.startswith("analytics-")
        assert ctx.kind == "analytics"
        assert ctx.trigger == "http"
        assert ctx.requested_datasets == ("analytics.function_metrics",)

    @staticmethod
    def test_new_run_context_for_graphs(sample_snapshot: SnapshotRef) -> None:
        """Verify new_run_context generates correct graphs context."""
        ctx = new_run_context(
            snapshot=sample_snapshot,
            kind="graphs",
            trigger="mcp",
        )

        assert ctx.run_id.startswith("graphs-")
        assert ctx.kind == "graphs"
        assert ctx.trigger == "mcp"


class TestFullRunLifecycle:
    """Test complete run lifecycle scenarios."""

    @staticmethod
    def test_multi_engine_run_lifecycle(
        gateway: StorageGateway,
        sample_snapshot: SnapshotRef,
    ) -> None:
        """Verify a full run lifecycle across multiple modules."""
        from datetime import UTC, datetime  # noqa: PLC0415

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
        rec = gateway.runs.fetch_run(ctx.run_id)
        assert rec is not None
        assert rec.status == "succeeded"
        assert rec.kind == "full"

        # Verify steps
        steps = gateway.runs.fetch_steps(ctx.run_id)
        expected_module_count = 3
        assert len(steps) == expected_module_count

        modules = {s.module for s in steps}
        assert modules == {"ingestion", "graphs", "analytics"}

    @staticmethod
    def test_failed_run_lifecycle(
        gateway: StorageGateway,
        sample_snapshot: SnapshotRef,
    ) -> None:
        """Verify failed run is recorded correctly."""
        from datetime import UTC, datetime  # noqa: PLC0415

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
        rec = gateway.runs.fetch_run(ctx.run_id)
        assert rec is not None
        assert rec.status == "failed"
        assert rec.error_summary == "function_metrics plugin failed"

        steps = gateway.runs.fetch_steps(ctx.run_id)
        assert len(steps) == 1
        assert steps[0].status == "failed"
        assert steps[0].extra == {"error": "Database connection failed"}
