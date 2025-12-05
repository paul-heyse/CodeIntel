"""Integration tests for operation prerequisite orchestration.

These tests verify the ensure_prerequisites_for_operation function and
integration with the pipeline executor and run tracking.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.pipeline.planning.op_planner import (
    OperationPrereqOptions,
    build_pipeline_for_operation,
    ensure_prerequisites_for_operation,
)
from codeintel.pipeline.spec.model import FULL_PIPELINE, NOOP_PIPELINE
from codeintel.core.execution import TriggerKind
from tests._helpers.gateway import open_ingestion_gateway_with_macros
from tests._helpers.orchestration.tooling import make_tools_config

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# Type alias for the prereq options builder callable
PrereqOptionsBuilder = Callable[[], OperationPrereqOptions]


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

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    SnapshotRef
        Sample snapshot reference for integration tests.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    return SnapshotRef(
        repo="test/repo",
        commit="deadbeef",
        repo_root=repo_root,
    )


@pytest.fixture
def build_paths(tmp_path: Path) -> BuildPaths:
    """Create build paths for testing.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Returns
    -------
    BuildPaths
        Build paths configuration for tests.
    """
    return BuildPaths.from_repo_root(tmp_path / "repo")


@pytest.fixture
def tools_config() -> ToolsConfig:
    """Create tools configuration for testing.

    Returns
    -------
    ToolsConfig
        Tools configuration with default values.
    """
    return make_tools_config()


@pytest.fixture
def prereq_options_builder(
    gateway: StorageGateway,
    sample_snapshot: SnapshotRef,
    build_paths: BuildPaths,
    tools_config: ToolsConfig,
) -> Callable[..., OperationPrereqOptions]:
    """Build an OperationPrereqOptions instance with common defaults.

    Returns
    -------
    Callable[..., OperationPrereqOptions]
        Factory function that creates OperationPrereqOptions with injected
        fixtures. Accepts optional include_analytics and trigger kwargs.
    """

    def _build(
        *,
        include_analytics: bool = False,
        trigger: TriggerKind = "api",
    ) -> OperationPrereqOptions:
        return OperationPrereqOptions(
            snapshot=sample_snapshot,
            paths=build_paths,
            gateway=gateway,
            tools=tools_config,
            include_analytics=include_analytics,
            trigger=trigger,
        )

    return _build


class TestBuildPipelineForOperationIntegration:
    """Integration tests for build_pipeline_for_operation."""

    @staticmethod
    def test_function_summary_prereqs_requires_full(
        sample_snapshot: SnapshotRef,
    ) -> None:
        """function.summary operation requires full pipeline due to callgraph."""
        spec = build_pipeline_for_operation("function.summary", sample_snapshot)
        assert spec.id == FULL_PIPELINE.id

    @staticmethod
    def test_datasets_list_prereqs_is_noop(
        sample_snapshot: SnapshotRef,
    ) -> None:
        """datasets.list operation requires no prerequisites."""
        spec = build_pipeline_for_operation(
            "datasets.list",
            sample_snapshot,
            include_analytics=False,
        )
        assert spec.id == NOOP_PIPELINE.id
        assert spec.stages == ()

    @staticmethod
    def test_health_status_prereqs_is_noop(
        sample_snapshot: SnapshotRef,
    ) -> None:
        """health.status operation requires no prerequisites."""
        spec = build_pipeline_for_operation(
            "health.status",
            sample_snapshot,
            include_analytics=False,
        )
        assert spec.id == NOOP_PIPELINE.id


class TestEnsurePrerequisitesForOperationNoop:
    """Test ensure_prerequisites_for_operation with NOOP operations."""

    @staticmethod
    def test_datasets_list_prereqs_is_noop_succeeds(
        prereq_options_builder: Callable[..., OperationPrereqOptions],
    ) -> None:
        """NOOP operation should succeed with kind=op_prereqs."""
        run = ensure_prerequisites_for_operation(
            op_id="datasets.list",
            options=prereq_options_builder(include_analytics=False, trigger="api"),
        )

        assert run.status == "succeeded"
        assert run.kind == "op_prereqs"
        assert run.pipeline_name == "noop"

    @staticmethod
    def test_noop_records_no_steps(
        gateway: StorageGateway,
        prereq_options_builder: Callable[..., OperationPrereqOptions],
    ) -> None:
        """NOOP operation should record no pipeline steps."""
        run = ensure_prerequisites_for_operation(
            op_id="datasets.list",
            options=prereq_options_builder(include_analytics=False, trigger="api"),
        )

        steps = gateway.runs.fetch_steps(run.run_id)
        assert steps == []

    @staticmethod
    def test_health_status_prereqs_succeeds(
        prereq_options_builder: Callable[..., OperationPrereqOptions],
    ) -> None:
        """health.status NOOP operation should succeed."""
        run = ensure_prerequisites_for_operation(
            op_id="health.status",
            options=prereq_options_builder(include_analytics=False, trigger="http"),
        )

        assert run.status == "succeeded"


class TestRunTracking:
    """Test run tracking integration for operation prerequisites."""

    @staticmethod
    def test_run_record_created_for_noop(
        gateway: StorageGateway,
        sample_snapshot: SnapshotRef,
        prereq_options_builder: Callable[..., OperationPrereqOptions],
    ) -> None:
        """Run record should be created with kind=op_prereqs."""
        run = ensure_prerequisites_for_operation(
            op_id="datasets.list",
            options=prereq_options_builder(include_analytics=False, trigger="mcp"),
        )

        # Fetch run from database to verify persistence
        fetched_run = gateway.runs.fetch_run(run.run_id)
        assert fetched_run is not None
        assert fetched_run.run_id == run.run_id
        assert fetched_run.status == "succeeded"
        assert fetched_run.kind == "op_prereqs"
        assert fetched_run.trigger == "mcp"
        assert fetched_run.repo == sample_snapshot.repo
        assert fetched_run.commit == sample_snapshot.commit

    @staticmethod
    def test_run_has_completed_at_timestamp(
        prereq_options_builder: Callable[..., OperationPrereqOptions],
    ) -> None:
        """Completed run should have completed_at timestamp."""
        run = ensure_prerequisites_for_operation(
            op_id="datasets.list",
            options=prereq_options_builder(include_analytics=False, trigger="cli"),
        )

        assert run.completed_at is not None
        assert run.started_at is not None
        assert run.completed_at >= run.started_at


class TestOpPrereqsRunKind:
    """Test that ensure_prerequisites_for_operation uses op_prereqs RunKind."""

    @staticmethod
    def test_noop_operation_has_op_prereqs_kind(
        prereq_options_builder: Callable[..., OperationPrereqOptions],
    ) -> None:
        """NOOP operations should have kind=op_prereqs, not kind=full."""
        run = ensure_prerequisites_for_operation(
            op_id="datasets.list",
            options=prereq_options_builder(include_analytics=False, trigger="api"),
        )

        # Key assertion: even though NOOP_PIPELINE has empty stages (which
        # _infer_run_kind would classify as "full"), the run_kind_override
        # ensures we get "op_prereqs"
        assert run.kind == "op_prereqs"

    @staticmethod
    def test_run_id_prefix_matches_op_prereqs(
        prereq_options_builder: Callable[..., OperationPrereqOptions],
    ) -> None:
        """Run ID should be prefixed with op_prereqs kind."""
        run = ensure_prerequisites_for_operation(
            op_id="datasets.list",
            options=prereq_options_builder(include_analytics=False, trigger="cli"),
        )

        # Run IDs are prefixed with the kind
        assert run.run_id.startswith("op_prereqs-")


class TestErrorHandling:
    """Test error handling in operation prerequisite orchestration."""

    @staticmethod
    def test_unknown_operation_raises_value_error(
        prereq_options_builder: Callable[..., OperationPrereqOptions],
    ) -> None:
        """Unknown operation should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown operation id"):
            ensure_prerequisites_for_operation(
                op_id="nonexistent.operation",
                options=prereq_options_builder(),
            )


class TestTriggerKinds:
    """Test different trigger kinds are recorded correctly."""

    @staticmethod
    @pytest.mark.parametrize(
        "trigger_kind",
        ["cli", "http", "mcp", "api"],
    )
    def test_trigger_kind_recorded(
        trigger_kind: TriggerKind,
        prereq_options_builder: Callable[..., OperationPrereqOptions],
    ) -> None:
        """Different trigger kinds should be recorded in run records."""
        run = ensure_prerequisites_for_operation(
            op_id="datasets.list",
            options=prereq_options_builder(include_analytics=False, trigger=trigger_kind),
        )

        assert run.trigger == trigger_kind
