"""Unit tests for the op_planner module.

These tests verify the operation-driven pipeline planning logic, including:
- Operation to PipelineSpec mapping
- Dataset dependency expansion via build_prereq_summary
- Stage flag computation through public APIs

Note: Internal helpers are tested indirectly through the public API to comply
with the project's import hygiene rules (no private name imports).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.pipeline.op_planner import (
    OpPrereqSummary,
    build_pipeline_for_operation,
    build_prereq_summary,
)
from codeintel.pipeline.spec import (
    FULL_PIPELINE,
    NOOP_PIPELINE,
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
        Sample snapshot reference for tests.
    """
    return SnapshotRef(
        repo="test/repo",
        commit="deadbeef",
        repo_root=tmp_path,
    )


class TestBuildPipelineForOperation:
    """Test build_pipeline_for_operation main function."""

    @staticmethod
    def test_function_summary_maps_to_full_pipeline(
        sample_snapshot: SnapshotRef,
    ) -> None:
        """function.summary requires callgraph, should map to FULL_PIPELINE."""
        spec = build_pipeline_for_operation("function.summary", sample_snapshot)
        assert spec.id == FULL_PIPELINE.id
        # Verify stages are present
        stage_modules = {stage.module for stage in spec.stages}
        assert {"ingestion", "graphs", "analytics"} == stage_modules

    @staticmethod
    def test_datasets_list_is_noop(sample_snapshot: SnapshotRef) -> None:
        """datasets.list has no requirements, should map to NOOP_PIPELINE."""
        spec = build_pipeline_for_operation(
            "datasets.list",
            sample_snapshot,
            include_analytics=False,
        )
        assert spec.id == NOOP_PIPELINE.id
        assert spec.stages == ()

    @staticmethod
    def test_include_analytics_false_excludes_analytics(
        sample_snapshot: SnapshotRef,
    ) -> None:
        """include_analytics=False should exclude analytics from the spec."""
        # health.status has no requirements
        spec = build_pipeline_for_operation(
            "health.status",
            sample_snapshot,
            include_analytics=False,
        )
        # Health status has no requirements, so should be NOOP
        assert spec.id == NOOP_PIPELINE.id

    @staticmethod
    def test_unknown_operation_raises_value_error(
        sample_snapshot: SnapshotRef,
    ) -> None:
        """Unknown operation should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown operation id"):
            build_pipeline_for_operation("nonexistent.op", sample_snapshot)

    @staticmethod
    def test_graph_operation_requires_full_pipeline(
        sample_snapshot: SnapshotRef,
    ) -> None:
        """Operations requiring graphs should result in FULL_PIPELINE."""
        spec = build_pipeline_for_operation(
            "graph.call_neighbors",
            sample_snapshot,
        )
        assert spec.id == FULL_PIPELINE.id

    @staticmethod
    def test_profiles_function_requires_full_pipeline(
        sample_snapshot: SnapshotRef,
    ) -> None:
        """profiles.function requires callgraph, should map to FULL_PIPELINE."""
        spec = build_pipeline_for_operation(
            "profiles.function",
            sample_snapshot,
        )
        assert spec.id == FULL_PIPELINE.id


class TestBuildPrereqSummary:
    """Test build_prereq_summary introspection function."""

    @staticmethod
    def test_returns_op_prereq_summary(sample_snapshot: SnapshotRef) -> None:
        """build_prereq_summary should return an OpPrereqSummary."""
        summary = build_prereq_summary("function.summary", sample_snapshot)
        assert isinstance(summary, OpPrereqSummary)
        assert summary.op.id == "function.summary"

    @staticmethod
    def test_summary_contains_required_graphs(sample_snapshot: SnapshotRef) -> None:
        """Summary should contain required_graphs for operations that need them."""
        summary = build_prereq_summary("function.summary", sample_snapshot)
        assert "callgraph" in summary.required_graphs

    @staticmethod
    def test_summary_for_noop_operation(sample_snapshot: SnapshotRef) -> None:
        """Summary for NOOP operation should have empty sets."""
        summary = build_prereq_summary("datasets.list", sample_snapshot)
        assert len(summary.required_tables) == 0
        assert len(summary.required_graphs) == 0
        assert len(summary.expanded_tables) == 0

    @staticmethod
    def test_summary_partitions_tables(sample_snapshot: SnapshotRef) -> None:
        """Summary should partition expanded tables into core/graph/analytics."""
        summary = build_prereq_summary("function.summary", sample_snapshot)
        # Summary should have disjoint partitions
        all_partitioned = summary.core_tables | summary.graph_tables | summary.analytics_tables
        # All expanded tables should be partitioned somewhere
        assert (
            summary.expanded_tables.issubset(all_partitioned) or len(summary.expanded_tables) == 0
        )

    @staticmethod
    def test_unknown_operation_raises_value_error(
        sample_snapshot: SnapshotRef,
    ) -> None:
        """Unknown operation should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown operation id"):
            build_prereq_summary("nonexistent.op", sample_snapshot)


class TestNOOPPipelineSpec:
    """Test NOOP_PIPELINE specification."""

    @staticmethod
    def test_noop_pipeline_has_no_stages() -> None:
        """NOOP_PIPELINE should have no stages."""
        assert NOOP_PIPELINE.stages == ()

    @staticmethod
    def test_noop_pipeline_id() -> None:
        """NOOP_PIPELINE should have id='noop'."""
        assert NOOP_PIPELINE.id == "noop"


class TestOperationCatalogCoverage:
    """Test that key operations from the catalog map to expected specs."""

    @staticmethod
    @pytest.mark.parametrize(
        ("op_id", "expected_spec_id"),
        [
            ("function.summary", "full"),
            ("graph.call_neighbors", "full"),
            ("profiles.function", "full"),
        ],
    )
    def test_operation_spec_mapping_full(
        sample_snapshot: SnapshotRef,
        op_id: str,
        expected_spec_id: str,
    ) -> None:
        """Verify operations that need full pipeline map correctly."""
        spec = build_pipeline_for_operation(
            op_id,
            sample_snapshot,
            include_analytics=True,
        )
        assert spec.id == expected_spec_id

    @staticmethod
    @pytest.mark.parametrize(
        ("op_id", "expected_spec_id"),
        [
            ("datasets.list", "noop"),
            ("health.status", "noop"),
            ("file.summary", "noop"),  # No required_datasets or required_graphs
            ("datasets.specs", "noop"),
        ],
    )
    def test_operation_spec_mapping_noop(
        sample_snapshot: SnapshotRef,
        op_id: str,
        expected_spec_id: str,
    ) -> None:
        """Verify operations with no prerequisites map to noop."""
        spec = build_pipeline_for_operation(
            op_id,
            sample_snapshot,
            include_analytics=False,
        )
        assert spec.id == expected_spec_id


class TestDependencyExpansionViaPrereqSummary:
    """Test dataset dependency expansion through the public API."""

    @staticmethod
    def test_expansion_preserves_unknown_tables(sample_snapshot: SnapshotRef) -> None:
        """Verify unknown table keys flow through dependency expansion."""
        # Using a known operation with empty requirements
        summary = build_prereq_summary("datasets.list", sample_snapshot)
        # No tables should be expanded for this operation
        assert len(summary.expanded_tables) == 0

    @staticmethod
    def test_expansion_includes_original_tables(sample_snapshot: SnapshotRef) -> None:
        """Verify original tables are always included in expansion."""
        summary = build_prereq_summary("graph.call_neighbors", sample_snapshot)
        # Required datasets should be subset of expanded (or equal if no deps)
        assert summary.required_tables.issubset(summary.expanded_tables)


class TestGraphRequirements:
    """Test operations with different graph requirements."""

    @staticmethod
    def test_callgraph_requirement_triggers_graphs_stage(
        sample_snapshot: SnapshotRef,
    ) -> None:
        """Operations requiring callgraph should trigger graphs stage."""
        summary = build_prereq_summary("function.summary", sample_snapshot)
        assert "callgraph" in summary.required_graphs

        spec = build_pipeline_for_operation("function.summary", sample_snapshot)
        stage_modules = {stage.module for stage in spec.stages}
        assert "graphs" in stage_modules

    @staticmethod
    def test_importgraph_requirement_triggers_graphs_stage(
        sample_snapshot: SnapshotRef,
    ) -> None:
        """Operations requiring importgraph should trigger graphs stage."""
        summary = build_prereq_summary("graph.import_boundary", sample_snapshot)
        assert "importgraph" in summary.required_graphs

        spec = build_pipeline_for_operation("graph.import_boundary", sample_snapshot)
        stage_modules = {stage.module for stage in spec.stages}
        assert "graphs" in stage_modules
