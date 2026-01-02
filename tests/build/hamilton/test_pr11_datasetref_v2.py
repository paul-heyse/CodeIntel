"""Tests for PR-11: DatasetRef v2 and ArtifactRef.

Validates DatasetRef includes repo/commit fields and ArtifactRef structure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.io.dataset_ref import DatasetRef, refs_from_target_result
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions import assert_record_has_artifacts, assert_record_has_datasets
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT

EXPECTED_DUAL_OUTPUT_REFS = 2


class TestDatasetRefV2Fields:
    """Tests for DatasetRef v2 repo/commit fields."""

    @staticmethod
    def test_datasetref_has_repo_field() -> None:
        """Verify DatasetRef has repo field."""
        ref = DatasetRef(
            table_key="analytics.metrics",
            repo="org/repo",
            commit="abc123",
        )
        if ref.repo != "org/repo":
            pytest.fail(f"Expected repo='org/repo', got '{ref.repo}'")

    @staticmethod
    def test_datasetref_has_commit_field() -> None:
        """Verify DatasetRef has commit field."""
        ref = DatasetRef(
            table_key="analytics.metrics",
            repo="org/repo",
            commit="abc123",
        )
        if ref.commit != "abc123":
            pytest.fail(f"Expected commit='abc123', got '{ref.commit}'")

    @staticmethod
    def test_datasetref_defaults_to_empty_strings() -> None:
        """Verify DatasetRef defaults repo/commit to empty strings."""
        ref = DatasetRef(table_key="analytics.metrics")
        if ref.repo:
            pytest.fail(f"Expected repo='', got '{ref.repo}'")
        if ref.commit:
            pytest.fail(f"Expected commit='', got '{ref.commit}'")

    @staticmethod
    def test_datasetref_is_frozen() -> None:
        """Verify DatasetRef is immutable."""
        ref = DatasetRef(
            table_key="analytics.metrics",
            repo="org/repo",
            commit="abc123",
        )
        attr = "repo"
        with pytest.raises(AttributeError):
            setattr(ref, attr, "changed")


class TestRefsFromTargetResult:
    """Tests for refs_from_target_result with snapshot support."""

    @staticmethod
    def test_refs_from_target_result_includes_snapshot() -> None:
        """Verify refs_from_target_result populates repo/commit from snapshot."""
        snapshot = SnapshotRef(
            repo=DEFAULT_VARIANT.repo,
            commit=DEFAULT_VARIANT.commit,
            repo_root=Path.cwd(),
        )

        refs = refs_from_target_result(
            target_name="function_types",
            table_keys=("analytics.function_types",),
            snapshot=snapshot,
        )

        ref = refs.get("analytics.function_types")
        if ref is None:
            pytest.fail("refs_from_target_result should return ref")
        if ref.repo != DEFAULT_VARIANT.repo:
            pytest.fail(f"Expected repo='{DEFAULT_VARIANT.repo}', got '{ref.repo}'")
        if ref.commit != DEFAULT_VARIANT.commit:
            pytest.fail(f"Expected commit='{DEFAULT_VARIANT.commit}', got '{ref.commit}'")

    @staticmethod
    def test_refs_from_target_result_without_snapshot() -> None:
        """Verify refs_from_target_result works without snapshot."""
        refs = refs_from_target_result(
            target_name="function_types",
            table_keys=("analytics.function_types",),
        )

        ref = refs.get("analytics.function_types")
        if ref is None:
            pytest.fail("refs_from_target_result should return ref")

        if ref.repo:
            pytest.fail(f"Expected repo='', got '{ref.repo}'")


class TestArtifactRef:
    """Tests for ArtifactRef dataclass."""

    @staticmethod
    def test_artifactref_has_required_fields() -> None:
        """Verify ArtifactRef has all required fields."""
        ref = ArtifactRef(
            name="scip_index",
            artifact_type="index",
            repo="org/repo",
            commit="abc123",
        )
        if ref.name != "scip_index":
            pytest.fail("name not set correctly")
        if ref.artifact_type != "index":
            pytest.fail("artifact_type not set correctly")
        if ref.repo != "org/repo":
            pytest.fail("repo not set correctly")
        if ref.commit != "abc123":
            pytest.fail("commit not set correctly")

    @staticmethod
    def test_artifactref_has_metadata_field() -> None:
        """Verify ArtifactRef has optional metadata field."""
        ref = ArtifactRef(
            name="scip_index",
            artifact_type="index",
            repo="org/repo",
            commit="abc123",
            metadata={"path": "/path/to/index.scip"},
        )
        if not ref.metadata:
            pytest.fail("metadata should be set")
        if ref.metadata.get("path") != "/path/to/index.scip":
            pytest.fail("metadata path incorrect")

    @staticmethod
    def test_artifactref_is_frozen() -> None:
        """Verify ArtifactRef is immutable."""
        ref = ArtifactRef(
            name="scip_index",
            artifact_type="index",
            repo="org/repo",
            commit="abc123",
        )
        attr = "name"
        with pytest.raises(AttributeError):
            setattr(ref, attr, "changed")


class TestTargetRunRecordArtifacts:
    """Tests for TargetRunRecord artifacts field."""

    @staticmethod
    def test_target_run_record_has_artifacts_field() -> None:
        """Verify TargetRunRecord has artifacts field."""
        record = TargetRunRecord(
            target="scip",
            impl_kind="native",
            status="succeeded",
            input_hash="hash123",
        )
        if not hasattr(record, "artifacts"):
            pytest.fail("TargetRunRecord missing artifacts field")
        if record.artifacts != ():
            pytest.fail("Default artifacts should be empty tuple")

    @staticmethod
    def test_target_run_record_accepts_artifacts() -> None:
        """Verify TargetRunRecord accepts artifacts in constructor."""
        artifact = ArtifactRef(
            name="scip_index",
            artifact_type="index",
            repo="org/repo",
            commit="abc123",
        )
        record = TargetRunRecord(
            target="scip",
            impl_kind="native",
            status="succeeded",
            input_hash="hash123",
            artifacts=(artifact,),
        )
        if len(record.artifacts) != 1:
            pytest.fail("artifacts should have 1 entry")
        assert_record_has_artifacts(record, ["scip_index"])


class TestSkippedTargetsDatasetRef:
    """Tests for skipped targets still populating DatasetRef."""

    @staticmethod
    def test_skipped_target_has_datasets_field() -> None:
        """Verify TargetRunRecord for skipped targets has datasets."""
        record = TargetRunRecord(
            target="function_types",
            impl_kind="native",
            status="skipped",
            input_hash="hash123",
            datasets=(
                DatasetRef(
                    table_key="analytics.function_types",
                    repo="org/repo",
                    commit="abc123",
                ),
            ),
        )
        assert_record_has_datasets(record, ["analytics.function_types"])
        if len(record.datasets) != 1:
            pytest.fail("Expected 1 dataset")

    @staticmethod
    def test_skipped_target_datasets_have_lineage() -> None:
        """Verify skipped target DatasetRef has repo/commit for lineage."""
        record = TargetRunRecord(
            target="function_types",
            impl_kind="native",
            status="skipped",
            input_hash="hash123",
            datasets=(
                DatasetRef(
                    table_key="analytics.function_types",
                    repo="org/repo",
                    commit="abc123",
                ),
            ),
        )

        dataset = record.datasets[0]
        if not dataset.repo:
            pytest.fail("Skipped dataset should have repo")
        if not dataset.commit:
            pytest.fail("Skipped dataset should have commit")
        if dataset.repo != "org/repo":
            pytest.fail(f"Expected repo='org/repo', got '{dataset.repo}'")
        if dataset.commit != "abc123":
            pytest.fail(f"Expected commit='abc123', got '{dataset.commit}'")

    @staticmethod
    def test_refs_from_target_result_multiple_tables() -> None:
        """Verify refs_from_target_result handles multiple table_keys."""
        snapshot = SnapshotRef(
            repo=DEFAULT_VARIANT.repo,
            commit=DEFAULT_VARIANT.commit,
            repo_root=Path.cwd(),
        )

        refs = refs_from_target_result(
            target_name="dual_output",
            table_keys=("analytics.metrics_a", "analytics.metrics_b"),
            snapshot=snapshot,
        )

        if len(refs) != EXPECTED_DUAL_OUTPUT_REFS:
            pytest.fail(f"Expected {EXPECTED_DUAL_OUTPUT_REFS} refs, got {len(refs)}")
        if "analytics.metrics_a" not in refs:
            pytest.fail("Missing ref for metrics_a")
        if "analytics.metrics_b" not in refs:
            pytest.fail("Missing ref for metrics_b")

        for key, ref in refs.items():
            if ref.repo != DEFAULT_VARIANT.repo:
                pytest.fail(f"{key} should have repo from snapshot")
            if ref.commit != DEFAULT_VARIANT.commit:
                pytest.fail(f"{key} should have commit from snapshot")

    @staticmethod
    def test_dataset_ref_fields_accessible() -> None:
        """Verify DatasetRef fields are accessible for serialization."""
        ref = DatasetRef(
            table_key="analytics.metrics",
            repo="org/repo",
            commit="abc123",
        )

        if ref.table_key != "analytics.metrics":
            pytest.fail("table_key field not accessible")
        if ref.repo != "org/repo":
            pytest.fail("repo field not accessible")
        if ref.commit != "abc123":
            pytest.fail("commit field not accessible")
