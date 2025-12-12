"""Tests for PR-11: DatasetRef v2 and ArtifactRef.

Validates DatasetRef includes repo/commit fields and ArtifactRef structure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.io.dataset_ref import DatasetRef, refs_from_target_result
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.config.primitives import SnapshotRef
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO


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
        with pytest.raises(AttributeError):
            ref.repo = "changed"  # type: ignore[misc]


class TestRefsFromTargetResult:
    """Tests for refs_from_target_result with snapshot support."""

    @staticmethod
    def test_refs_from_target_result_includes_snapshot() -> None:
        """Verify refs_from_target_result populates repo/commit from snapshot."""
        snapshot = SnapshotRef(
            repo=DEFAULT_REPO,
            commit=DEFAULT_COMMIT,
            repo_root=Path.cwd(),
        )

        refs = refs_from_target_result(
            target_name="function_metrics",
            table_keys=("analytics.function_metrics",),
            snapshot=snapshot,
        )

        ref = refs.get("analytics.function_metrics")
        if ref is None:
            pytest.fail("refs_from_target_result should return ref")
        if ref.repo != DEFAULT_REPO:
            pytest.fail(f"Expected repo='{DEFAULT_REPO}', got '{ref.repo}'")
        if ref.commit != DEFAULT_COMMIT:
            pytest.fail(f"Expected commit='{DEFAULT_COMMIT}', got '{ref.commit}'")

    @staticmethod
    def test_refs_from_target_result_without_snapshot() -> None:
        """Verify refs_from_target_result works without snapshot."""
        refs = refs_from_target_result(
            target_name="function_metrics",
            table_keys=("analytics.function_metrics",),
        )

        ref = refs.get("analytics.function_metrics")
        if ref is None:
            pytest.fail("refs_from_target_result should return ref")
        # Without snapshot, should have empty strings
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
        with pytest.raises(AttributeError):
            ref.name = "changed"  # type: ignore[misc]


class TestTargetRunRecordArtifacts:
    """Tests for TargetRunRecord artifacts field."""

    @staticmethod
    def test_target_run_record_has_artifacts_field() -> None:
        """Verify TargetRunRecord has artifacts field."""
        record = TargetRunRecord(
            target="scip",
            plugin_name="ingestion.scip",
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
            plugin_name="ingestion.scip",
            status="succeeded",
            input_hash="hash123",
            artifacts=(artifact,),
        )
        if len(record.artifacts) != 1:
            pytest.fail("artifacts should have 1 entry")
        if record.artifacts[0].name != "scip_index":
            pytest.fail("artifact not stored correctly")
