"""Tests for v2 manifest format (PR-72).

This module tests the extended SchemaManifest format that includes
views and export artifacts alongside tables.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from codeintel.build.schemas.diff import (
    ManifestDiffResult,
    compute_manifest_diffs,
)
from codeintel.build.schemas.manifest import (
    ExportArtifact,
    SchemaManifest,
)
from codeintel.core.schemas.primitives import Column, TableSchema
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_not_in,
    expect_true,
)

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import ColumnType


def _make_table(table_key: str, columns: list[tuple[str, ColumnType, bool]]) -> TableSchema:
    """Create a TableSchema for testing.

    Parameters
    ----------
    table_key
        Table key in "schema.name" format.
    columns
        List of (name, type, nullable) tuples.

    Returns
    -------
    TableSchema
        Test schema.
    """
    schema_name, table_name = table_key.split(".", maxsplit=1)
    return TableSchema(
        schema=schema_name,
        name=table_name,
        columns=[
            Column(name=name, type=col_type, nullable=nullable)
            for name, col_type, nullable in columns
        ],
    )


class TestExportArtifact:
    """Tests for ExportArtifact dataclass."""

    @staticmethod
    def test_create_parquet_artifact() -> None:
        """Test creating a Parquet artifact."""
        artifact = ExportArtifact(
            kind="parquet",
            filename="function_metrics.parquet",
            table_key="analytics.function_metrics",
        )
        expect_equal(artifact.kind, "parquet")
        expect_equal(artifact.filename, "function_metrics.parquet")
        expect_equal(artifact.table_key, "analytics.function_metrics")
        expect_true(artifact.description is None)

    @staticmethod
    def test_create_jsonl_artifact_with_description() -> None:
        """Test creating a JSONL artifact with description."""
        artifact = ExportArtifact(
            kind="jsonl",
            filename="modules.jsonl",
            table_key="ingestion.modules",
            description="All modules in the repository",
        )
        expect_equal(artifact.kind, "jsonl")
        expect_equal(artifact.description, "All modules in the repository")

    @staticmethod
    def test_artifact_without_table_key() -> None:
        """Test artifact not tied to a specific table."""
        artifact = ExportArtifact(
            kind="json",
            filename="summary.json",
            table_key=None,
        )
        expect_true(artifact.table_key is None)

    @staticmethod
    def test_artifact_to_json_obj() -> None:
        """Test JSON serialization of artifact."""
        artifact = ExportArtifact(
            kind="parquet",
            filename="test.parquet",
            table_key="analytics.test",
            description="Test artifact",
        )
        json_obj = artifact.to_json_obj()

        expect_equal(json_obj["kind"], "parquet")
        expect_equal(json_obj["filename"], "test.parquet")
        expect_equal(json_obj["table_key"], "analytics.test")
        expect_equal(json_obj["description"], "Test artifact")

    @staticmethod
    def test_artifact_to_json_obj_omits_none() -> None:
        """Test that None values are omitted from JSON."""
        artifact = ExportArtifact(
            kind="csv",
            filename="export.csv",
        )
        json_obj = artifact.to_json_obj()

        expect_in("kind", json_obj)
        expect_in("filename", json_obj)
        expect_not_in("table_key", json_obj)
        expect_not_in("description", json_obj)


@pytest.fixture
def sample_table() -> TableSchema:
    """Create a sample table schema.

    Returns
    -------
    TableSchema
        Sample table schema for testing.
    """
    return _make_table(
        "analytics.function_metrics",
        [
            ("function_goid_h128", "VARCHAR", False),
            ("loc", "INTEGER", True),
        ],
    )


@pytest.fixture
def sample_view() -> TableSchema:
    """Create a sample view schema.

    Returns
    -------
    TableSchema
        Sample view schema for testing.
    """
    return _make_table(
        "docs.v_function_summary",
        [
            ("function_name", "VARCHAR", True),
            ("total_loc", "INTEGER", True),
        ],
    )


@pytest.fixture
def sample_artifact() -> ExportArtifact:
    """Create a sample export artifact.

    Returns
    -------
    ExportArtifact
        Sample artifact for testing.
    """
    return ExportArtifact(
        kind="parquet",
        filename="function_metrics.parquet",
        table_key="analytics.function_metrics",
    )


class TestSchemaManifestV2:
    """Tests for v2 SchemaManifest format."""

    @staticmethod
    def test_v1_manifest_tables_only(sample_table: TableSchema) -> None:
        """Test v1 manifest with tables only."""
        manifest = SchemaManifest(
            version="v1",
            tables=(sample_table,),
        )
        expect_equal(manifest.version, "v1")
        expect_equal(len(manifest.tables), 1)
        expect_equal(len(manifest.views), 0)
        expect_equal(len(manifest.artifacts), 0)
        expect_false(manifest.is_v2)

    @staticmethod
    def test_v2_manifest_with_views(
        sample_table: TableSchema,
        sample_view: TableSchema,
    ) -> None:
        """Test v2 manifest with views."""
        manifest = SchemaManifest(
            version="v2",
            tables=(sample_table,),
            views=(sample_view,),
        )
        expect_equal(manifest.version, "v2")
        expect_equal(len(manifest.tables), 1)
        expect_equal(len(manifest.views), 1)
        expect_true(manifest.is_v2)

    @staticmethod
    def test_v2_manifest_with_artifacts(
        sample_table: TableSchema,
        sample_artifact: ExportArtifact,
    ) -> None:
        """Test v2 manifest with artifacts."""
        manifest = SchemaManifest(
            version="v2",
            tables=(sample_table,),
            artifacts=(sample_artifact,),
        )
        expect_equal(len(manifest.artifacts), 1)
        expect_true(manifest.is_v2)

    @staticmethod
    def test_v2_manifest_to_json_complete(
        sample_table: TableSchema,
        sample_view: TableSchema,
        sample_artifact: ExportArtifact,
    ) -> None:
        """Test complete v2 manifest JSON serialization."""
        manifest = SchemaManifest(
            version="v2",
            tables=(sample_table,),
            views=(sample_view,),
            artifacts=(sample_artifact,),
        )
        json_obj = manifest.to_json_obj()

        expect_equal(json_obj["version"], "v2")
        expect_in("tables", json_obj)
        expect_in("views", json_obj)
        expect_in("artifacts", json_obj)
        tables = json_obj["tables"]
        views = json_obj["views"]
        artifacts = json_obj["artifacts"]
        if not isinstance(tables, list):
            pytest.fail("Expected tables to be a list")
        if not isinstance(views, list):
            pytest.fail("Expected views to be a list")
        if not isinstance(artifacts, list):
            pytest.fail("Expected artifacts to be a list")
        expect_equal(len(tables), 1)
        expect_equal(len(views), 1)
        expect_equal(len(artifacts), 1)

    @staticmethod
    def test_v2_manifest_to_json_omits_empty(
        sample_table: TableSchema,
    ) -> None:
        """Test that empty views/artifacts are omitted from JSON."""
        manifest = SchemaManifest(
            version="v2",
            tables=(sample_table,),
        )
        json_obj = manifest.to_json_obj()

        expect_in("tables", json_obj)
        expect_not_in("views", json_obj)
        expect_not_in("artifacts", json_obj)

    @staticmethod
    def test_manifest_json_roundtrip(
        sample_table: TableSchema,
        sample_view: TableSchema,
        sample_artifact: ExportArtifact,
    ) -> None:
        """Test JSON serialization produces valid JSON."""
        manifest = SchemaManifest(
            version="v2",
            tables=(sample_table,),
            views=(sample_view,),
            artifacts=(sample_artifact,),
        )
        json_str = json.dumps(manifest.to_json_obj(), indent=2)
        parsed = json.loads(json_str)

        expect_equal(parsed["version"], "v2")
        expect_equal(len(parsed["tables"]), 1)
        expect_equal(len(parsed["views"]), 1)
        expect_equal(len(parsed["artifacts"]), 1)


@pytest.fixture
def base_table() -> TableSchema:
    """Create a base table for diffing.

    Returns
    -------
    TableSchema
        Base table schema for diff tests.
    """
    return _make_table(
        "analytics.metrics",
        [
            ("id", "INTEGER", False),
            ("value", "DOUBLE", True),
        ],
    )


@pytest.fixture
def base_view() -> TableSchema:
    """Create a base view for diffing.

    Returns
    -------
    TableSchema
        Base view schema for diff tests.
    """
    return _make_table(
        "docs.v_summary",
        [
            ("name", "VARCHAR", True),
        ],
    )


@pytest.fixture
def base_artifact() -> ExportArtifact:
    """Create a base artifact for diffing.

    Returns
    -------
    ExportArtifact
        Base artifact for diff tests.
    """
    return ExportArtifact(
        kind="parquet",
        filename="metrics.parquet",
        table_key="analytics.metrics",
    )


EXPECTED_BREAKING_COUNT_VIEW_AND_ARTIFACT = 2


class TestManifestDiffV2:
    """Tests for v2 manifest diff functionality."""

    @staticmethod
    def test_diff_no_changes(
        base_table: TableSchema,
        base_view: TableSchema,
        base_artifact: ExportArtifact,
    ) -> None:
        """Test diff with identical manifests."""
        manifest = SchemaManifest(
            version="v2",
            tables=(base_table,),
            views=(base_view,),
            artifacts=(base_artifact,),
        )
        result = compute_manifest_diffs(manifest, manifest)

        expect_false(result.has_any_changes)
        expect_false(result.has_breaking_changes)
        expect_equal(result.tables_with_drift, 0)
        expect_equal(result.views_with_drift, 0)
        expect_equal(result.artifacts_with_drift, 0)

    @staticmethod
    def test_diff_view_added(
        base_table: TableSchema,
        base_view: TableSchema,
    ) -> None:
        """Test diff when a view is added."""
        expected = SchemaManifest(version="v2", tables=(base_table,))
        actual = SchemaManifest(
            version="v2",
            tables=(base_table,),
            views=(base_view,),
        )
        result = compute_manifest_diffs(expected, actual)

        expect_true(result.has_any_changes)
        expect_false(result.has_breaking_changes)
        expect_equal(len(result.added_views), 1)
        expect_in(base_view.table_key, result.added_views)

    @staticmethod
    def test_diff_view_removed_is_breaking(
        base_table: TableSchema,
        base_view: TableSchema,
    ) -> None:
        """Test diff when a view is removed (breaking change)."""
        expected = SchemaManifest(
            version="v2",
            tables=(base_table,),
            views=(base_view,),
        )
        actual = SchemaManifest(version="v2", tables=(base_table,))
        result = compute_manifest_diffs(expected, actual)

        expect_true(result.has_any_changes)
        expect_true(result.has_breaking_changes)
        expect_equal(len(result.removed_views), 1)
        expect_in(base_view.table_key, result.removed_views)

    @staticmethod
    def test_diff_artifact_added(
        base_table: TableSchema,
        base_artifact: ExportArtifact,
    ) -> None:
        """Test diff when an artifact is added."""
        expected = SchemaManifest(version="v2", tables=(base_table,))
        actual = SchemaManifest(
            version="v2",
            tables=(base_table,),
            artifacts=(base_artifact,),
        )
        result = compute_manifest_diffs(expected, actual)

        expect_true(result.has_any_changes)
        expect_false(result.has_breaking_changes)
        expect_equal(len(result.added_artifacts), 1)
        expect_in(base_artifact.filename, result.added_artifacts)

    @staticmethod
    def test_diff_artifact_removed_is_breaking(
        base_table: TableSchema,
        base_artifact: ExportArtifact,
    ) -> None:
        """Test diff when an artifact is removed (breaking change)."""
        expected = SchemaManifest(
            version="v2",
            tables=(base_table,),
            artifacts=(base_artifact,),
        )
        actual = SchemaManifest(version="v2", tables=(base_table,))
        result = compute_manifest_diffs(expected, actual)

        expect_true(result.has_any_changes)
        expect_true(result.has_breaking_changes)
        expect_equal(len(result.removed_artifacts), 1)
        expect_in(base_artifact.filename, result.removed_artifacts)

    @staticmethod
    def test_diff_view_column_changed(
        base_table: TableSchema,
        base_view: TableSchema,
    ) -> None:
        """Test diff when view column is modified."""
        modified_view = _make_table(
            "docs.v_summary",
            [
                ("name", "INTEGER", True),  # Changed type
            ],
        )
        expected = SchemaManifest(
            version="v2",
            tables=(base_table,),
            views=(base_view,),
        )
        actual = SchemaManifest(
            version="v2",
            tables=(base_table,),
            views=(modified_view,),
        )
        result = compute_manifest_diffs(expected, actual)

        expect_true(result.has_any_changes)
        expect_true(result.has_breaking_changes)
        expect_equal(len(result.view_diffs), 1)
        expect_equal(result.view_diffs[0].table_key, "docs.v_summary")
        expect_equal(len(result.view_diffs[0].type_changes), 1)

    @staticmethod
    def test_breaking_change_count_includes_views_and_artifacts(
        base_table: TableSchema,
        base_view: TableSchema,
        base_artifact: ExportArtifact,
    ) -> None:
        """Test that breaking change count includes all types."""
        expected = SchemaManifest(
            version="v2",
            tables=(base_table,),
            views=(base_view,),
            artifacts=(base_artifact,),
        )
        actual = SchemaManifest(version="v2", tables=(base_table,))
        result = compute_manifest_diffs(expected, actual)

        # Should count both removed view and removed artifact
        expect_equal(result.breaking_change_count, EXPECTED_BREAKING_COUNT_VIEW_AND_ARTIFACT)
        expect_equal(result.views_with_drift, 1)
        expect_equal(result.artifacts_with_drift, 1)


class TestManifestDiffFormatSummary:
    """Tests for format_summary with v2 changes."""

    @staticmethod
    def test_format_summary_view_changes() -> None:
        """Test that format_summary includes view changes."""
        result = ManifestDiffResult(
            diffs=(),
            added_tables=(),
            removed_tables=(),
            view_diffs=(),
            added_views=("docs.v_new_view",),
            removed_views=("docs.v_old_view",),
            added_artifacts=(),
            removed_artifacts=(),
        )
        summary = result.format_summary()

        expect_in("docs.v_new_view (view)", summary)
        expect_in("View added", summary)
        expect_in("docs.v_old_view (view)", summary)
        expect_in("[BREAKING] View removed", summary)

    @staticmethod
    def test_format_summary_artifact_changes() -> None:
        """Test that format_summary includes artifact changes."""
        result = ManifestDiffResult(
            diffs=(),
            added_tables=(),
            removed_tables=(),
            view_diffs=(),
            added_views=(),
            removed_views=(),
            added_artifacts=("new.parquet",),
            removed_artifacts=("old.jsonl",),
        )
        summary = result.format_summary()

        expect_in("new.parquet (artifact)", summary)
        expect_in("Artifact added", summary)
        expect_in("old.jsonl (artifact)", summary)
        expect_in("[BREAKING] Artifact removed", summary)

    @staticmethod
    def test_format_summary_mixed_changes() -> None:
        """Test format_summary with all types of changes."""
        result = ManifestDiffResult(
            diffs=(),
            added_tables=("analytics.new_table",),
            removed_tables=(),
            view_diffs=(),
            added_views=("docs.v_new",),
            removed_views=(),
            added_artifacts=("export.parquet",),
            removed_artifacts=(),
        )
        summary = result.format_summary()

        expect_in("analytics.new_table", summary)
        expect_in("docs.v_new (view)", summary)
        expect_in("export.parquet (artifact)", summary)
        expect_in("1 table(s)", summary)
        expect_in("1 view(s)", summary)
        expect_in("1 artifact(s)", summary)


__all__ = [
    "TestExportArtifact",
    "TestManifestDiffFormatSummary",
    "TestManifestDiffV2",
    "TestSchemaManifestV2",
]
