"""PR-71: Schema drift detection and breaking change classification tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.build.schemas.diff import (
    ManifestDiffResult,
    SchemaDiff,
    compute_manifest_diffs,
    compute_schema_diff,
)
from codeintel.build.schemas.manifest import SchemaManifest
from codeintel.core.schemas.primitives import Column, TableSchema

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import ColumnType


def _make_schema(
    table_key: str,
    columns: list[tuple[str, ColumnType, bool]],
) -> TableSchema:
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


# Expected number of breaking changes in test_breaking_change_count
_EXPECTED_BREAKING_CHANGES = 3


class TestComputeSchemaDiff:
    """Tests for compute_schema_diff function."""

    @staticmethod
    def test_schema_diff_detects_column_addition() -> None:
        """Detect when a column is added to the schema."""
        expected = _make_schema(
            "analytics.test_table",
            [("col_a", "VARCHAR", False)],
        )
        actual = _make_schema(
            "analytics.test_table",
            [("col_a", "VARCHAR", False), ("col_b", "INTEGER", True)],
        )

        diff = compute_schema_diff(expected, actual)

        if diff.added_columns != ("col_b",):
            pytest.fail(f"Expected added_columns=('col_b',), got {diff.added_columns}")
        if diff.has_breaking_changes:
            pytest.fail("Column addition should not be a breaking change")

    @staticmethod
    def test_schema_diff_detects_column_removal() -> None:
        """Detect when a column is removed (BREAKING)."""
        expected = _make_schema(
            "analytics.test_table",
            [("col_a", "VARCHAR", False), ("col_b", "INTEGER", True)],
        )
        actual = _make_schema(
            "analytics.test_table",
            [("col_a", "VARCHAR", False)],
        )

        diff = compute_schema_diff(expected, actual)

        if diff.removed_columns != ("col_b",):
            pytest.fail(f"Expected removed_columns=('col_b',), got {diff.removed_columns}")
        if not diff.has_breaking_changes:
            pytest.fail("Column removal should be a breaking change")

    @staticmethod
    def test_schema_diff_detects_type_change() -> None:
        """Detect when a column type changes (BREAKING)."""
        expected = _make_schema(
            "analytics.test_table",
            [("score", "VARCHAR", False)],
        )
        actual = _make_schema(
            "analytics.test_table",
            [("score", "INTEGER", False)],
        )

        diff = compute_schema_diff(expected, actual)

        if len(diff.type_changes) != 1:
            pytest.fail(f"Expected 1 type change, got {len(diff.type_changes)}")
        col, old_type, new_type = diff.type_changes[0]
        if col != "score" or old_type != "VARCHAR" or new_type != "INTEGER":
            pytest.fail(f"Type change mismatch: {diff.type_changes[0]}")
        if not diff.has_breaking_changes:
            pytest.fail("Type change should be a breaking change")

    @staticmethod
    def test_schema_diff_detects_nullable_to_nonnull() -> None:
        """Detect when nullable changes from True to False (BREAKING)."""
        expected = _make_schema(
            "analytics.test_table",
            [("required_col", "VARCHAR", True)],  # nullable=True
        )
        actual = _make_schema(
            "analytics.test_table",
            [("required_col", "VARCHAR", False)],  # nullable=False
        )

        diff = compute_schema_diff(expected, actual)

        if len(diff.nullable_changes) != 1:
            pytest.fail(f"Expected 1 nullable change, got {len(diff.nullable_changes)}")
        col, old_nullable, new_nullable = diff.nullable_changes[0]
        if col != "required_col" or old_nullable is not True or new_nullable is not False:
            pytest.fail(f"Nullable change mismatch: {diff.nullable_changes[0]}")
        if not diff.has_breaking_changes:
            pytest.fail("nullable=True -> False should be a breaking change")

    @staticmethod
    def test_schema_diff_detects_nonnull_to_nullable() -> None:
        """Detect when nullable changes from False to True (non-breaking)."""
        expected = _make_schema(
            "analytics.test_table",
            [("optional_col", "VARCHAR", False)],  # nullable=False
        )
        actual = _make_schema(
            "analytics.test_table",
            [("optional_col", "VARCHAR", True)],  # nullable=True
        )

        diff = compute_schema_diff(expected, actual)

        if len(diff.nullable_changes) != 1:
            pytest.fail(f"Expected 1 nullable change, got {len(diff.nullable_changes)}")
        if diff.has_breaking_changes:
            pytest.fail("nullable=False -> True should NOT be a breaking change")

    @staticmethod
    def test_schema_diff_no_changes() -> None:
        """No changes when schemas are identical."""
        schema = _make_schema(
            "analytics.test_table",
            [("col_a", "VARCHAR", False), ("col_b", "INTEGER", True)],
        )

        diff = compute_schema_diff(schema, schema)

        if diff.has_any_changes:
            pytest.fail("Expected no changes for identical schemas")

    @staticmethod
    def test_breaking_change_count() -> None:
        """Count breaking changes correctly."""
        expected = _make_schema(
            "analytics.test_table",
            [
                ("removed_col", "VARCHAR", False),
                ("type_col", "VARCHAR", False),
                ("nullable_col", "INTEGER", True),
            ],
        )
        actual = _make_schema(
            "analytics.test_table",
            [
                # removed_col is gone (1 breaking)
                ("type_col", "INTEGER", False),  # type changed (1 breaking)
                ("nullable_col", "INTEGER", False),  # nullable changed (1 breaking)
            ],
        )

        diff = compute_schema_diff(expected, actual)

        if diff.breaking_change_count != _EXPECTED_BREAKING_CHANGES:
            pytest.fail(
                f"Expected {_EXPECTED_BREAKING_CHANGES} breaking changes, "
                f"got {diff.breaking_change_count}"
            )


class TestComputeManifestDiffs:
    """Tests for compute_manifest_diffs function."""

    @staticmethod
    def test_manifest_diff_detects_table_addition() -> None:
        """Detect when a table is added to the manifest."""
        expected = SchemaManifest(
            version="v1",
            tables=(_make_schema("analytics.table_a", [("col", "VARCHAR", False)]),),
        )
        actual = SchemaManifest(
            version="v1",
            tables=(
                _make_schema("analytics.table_a", [("col", "VARCHAR", False)]),
                _make_schema("analytics.table_b", [("col", "INTEGER", True)]),
            ),
        )

        result = compute_manifest_diffs(expected, actual)

        if result.added_tables != ("analytics.table_b",):
            pytest.fail(f"Expected added table, got {result.added_tables}")
        if result.has_breaking_changes:
            pytest.fail("Table addition should not be breaking")

    @staticmethod
    def test_manifest_diff_detects_table_removal() -> None:
        """Detect when a table is removed (BREAKING)."""
        expected = SchemaManifest(
            version="v1",
            tables=(
                _make_schema("analytics.table_a", [("col", "VARCHAR", False)]),
                _make_schema("analytics.table_b", [("col", "INTEGER", True)]),
            ),
        )
        actual = SchemaManifest(
            version="v1",
            tables=(_make_schema("analytics.table_a", [("col", "VARCHAR", False)]),),
        )

        result = compute_manifest_diffs(expected, actual)

        if result.removed_tables != ("analytics.table_b",):
            pytest.fail(f"Expected removed table, got {result.removed_tables}")
        if not result.has_breaking_changes:
            pytest.fail("Table removal should be breaking")

    @staticmethod
    def test_manifest_diff_tracks_per_table_changes() -> None:
        """Track schema changes within individual tables."""
        expected = SchemaManifest(
            version="v1",
            tables=(_make_schema("analytics.table_a", [("col", "VARCHAR", False)]),),
        )
        actual = SchemaManifest(
            version="v1",
            tables=(_make_schema("analytics.table_a", [("col", "INTEGER", False)]),),
        )

        result = compute_manifest_diffs(expected, actual)

        if len(result.diffs) != 1:
            pytest.fail(f"Expected 1 diff, got {len(result.diffs)}")
        if result.diffs[0].table_key != "analytics.table_a":
            pytest.fail(f"Wrong table in diff: {result.diffs[0].table_key}")
        if not result.has_breaking_changes:
            pytest.fail("Type change should be breaking")


class TestSchemaDiffFormatting:
    """Tests for SchemaDiff formatting methods."""

    @staticmethod
    def test_format_changes_shows_breaking_markers() -> None:
        """Format changes with [BREAKING] markers."""
        diff = SchemaDiff(
            table_key="analytics.test_table",
            added_columns=("new_col",),
            removed_columns=("old_col",),
            type_changes=(("score", "VARCHAR", "INTEGER"),),
            nullable_changes=(("required", True, False),),
        )

        lines = diff.format_changes()
        text = "\n".join(lines)

        if "[BREAKING] Column removed: old_col" not in text:
            pytest.fail("Missing breaking marker for removed column")
        if "[BREAKING] Type changed: score" not in text:
            pytest.fail("Missing breaking marker for type change")
        if "[BREAKING] Nullable changed: required" not in text:
            pytest.fail("Missing breaking marker for nullable change")
        if "Column added: new_col" not in text:
            pytest.fail("Missing non-breaking column addition")


class TestManifestDiffResultFormatting:
    """Tests for ManifestDiffResult formatting."""

    @staticmethod
    def test_format_summary_no_changes() -> None:
        """Format summary shows no drift when manifests match."""
        result = ManifestDiffResult(
            diffs=(),
            added_tables=(),
            removed_tables=(),
        )

        summary = result.format_summary()

        if "No schema drift detected" not in summary:
            pytest.fail(f"Expected 'No schema drift' message, got: {summary}")

    @staticmethod
    def test_format_summary_with_changes() -> None:
        """Format summary shows drift details."""
        diff = SchemaDiff(
            table_key="analytics.test_table",
            added_columns=(),
            removed_columns=("col",),
            type_changes=(),
            nullable_changes=(),
        )
        result = ManifestDiffResult(
            diffs=(diff,),
            added_tables=(),
            removed_tables=(),
        )

        summary = result.format_summary()

        if "Schema drift detected" not in summary:
            pytest.fail(f"Expected 'Schema drift detected', got: {summary}")
        if "analytics.test_table" not in summary:
            pytest.fail(f"Expected table key in summary, got: {summary}")
        if "1 table(s) with drift" not in summary:
            pytest.fail(f"Expected drift count in summary, got: {summary}")


class TestBreakingChangeClassification:
    """Verify the breaking change classification rules."""

    @staticmethod
    @pytest.mark.parametrize(
        ("change_type", "expected_breaking"),
        [
            ("column_removed", True),
            ("type_changed", True),
            ("nullable_true_to_false", True),
            ("column_added", False),
            ("nullable_false_to_true", False),
        ],
    )
    def test_classification_rules(change_type: str, *, expected_breaking: bool) -> None:
        """Verify each change type is classified correctly."""
        base_cols: list[tuple[str, ColumnType, bool]] = [("col_a", "VARCHAR", False)]
        modified_cols: list[tuple[str, ColumnType, bool]] = [("col_a", "VARCHAR", False)]

        if change_type == "column_removed":
            base_cols.append(("removed", "INTEGER", True))
        elif change_type == "type_changed":
            modified_cols = [("col_a", "INTEGER", False)]  # Changed type
        elif change_type == "nullable_true_to_false":
            base_cols = [("col_a", "VARCHAR", True)]
            modified_cols = [("col_a", "VARCHAR", False)]
        elif change_type == "column_added":
            modified_cols.append(("added", "INTEGER", True))
        elif change_type == "nullable_false_to_true":
            base_cols = [("col_a", "VARCHAR", False)]
            modified_cols = [("col_a", "VARCHAR", True)]

        expected = _make_schema("analytics.test", base_cols)
        actual = _make_schema("analytics.test", modified_cols)

        diff = compute_schema_diff(expected, actual)

        if diff.has_breaking_changes != expected_breaking:
            pytest.fail(
                f"Change type '{change_type}' should be "
                f"{'breaking' if expected_breaking else 'non-breaking'}, "
                f"but got {'breaking' if diff.has_breaking_changes else 'non-breaking'}"
            )
