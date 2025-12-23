"""Schema diffing utilities for detecting drift and breaking changes.

This module provides structured comparison of TableSchema instances,
classifying changes as breaking or non-breaking for CI gate enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from codeintel.build.schemas.manifest import SchemaManifest
    from codeintel.core.schemas.primitives import TableSchema


ChangeType = Literal["added", "removed", "type_changed", "nullable_changed"]


@dataclass(frozen=True)
class ColumnDiff:
    """Represents a single column-level change.

    Parameters
    ----------
    column_name
        Name of the affected column.
    change_type
        Category of change detected.
    old_value
        Previous value (type string or nullable bool), None for additions.
    new_value
        New value (type string or nullable bool), None for removals.
    """

    column_name: str
    change_type: ChangeType
    old_value: str | bool | None = None
    new_value: str | bool | None = None

    @property
    def is_breaking(self) -> bool:
        """Determine if this column change is breaking.

        Returns
        -------
        bool
            True if this change would break existing consumers.
        """
        if self.change_type == "removed":
            return True
        if self.change_type == "type_changed":
            return True
        if self.change_type == "nullable_changed":
            # nullable=True -> nullable=False is breaking
            return self.old_value is True and self.new_value is False
        return False


@dataclass(frozen=True)
class SchemaDiff:
    """Structured diff between expected and actual table schemas.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    added_columns
        Column names that were added.
    removed_columns
        Column names that were removed (BREAKING).
    type_changes
        Tuples of (column_name, old_type, new_type) (BREAKING).
    nullable_changes
        Tuples of (column_name, old_nullable, new_nullable).
    """

    table_key: str
    added_columns: tuple[str, ...]
    removed_columns: tuple[str, ...]
    type_changes: tuple[tuple[str, str, str], ...]
    nullable_changes: tuple[tuple[str, bool, bool], ...]

    @property
    def has_breaking_changes(self) -> bool:
        """Check if any changes are breaking.

        Breaking changes:
        - Column removed
        - Type changed (any type change)
        - Nullable changed from True to False

        Returns
        -------
        bool
            True if this diff contains breaking changes.
        """
        if self.removed_columns:
            return True
        if self.type_changes:
            return True
        # Check for nullable -> non-nullable changes
        for _, old_nullable, new_nullable in self.nullable_changes:
            if old_nullable and not new_nullable:
                return True
        return False

    @property
    def has_any_changes(self) -> bool:
        """Check if there are any changes at all.

        Returns
        -------
        bool
            True if any changes were detected.
        """
        return bool(
            self.added_columns or self.removed_columns or self.type_changes or self.nullable_changes
        )

    @property
    def breaking_change_count(self) -> int:
        """Count the number of breaking changes.

        Returns
        -------
        int
            Total number of breaking changes.
        """
        count = len(self.removed_columns) + len(self.type_changes)
        for _, old_nullable, new_nullable in self.nullable_changes:
            if old_nullable and not new_nullable:
                count += 1
        return count

    def format_changes(self) -> list[str]:
        """Format changes as human-readable lines.

        Returns
        -------
        list[str]
            Formatted change descriptions, one per line.
        """
        lines: list[str] = [f"  [BREAKING] Column removed: {col}" for col in self.removed_columns]

        lines.extend(
            f"  [BREAKING] Type changed: {col} ({old_type} -> {new_type})"
            for col, old_type, new_type in self.type_changes
        )

        for col, old_nullable, new_nullable in self.nullable_changes:
            if old_nullable and not new_nullable:
                lines.append(f"  [BREAKING] Nullable changed: {col} (true -> false)")
            else:
                lines.append(
                    f"  Nullable changed: {col} ({str(old_nullable).lower()} -> "
                    f"{str(new_nullable).lower()})"
                )

        lines.extend(f"  Column added: {col}" for col in self.added_columns)

        return lines


@dataclass(frozen=True)
class ManifestDiffResult:
    """Result of comparing two schema manifests.

    Parameters
    ----------
    diffs
        Per-table schema diffs (only tables with changes).
    added_tables
        Table keys that exist in actual but not expected.
    removed_tables
        Table keys that exist in expected but not actual (BREAKING).
    view_diffs
        Per-view schema diffs (v2, only views with changes).
    added_views
        View keys that exist in actual but not expected (v2).
    removed_views
        View keys that exist in expected but not actual (v2, BREAKING).
    added_artifacts
        Artifact filenames that exist in actual but not expected (v2).
    removed_artifacts
        Artifact filenames that exist in expected but not actual (BREAKING).
    """

    diffs: tuple[SchemaDiff, ...]
    added_tables: tuple[str, ...]
    removed_tables: tuple[str, ...]
    view_diffs: tuple[SchemaDiff, ...] = ()
    added_views: tuple[str, ...] = ()
    removed_views: tuple[str, ...] = ()
    added_artifacts: tuple[str, ...] = ()
    removed_artifacts: tuple[str, ...] = ()

    @property
    def has_breaking_changes(self) -> bool:
        """Check if any changes are breaking.

        Returns
        -------
        bool
            True if any breaking changes were detected.
        """
        if self.removed_tables or self.removed_views or self.removed_artifacts:
            return True
        if any(diff.has_breaking_changes for diff in self.diffs):
            return True
        return any(diff.has_breaking_changes for diff in self.view_diffs)

    @property
    def has_any_changes(self) -> bool:
        """Check if there are any changes.

        Returns
        -------
        bool
            True if any changes were detected.
        """
        return bool(
            self.diffs
            or self.added_tables
            or self.removed_tables
            or self.view_diffs
            or self.added_views
            or self.removed_views
            or self.added_artifacts
            or self.removed_artifacts
        )

    @property
    def breaking_change_count(self) -> int:
        """Count total breaking changes.

        Returns
        -------
        int
            Total number of breaking changes across all tables.
        """
        count = len(self.removed_tables) + len(self.removed_views) + len(self.removed_artifacts)
        for diff in self.diffs:
            count += diff.breaking_change_count
        for diff in self.view_diffs:
            count += diff.breaking_change_count
        return count

    @property
    def tables_with_drift(self) -> int:
        """Count tables that have drift.

        Returns
        -------
        int
            Number of tables with any changes.
        """
        return len(self.diffs) + len(self.added_tables) + len(self.removed_tables)

    @property
    def views_with_drift(self) -> int:
        """Count views that have drift.

        Returns
        -------
        int
            Number of views with any changes (v2 only).
        """
        return len(self.view_diffs) + len(self.added_views) + len(self.removed_views)

    @property
    def artifacts_with_drift(self) -> int:
        """Count artifacts that have drift.

        Returns
        -------
        int
            Number of artifacts with any changes (v2 only).
        """
        return len(self.added_artifacts) + len(self.removed_artifacts)

    def _format_table_changes(self) -> list[str]:
        """Format table-related changes (removals, diffs, additions).

        Returns
        -------
        list[str]
            Lines describing table changes.
        """
        lines: list[str] = []
        for table_key in self.removed_tables:
            lines.extend([f"{table_key}:", "  [BREAKING] Table removed", ""])
        for diff in self.diffs:
            if diff.has_any_changes:
                lines.append(f"{diff.table_key}:")
                lines.extend(diff.format_changes())
                lines.append("")
        for table_key in self.added_tables:
            lines.extend([f"{table_key}:", "  Table added", ""])
        return lines

    def _format_view_changes(self) -> list[str]:
        """Format view-related changes (removals, diffs, additions).

        Returns
        -------
        list[str]
            Lines describing view changes.
        """
        lines: list[str] = []
        for view_key in self.removed_views:
            lines.extend([f"{view_key} (view):", "  [BREAKING] View removed", ""])
        for diff in self.view_diffs:
            if diff.has_any_changes:
                lines.append(f"{diff.table_key} (view):")
                lines.extend(diff.format_changes())
                lines.append("")
        for view_key in self.added_views:
            lines.extend([f"{view_key} (view):", "  View added", ""])
        return lines

    def _format_artifact_changes(self) -> list[str]:
        """Format artifact-related changes (removals, additions).

        Returns
        -------
        list[str]
            Lines describing artifact changes.
        """
        lines: list[str] = []
        for artifact in self.removed_artifacts:
            lines.extend([f"{artifact} (artifact):", "  [BREAKING] Artifact removed", ""])
        for artifact in self.added_artifacts:
            lines.extend([f"{artifact} (artifact):", "  Artifact added", ""])
        return lines

    def _format_drift_summary(self) -> str:
        """Format the drift summary line.

        Returns
        -------
        str
            Summary line with drift counts.
        """
        summary_parts: list[str] = []
        if self.tables_with_drift > 0:
            summary_parts.append(f"{self.tables_with_drift} table(s)")
        if self.views_with_drift > 0:
            summary_parts.append(f"{self.views_with_drift} view(s)")
        if self.artifacts_with_drift > 0:
            summary_parts.append(f"{self.artifacts_with_drift} artifact(s)")
        drift_summary = ", ".join(summary_parts) if summary_parts else "0 items"
        return (
            f"Summary: {drift_summary} with drift, {self.breaking_change_count} breaking change(s)"
        )

    def format_summary(self) -> str:
        """Format a human-readable summary of all changes.

        Returns
        -------
        str
            Multi-line summary of drift detection results.
        """
        if not self.has_any_changes:
            return "No schema drift detected.\n"

        lines: list[str] = ["Schema drift detected:\n"]
        lines.extend(self._format_table_changes())
        lines.extend(self._format_view_changes())
        lines.extend(self._format_artifact_changes())
        lines.append(self._format_drift_summary())
        return "\n".join(lines)


def compute_schema_diff(expected: TableSchema, actual: TableSchema) -> SchemaDiff:
    """Compute detailed diff between expected and actual schemas.

    Parameters
    ----------
    expected
        The expected (baseline) schema.
    actual
        The actual (current) schema.

    Returns
    -------
    SchemaDiff
        Structured diff showing all changes between schemas.
    """
    expected_cols = {col.name: col for col in expected.columns}
    actual_cols = {col.name: col for col in actual.columns}

    expected_names = set(expected_cols.keys())
    actual_names = set(actual_cols.keys())

    added_columns = tuple(sorted(actual_names - expected_names))
    removed_columns = tuple(sorted(expected_names - actual_names))

    type_changes: list[tuple[str, str, str]] = []
    nullable_changes: list[tuple[str, bool, bool]] = []

    for name in sorted(expected_names & actual_names):
        exp_col = expected_cols[name]
        act_col = actual_cols[name]

        if exp_col.type != act_col.type:
            type_changes.append((name, exp_col.type, act_col.type))

        if exp_col.nullable != act_col.nullable:
            nullable_changes.append((name, exp_col.nullable, act_col.nullable))

    return SchemaDiff(
        table_key=expected.table_key,
        added_columns=added_columns,
        removed_columns=removed_columns,
        type_changes=tuple(type_changes),
        nullable_changes=tuple(nullable_changes),
    )


def _compare_schemas(
    expected_items: tuple[TableSchema, ...],
    actual_items: tuple[TableSchema, ...],
) -> tuple[tuple[SchemaDiff, ...], tuple[str, ...], tuple[str, ...]]:
    """Compare two collections of schemas and compute diffs.

    Parameters
    ----------
    expected_items
        Expected schemas.
    actual_items
        Actual schemas.

    Returns
    -------
    tuple[tuple[SchemaDiff, ...], tuple[str, ...], tuple[str, ...]]
        Tuple of (diffs, added_keys, removed_keys).
    """
    expected_map = {item.table_key: item for item in expected_items}
    actual_map = {item.table_key: item for item in actual_items}

    expected_keys = set(expected_map.keys())
    actual_keys = set(actual_map.keys())

    diffs: list[SchemaDiff] = []
    for key in sorted(expected_keys & actual_keys):
        diff = compute_schema_diff(expected_map[key], actual_map[key])
        if diff.has_any_changes:
            diffs.append(diff)

    added = tuple(sorted(actual_keys - expected_keys))
    removed = tuple(sorted(expected_keys - actual_keys))
    return tuple(diffs), added, removed


def compute_manifest_diffs(
    expected: SchemaManifest,
    actual: SchemaManifest,
) -> ManifestDiffResult:
    """Compute diffs for all table schemas in manifests.

    Parameters
    ----------
    expected
        The expected (baseline) manifest.
    actual
        The actual (current) manifest.

    Returns
    -------
    ManifestDiffResult
        Complete diff result including per-table diffs and table additions/removals,
        plus view and artifact diffs for v2 manifests.
    """
    table_diffs, added_tables, removed_tables = _compare_schemas(expected.tables, actual.tables)
    view_diffs, added_views, removed_views = _compare_schemas(expected.views, actual.views)

    expected_artifacts = {a.filename for a in expected.artifacts}
    actual_artifacts = {a.filename for a in actual.artifacts}

    return ManifestDiffResult(
        diffs=table_diffs,
        added_tables=added_tables,
        removed_tables=removed_tables,
        view_diffs=view_diffs,
        added_views=added_views,
        removed_views=removed_views,
        added_artifacts=tuple(sorted(actual_artifacts - expected_artifacts)),
        removed_artifacts=tuple(sorted(expected_artifacts - actual_artifacts)),
    )


__all__ = [
    "ChangeType",
    "ColumnDiff",
    "ManifestDiffResult",
    "SchemaDiff",
    "compute_manifest_diffs",
    "compute_schema_diff",
]
