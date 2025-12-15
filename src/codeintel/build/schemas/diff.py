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
            self.added_columns
            or self.removed_columns
            or self.type_changes
            or self.nullable_changes
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
        lines: list[str] = [
            f"  [BREAKING] Column removed: {col}" for col in self.removed_columns
        ]

        lines.extend(
            f"  [BREAKING] Type changed: {col} ({old_type} -> {new_type})"
            for col, old_type, new_type in self.type_changes
        )

        for col, old_nullable, new_nullable in self.nullable_changes:
            if old_nullable and not new_nullable:
                lines.append(
                    f"  [BREAKING] Nullable changed: {col} (true -> false)"
                )
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
    """

    diffs: tuple[SchemaDiff, ...]
    added_tables: tuple[str, ...]
    removed_tables: tuple[str, ...]

    @property
    def has_breaking_changes(self) -> bool:
        """Check if any changes are breaking.

        Returns
        -------
        bool
            True if any breaking changes were detected.
        """
        if self.removed_tables:
            return True
        return any(diff.has_breaking_changes for diff in self.diffs)

    @property
    def has_any_changes(self) -> bool:
        """Check if there are any changes.

        Returns
        -------
        bool
            True if any changes were detected.
        """
        return bool(
            self.diffs or self.added_tables or self.removed_tables
        )

    @property
    def breaking_change_count(self) -> int:
        """Count total breaking changes.

        Returns
        -------
        int
            Total number of breaking changes across all tables.
        """
        count = len(self.removed_tables)
        for diff in self.diffs:
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

        for table_key in self.removed_tables:
            lines.append(f"{table_key}:")
            lines.append("  [BREAKING] Table removed")
            lines.append("")

        for diff in self.diffs:
            if diff.has_any_changes:
                lines.append(f"{diff.table_key}:")
                lines.extend(diff.format_changes())
                lines.append("")

        for table_key in self.added_tables:
            lines.append(f"{table_key}:")
            lines.append("  Table added")
            lines.append("")

        lines.append(
            f"Summary: {self.tables_with_drift} table(s) with drift, "
            f"{self.breaking_change_count} breaking change(s)"
        )

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
        Complete diff result including per-table diffs and table additions/removals.
    """
    expected_tables = {table.table_key: table for table in expected.tables}
    actual_tables = {table.table_key: table for table in actual.tables}

    expected_keys = set(expected_tables.keys())
    actual_keys = set(actual_tables.keys())

    added_tables = tuple(sorted(actual_keys - expected_keys))
    removed_tables = tuple(sorted(expected_keys - actual_keys))

    diffs: list[SchemaDiff] = []
    for table_key in sorted(expected_keys & actual_keys):
        diff = compute_schema_diff(expected_tables[table_key], actual_tables[table_key])
        if diff.has_any_changes:
            diffs.append(diff)

    return ManifestDiffResult(
        diffs=tuple(diffs),
        added_tables=added_tables,
        removed_tables=removed_tables,
    )


__all__ = [
    "ChangeType",
    "ColumnDiff",
    "ManifestDiffResult",
    "SchemaDiff",
    "compute_manifest_diffs",
    "compute_schema_diff",
]
