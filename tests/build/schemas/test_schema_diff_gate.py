"""Schema diff gate helper tests."""

from __future__ import annotations

from tools.schema_diff import (
    SchemaBreakAllowlist,
    SchemaBreakApproval,
    collect_breaking_changes,
    find_unapproved_breaks,
)

from codeintel.build.schemas.diff import compute_manifest_diffs
from codeintel.core.manifests import SchemaManifest
from codeintel.core.schemas.primitives import Column, TableSchema


def test_collect_breaking_changes_detects_column_removal() -> None:
    """Detect column removal as a breaking schema change."""
    expected = _manifest([Column(name="a", type="INTEGER"), Column(name="b", type="VARCHAR")])
    actual = _manifest([Column(name="a", type="INTEGER")])

    diff = compute_manifest_diffs(expected, actual)
    changes = collect_breaking_changes(diff, expected_manifest=expected)

    assert any(
        change.kind == "table_column_removed"
        and change.table_key == "analytics.foo"
        and change.column_name == "b"
        and change.old_value == "VARCHAR"
        for change in changes
    )


def test_allowlist_approves_breaking_change() -> None:
    """Ensure allowlist approvals suppress breaking change failures."""
    expected = _manifest([Column(name="a", type="INTEGER"), Column(name="b", type="VARCHAR")])
    actual = _manifest([Column(name="a", type="INTEGER")])
    diff = compute_manifest_diffs(expected, actual)

    allowlist = SchemaBreakAllowlist(
        version=1,
        approvals=(
            SchemaBreakApproval(
                kind="table_column_removed",
                table_key="analytics.foo",
                column_name="b",
            ),
        ),
    )

    unapproved = find_unapproved_breaks(
        diff,
        expected_manifest=expected,
        allowlist=allowlist,
    )

    assert not unapproved


def _manifest(columns: list[Column]) -> SchemaManifest:
    schema = TableSchema(schema="analytics", name="foo", columns=columns)
    return SchemaManifest(version="v2", tables=(schema,))
