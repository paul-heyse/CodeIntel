"""Tests for SCIP row builder filters."""

from __future__ import annotations

from datetime import UTC, datetime

from codeintel.ingestion.ports.tools import ScipDocument, ScipOccurrence
from codeintel.ingestion.scip.models import ScipSymbolRelationship
from codeintel.ingestion.scip.rows import (
    ScipRowContext,
    build_occurrence_rows,
    build_symbol_relationship_rows,
)


def test_build_occurrence_rows_filters_references() -> None:
    """Reference-only occurrences are dropped when include_references is false."""
    expected_all = 4
    expected_filtered = 3
    created_at = datetime(2024, 1, 1, tzinfo=UTC)
    occurrences = (
        ScipOccurrence(
            symbol="sym_def",
            range_start_line=1,
            range_start_col=0,
            range_end_line=1,
            range_end_col=5,
            symbol_roles=1,
        ),
        ScipOccurrence(
            symbol="sym_ref",
            range_start_line=2,
            range_start_col=0,
            range_end_line=2,
            range_end_col=5,
            symbol_roles=2,
        ),
        ScipOccurrence(
            symbol="sym_both",
            range_start_line=3,
            range_start_col=0,
            range_end_line=3,
            range_end_col=5,
            symbol_roles=3,
        ),
        ScipOccurrence(
            symbol="sym_unknown",
            range_start_line=4,
            range_start_col=0,
            range_end_line=4,
            range_end_col=5,
            symbol_roles=0,
        ),
    )
    document = ScipDocument(relative_path="mod.py", symbols=(), occurrences=occurrences)
    context = ScipRowContext(
        repo="repo",
        commit="commit",
        created_at=created_at,
        include_references=True,
    )

    rows_all = build_occurrence_rows((document,), context)
    rows_filtered = build_occurrence_rows(
        (document,),
        ScipRowContext(
            repo="repo",
            commit="commit",
            created_at=created_at,
            include_references=False,
        ),
    )

    assert len(rows_all) == expected_all
    assert len(rows_filtered) == expected_filtered
    filtered_symbols = {row[3] for row in rows_filtered}
    assert "sym_ref" not in filtered_symbols
    assert {"sym_def", "sym_both", "sym_unknown"}.issubset(filtered_symbols)


def test_build_symbol_relationship_rows_filters_options() -> None:
    """Relationship rows honor include_references/include_implementations settings."""
    expected_all = 4
    expected_without_refs = 3
    expected_without_impl = 3
    expected_filtered = 2
    created_at = datetime(2024, 1, 1, tzinfo=UTC)
    relationships = (
        ScipSymbolRelationship(
            symbol="sym",
            related_symbol="ref",
            relationship_kind="reference",
        ),
        ScipSymbolRelationship(
            symbol="sym",
            related_symbol="impl",
            relationship_kind="implementation",
        ),
        ScipSymbolRelationship(
            symbol="sym",
            related_symbol="def",
            relationship_kind="definition",
        ),
        ScipSymbolRelationship(
            symbol="sym",
            related_symbol="type",
            relationship_kind="type_definition",
        ),
    )
    base_context = ScipRowContext(repo="repo", commit="commit", created_at=created_at)

    rows_all = build_symbol_relationship_rows(
        relationships,
        base_context,
    )
    rows_no_refs = build_symbol_relationship_rows(
        relationships,
        ScipRowContext(
            repo="repo",
            commit="commit",
            created_at=created_at,
            include_references=False,
        ),
    )
    rows_no_impl = build_symbol_relationship_rows(
        relationships,
        ScipRowContext(
            repo="repo",
            commit="commit",
            created_at=created_at,
            include_implementations=False,
        ),
    )
    rows_filtered = build_symbol_relationship_rows(
        relationships,
        ScipRowContext(
            repo="repo",
            commit="commit",
            created_at=created_at,
            include_references=False,
            include_implementations=False,
        ),
    )

    assert len(rows_all) == expected_all
    assert len(rows_no_refs) == expected_without_refs
    assert len(rows_no_impl) == expected_without_impl
    assert len(rows_filtered) == expected_filtered

    filtered_kinds = {row[4] for row in rows_filtered}
    assert filtered_kinds == {"definition", "type_definition"}
