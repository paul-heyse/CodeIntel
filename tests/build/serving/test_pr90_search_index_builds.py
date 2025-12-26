"""Tests for PR-90: DuckDB FTS search index builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import pytest

from codeintel.storage.serving.search_index import build_search_documents_table, ensure_fts_index
from tests._helpers.assertions import assert_target_ok, expect_true
from tests._helpers.harnesses.serving_harness import ServingTargetHarness

if TYPE_CHECKING:
    from pathlib import Path


def test_pr90_ensure_fts_index_creates_schema(tmp_path: Path) -> None:
    """Ensure `ensure_fts_index()` creates the expected FTS schema when available."""
    db_path = tmp_path / "search.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        build_search_documents_table(con)
        con.execute(
            """
            INSERT INTO docs.search_documents(doc_id, kind, name, module, rel_path, text, ref_goid_h128, repo, commit)
            VALUES ('doc:1', 'docstring', 'demo', 'demo', 'demo.py', 'duckdb search', NULL, 'r', 'c')
            """
        )
        try:
            fts_schema = ensure_fts_index(con)
        except duckdb.Error as exc:
            pytest.skip(f"DuckDB FTS extension unavailable: {exc}")

        present = con.execute(
            """
            SELECT 1
            FROM information_schema.schemata
            WHERE schema_name = ?
            LIMIT 1
            """,
            [fts_schema],
        ).fetchone()
        expect_true(present is not None)
    finally:
        con.close()


def test_pr90_search_index_with_serving_harness(
    serving_target_harness: ServingTargetHarness,
) -> None:
    """Ensure search index schema can be created in a harness-backed gateway."""
    records = serving_target_harness.run_targets()
    assert_target_ok(records["serving_artifacts"])

    con = serving_target_harness.harness.ctx.gateway.con
    build_search_documents_table(con)
    con.execute(
        """
        INSERT INTO docs.search_documents(doc_id, kind, name, module, rel_path, text, ref_goid_h128, repo, commit)
        VALUES ('doc:1', 'docstring', 'demo', 'demo', 'demo.py', 'duckdb search', NULL, 'r', 'c')
        """
    )
    try:
        fts_schema = ensure_fts_index(con)
    except duckdb.Error as exc:
        pytest.skip(f"DuckDB FTS extension unavailable: {exc}")

    present = con.execute(
        """
        SELECT 1
        FROM information_schema.schemata
        WHERE schema_name = ?
        LIMIT 1
        """,
        [fts_schema],
    ).fetchone()
    expect_true(present is not None)
