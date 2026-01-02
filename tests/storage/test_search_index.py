"""Tests for serving search index helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.storage.serving.search_index import build_search_documents_table
from tests._helpers.assertions import expect_true
from tests._helpers.fixtures.rows import insert_rows, module_row

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def test_search_documents_includes_modules(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify search documents include module entries."""
    row = module_row(
        path="pkg/mod.py",
        module="pkg.mod",
        repo="demo/repo",
        commit="c1",
    )
    insert_rows(fresh_gateway, [row])

    build_search_documents_table(fresh_gateway.con)

    rows = fresh_gateway.con.execute(
        """
        SELECT ref_goid_h128, name
        FROM docs.search_documents
        WHERE kind = 'module'
        """
    ).fetchall()

    expect_true(rows is not None and len(rows) > 0, message="module search docs should exist")
    expect_true(
        any(ref == "pkg/mod.py" and name == "pkg.mod" for ref, name in rows),
        message="module entries should populate search documents",
    )
