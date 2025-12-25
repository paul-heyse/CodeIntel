"""Tests for serving search index helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.storage.serving.search_index import build_search_documents_table
from tests._helpers.assertions import expect_true
from tests._helpers.fixtures.rows import function_metrics_row, insert_rows

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def test_search_documents_includes_function_metrics(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify search documents include function_metrics entries."""
    row = function_metrics_row(
        goid=123,
        rel_path="pkg/mod.py",
        qualname="hello",
        snapshot=("demo/repo", "c1"),
        metrics={"created_at": datetime(2024, 1, 1, tzinfo=UTC)},
    )
    insert_rows(fresh_gateway, [row])

    build_search_documents_table(fresh_gateway.con)

    rows = fresh_gateway.con.execute(
        """
        SELECT ref_goid_h128, name
        FROM docs.search_documents
        WHERE kind = 'function'
        """
    ).fetchall()

    expect_true(rows is not None and len(rows) > 0, message="function search docs should exist")
    expect_true(
        any(ref == "123" and name == "hello" for ref, name in rows),
        message="function_metrics entries should populate search documents",
    )
