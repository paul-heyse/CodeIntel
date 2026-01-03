"""Tests for export validation using the TableSchema contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq

from codeintel.build.exports.validation import validate_export_files
from tests._helpers.assertions.expectation_assertions import expect_equal
from tests._helpers.columnar_streams import table_for_rows
from tests._helpers.fixtures.rows import row_for

if TYPE_CHECKING:
    from pathlib import Path


def test_validate_export_files_uses_contract_schema(tmp_path: Path) -> None:
    """Parquet validation should succeed for a contract-aligned table."""
    table_key = "graph.call_graph_edges"
    row = row_for(
        table_key,
        repo="r",
        commit="c",
        caller_goid_h128=1,
        callee_goid_h128=2,
        callsite_path="a.py",
        callsite_line=10,
        callsite_col=5,
        language="python",
        kind="direct",
    )
    table = table_for_rows(table_key, [row])
    path = tmp_path / "edges.parquet"
    pq.write_table(table, path)

    result = validate_export_files(table_key, [path])
    expect_equal(result, expected=0)
