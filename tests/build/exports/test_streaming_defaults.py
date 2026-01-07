"""Tests for streaming export defaults."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.exports.writers import write_jsonl_records
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from tests._helpers.assertions.expectation_assertions import expect_equal

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation
    from duckdb import Expression as DuckDBExpression

_EMPTY_SCHEMA = pa.schema(())


class _BatchRecorder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def fetch_record_batch(
        self, rows_per_batch: int = DEFAULT_ARROW_BATCH_SIZE
    ) -> pa.RecordBatchReader:
        self.batch_sizes.append(rows_per_batch)
        return pa.RecordBatchReader.from_batches(_EMPTY_SCHEMA, [])

    def aggregate(
        self,
        aggr_expr: DuckDBExpression | str,
        group_expr: DuckDBExpression | str = "",
    ) -> DuckDBPyRelation:
        raise NotImplementedError

    def fetchone(self) -> tuple[object, ...] | None:
        return (0,) if self.batch_sizes else None


def test_write_jsonl_records_uses_default_batch_size() -> None:
    """write_jsonl_records uses the canonical default batch size."""
    rel = _BatchRecorder()
    with tempfile.TemporaryDirectory() as tempdir:
        output_path = Path(tempdir) / "output.jsonl"
        count = write_jsonl_records(output_path, rel=rel)
    expect_equal(count, 0)
    expect_equal(rel.batch_sizes, [DEFAULT_ARROW_BATCH_SIZE], label="batch size")
