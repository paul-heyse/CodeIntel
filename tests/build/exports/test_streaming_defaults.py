"""Tests for streaming export defaults."""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.exports.writers import write_jsonl_records
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from tests._helpers.assertions.expectation_assertions import expect_equal

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation
    from duckdb import Expression as DuckDBExpression


@dataclass
class _EmptyBatch:
    num_rows: int = 0

    def to_pydict(self) -> dict[str, list[object]]:
        return {} if self.num_rows == 0 else {"rows": []}


class _EmptyReader:
    schema: pa.Schema = pa.schema(())

    def __iter__(self) -> Iterator[_EmptyBatch]:
        return iter(())


class _BatchRecorder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def fetch_record_batch(self, rows_per_batch: int = DEFAULT_ARROW_BATCH_SIZE) -> _EmptyReader:
        self.batch_sizes.append(rows_per_batch)
        return _EmptyReader()

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
    handle = io.StringIO()
    count = write_jsonl_records(handle, rel=rel)
    expect_equal(count, 0)
    expect_equal(rel.batch_sizes, [DEFAULT_ARROW_BATCH_SIZE], label="batch size")
