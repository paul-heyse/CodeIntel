"""Tests for Arrow dataset stats and streaming writes."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from codeintel.core.datasets.arrow_store import ArrowDatasetWriteOptions, write_dataset
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


def test_write_dataset_from_reader_populates_stats(tmp_path: Path) -> None:
    """Streaming writes should populate manifest stats with row group metadata."""
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)

    table = pa.table({"value": [1, 2, 3]})
    reader = pa.RecordBatchReader.from_batches(
        table.schema,
        table.to_batches(max_chunksize=2),
    )

    manifest = write_dataset(
        dataset_root=dataset_root,
        table_key="demo.table",
        snapshot_id="snap-1",
        data=reader,
        options=ArrowDatasetWriteOptions(persist_manifest=False),
    )

    expect_equal(manifest.row_count, expected=3)

    stats = manifest.stats
    expect_true(stats is not None, message="Expected dataset stats to be populated")
    if stats is None:
        return

    row_groups = stats.get("row_groups")
    expect_true(
        isinstance(row_groups, int) and row_groups >= 1,
        message="Expected row_groups to be a positive integer",
    )

    total_bytes = stats.get("total_bytes")
    expect_true(
        isinstance(total_bytes, int) and total_bytes > 0,
        message="Expected total_bytes to be a positive integer",
    )

    min_max = stats.get("min_max")
    expect_true(isinstance(min_max, dict), message="Expected min_max stats to be present")
    if isinstance(min_max, dict):
        value_stats = min_max.get("value")
        expect_true(
            isinstance(value_stats, dict),
            message="Expected min_max stats for value column",
        )
        if isinstance(value_stats, dict):
            expect_equal(value_stats.get("min"), expected=1)
            expect_equal(value_stats.get("max"), expected=3)
