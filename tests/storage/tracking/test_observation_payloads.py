"""Tests for schema observation payloads."""

from __future__ import annotations

import pyarrow as pa
import pytest

from codeintel.build.schemas import observations

pytestmark = pytest.mark.no_runtime_env

ROW_COUNT = 3
MIN_VALUE = 1
MAX_VALUE = 2
DISTINCT_COUNT = 2
ICEBERG_SNAPSHOT_ID = 10


def test_schema_observation_includes_dataset_stats() -> None:
    """Ensure observation payload includes dataset-level stats metadata."""
    schema = pa.schema([("value", pa.int64())])
    batch = pa.record_batch([[1, 2, 3]], schema=schema)
    accumulator = observations.SchemaObservationAccumulator(table_key="analytics.demo")
    accumulator.observe_batch(batch)
    bundle = accumulator.finalize(arrow_schema=schema)

    dataset_stats = bundle.observation.dataset_stats
    assert dataset_stats is not None
    assert "row_count" in dataset_stats
    assert dataset_stats["row_count"] == ROW_COUNT
    assert "batch_count" in dataset_stats
    assert dataset_stats["batch_count"] == 1
    assert "total_bytes" in dataset_stats
    assert dataset_stats["total_bytes"] >= 0


def test_schema_observation_column_stats_are_populated() -> None:
    """Ensure observation column stats contain expected entries."""
    schema = pa.schema([("value", pa.int64())])
    values = pa.array([MIN_VALUE, MAX_VALUE, MAX_VALUE], type=pa.int64())
    batch = pa.record_batch([values], schema=schema)
    accumulator = observations.SchemaObservationAccumulator(table_key="analytics.demo")
    accumulator.observe_batch(batch)
    bundle = accumulator.finalize(arrow_schema=schema)

    column_stats = bundle.observation.column_stats
    assert column_stats is not None
    entry = column_stats["value"]
    assert "null_count" in entry
    assert entry["null_count"] == 0
    assert "non_null_count" in entry
    assert entry["non_null_count"] == ROW_COUNT
    assert "distinct_count_max" in entry
    assert entry["distinct_count_max"] == DISTINCT_COUNT
    assert "min" in entry
    assert entry["min"] == MIN_VALUE
    assert "max" in entry
    assert entry["max"] == MAX_VALUE


def test_schema_observation_includes_iceberg_stats() -> None:
    """Ensure observation payload includes Iceberg stats metadata."""
    schema = pa.schema([("value", pa.int64())])
    batch = pa.record_batch([[1, 2, 3]], schema=schema)
    accumulator = observations.SchemaObservationAccumulator(table_key="analytics.demo")
    accumulator.observe_batch(batch)
    inputs = observations.SchemaObservationInputs(
        iceberg_stats={"snapshot_id": ICEBERG_SNAPSHOT_ID, "total_records": ROW_COUNT},
    )
    bundle = accumulator.finalize(arrow_schema=schema, inputs=inputs)

    dataset_stats = bundle.observation.dataset_stats
    assert dataset_stats is not None
    assert "iceberg_stats" in dataset_stats
    iceberg_stats = dataset_stats["iceberg_stats"]
    assert "snapshot_id" in iceberg_stats
    assert iceberg_stats["snapshot_id"] == ICEBERG_SNAPSHOT_ID
    assert "total_records" in iceberg_stats
    assert iceberg_stats["total_records"] == ROW_COUNT
