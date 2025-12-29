"""Tests for schema observation payloads."""

from __future__ import annotations

import pyarrow as pa
import pytest

from codeintel.build.schemas import observations

pytestmark = pytest.mark.no_runtime_env

ROW_COUNT = 3
ROW_GROUPS = 1
MIN_VALUE = 1
MAX_VALUE = 2
DISTINCT_COUNT = 2


def test_schema_observation_includes_parquet_stats() -> None:
    """Ensure observation payload includes parquet stats metadata."""
    schema = pa.schema([("value", pa.int64())])
    batch = pa.record_batch([[1, 2, 3]], schema=schema)
    accumulator = observations.SchemaObservationAccumulator(table_key="analytics.demo")
    accumulator.observe_batch(batch)
    inputs = observations.SchemaObservationInputs(
        dataset_stats={"row_groups": ROW_GROUPS},
        manifest_row_count=ROW_COUNT,
    )
    bundle = accumulator.finalize(arrow_schema=schema, inputs=inputs)

    dataset_stats = bundle.observation.dataset_stats
    assert dataset_stats is not None
    assert "row_count" in dataset_stats
    assert dataset_stats["row_count"] == ROW_COUNT
    assert "manifest_row_count" in dataset_stats
    assert dataset_stats["manifest_row_count"] == ROW_COUNT
    assert "parquet_stats" in dataset_stats
    parquet_stats = dataset_stats["parquet_stats"]
    assert parquet_stats["row_groups"] == ROW_GROUPS


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
