"""Tests for schema observation payloads."""

from __future__ import annotations

import pyarrow as pa
import pytest

from codeintel.build.schemas import observations
from codeintel.core.columnar.ipc import schema_to_ipc_payload
from codeintel.storage.tracking.schema_catalog_models import SchemaObservationRecord

pytestmark = pytest.mark.no_runtime_env

ROW_COUNT = 3
ROW_GROUPS = 1
MIN_VALUE = 1
MAX_VALUE = 2
DISTINCT_COUNT = 2


def _previous_observation(table_key: str, schema: pa.Schema) -> SchemaObservationRecord:
    return SchemaObservationRecord(
        table_key=table_key,
        schema_digest="digest",
        schema_hash="hash",
        arrow_schema_ipc_b64=schema_to_ipc_payload(schema),
    )


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


def test_schema_observation_derived_settings_include_tuning() -> None:
    """Ensure derived settings capture dictionary and row group hints."""
    schema = pa.schema([("label", pa.string())])
    values = pa.array(["a"] * 90 + ["b"] * 10, type=pa.string())
    batch = pa.record_batch([values], schema=schema)
    accumulator = observations.SchemaObservationAccumulator(table_key="analytics.demo")
    accumulator.observe_batch(batch)
    bundle = accumulator.finalize(arrow_schema=schema)

    derived = bundle.observation.derived_settings
    assert derived is not None
    dict_columns = derived.get("dictionary_encode_columns")
    assert dict_columns == ["label"]
    dict_max = derived.get("dictionary_max_cardinality")
    assert dict_max == DISTINCT_COUNT
    unify = derived.get("unify_dictionaries")
    assert unify is True
    assert "row_group_size" in derived
    assert "data_page_size" in derived


def test_schema_observation_drift_uses_previous_observation_for_extra_columns() -> None:
    """Ensure drift summary compares against the previous observation when present."""
    table_key = "analytics.demo"
    previous_schema = pa.schema([("alpha", pa.int64())])
    current_schema = pa.schema([("alpha", pa.int64()), ("beta", pa.int64())])
    previous = _previous_observation(table_key, previous_schema)
    batch = pa.record_batch([pa.array([1, 2]), pa.array([3, 4])], schema=current_schema)

    accumulator = observations.SchemaObservationAccumulator(table_key=table_key)
    accumulator.observe_batch(batch)
    inputs = observations.SchemaObservationInputs(previous_observation=previous)
    bundle = accumulator.finalize(arrow_schema=current_schema, inputs=inputs)

    drift = bundle.observation.drift_summary
    assert drift is not None
    assert drift["extra_columns"] == ["beta"]
    assert drift["baseline_kind"] == "previous_observation"


def test_schema_observation_drift_reports_missing_columns() -> None:
    """Ensure drift summary captures missing columns relative to the baseline."""
    table_key = "analytics.demo"
    previous_schema = pa.schema([("alpha", pa.int64()), ("beta", pa.int64())])
    current_schema = pa.schema([("alpha", pa.int64())])
    previous = _previous_observation(table_key, previous_schema)
    batch = pa.record_batch([pa.array([1, 2])], schema=current_schema)

    accumulator = observations.SchemaObservationAccumulator(table_key=table_key)
    accumulator.observe_batch(batch)
    inputs = observations.SchemaObservationInputs(previous_observation=previous)
    bundle = accumulator.finalize(arrow_schema=current_schema, inputs=inputs)

    drift = bundle.observation.drift_summary
    assert drift is not None
    assert drift["missing_columns"] == ["beta"]


def test_schema_observation_drift_reports_type_changes() -> None:
    """Ensure drift summary captures type changes relative to the baseline."""
    table_key = "analytics.demo"
    previous_schema = pa.schema([("alpha", pa.int64())])
    current_schema = pa.schema([("alpha", pa.string())])
    previous = _previous_observation(table_key, previous_schema)
    batch = pa.record_batch([pa.array(["x", "y"])], schema=current_schema)

    accumulator = observations.SchemaObservationAccumulator(table_key=table_key)
    accumulator.observe_batch(batch)
    inputs = observations.SchemaObservationInputs(previous_observation=previous)
    bundle = accumulator.finalize(arrow_schema=current_schema, inputs=inputs)

    drift = bundle.observation.drift_summary
    assert drift is not None
    type_changes = drift.get("type_changes")
    assert isinstance(type_changes, list)
    assert type_changes[0]["column"] == "alpha"
    assert type_changes[0]["declared"] != type_changes[0]["observed"]
