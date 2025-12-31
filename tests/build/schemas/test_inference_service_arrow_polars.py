"""Inference service Arrow/Polars path tests."""

from __future__ import annotations

import polars as pl
import pyarrow as pa
import pytest

from codeintel.build.schemas.inference_service import table_schema_from_tabular


def test_inference_service_handles_arrow_table() -> None:
    """Ensure Arrow tables use the Arrow/Polars schema path."""
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("name", pa.string(), nullable=True),
        ]
    )
    table = pa.Table.from_arrays([[1], ["alpha"]], schema=schema)
    table_schema = table_schema_from_tabular(table, table_key="analytics.arrow_demo")
    actual = [(col.name, col.type, col.nullable) for col in table_schema.columns]
    expected = [("id", "BIGINT", False), ("name", "VARCHAR", True)]
    if actual != expected:
        pytest.fail(f"Arrow schema inference mismatch: {actual} != {expected}")


def test_inference_service_handles_record_batch_reader() -> None:
    """Ensure RecordBatchReader inputs are supported."""
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("name", pa.string(), nullable=True),
        ]
    )
    batch = pa.record_batch([[1], ["alpha"]], schema=schema)
    reader = pa.RecordBatchReader.from_batches(schema, [batch])
    table_schema = table_schema_from_tabular(reader, table_key="analytics.reader_demo")
    actual = [(col.name, col.type, col.nullable) for col in table_schema.columns]
    expected = [("id", "BIGINT", False), ("name", "VARCHAR", True)]
    if actual != expected:
        pytest.fail(f"RecordBatchReader schema inference mismatch: {actual} != {expected}")


def test_inference_service_handles_record_batch_iterable() -> None:
    """Ensure RecordBatch iterables are supported."""
    schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
    batches = [pa.record_batch([[1, 2]], schema=schema)]
    table_schema = table_schema_from_tabular(batches, table_key="analytics.batch_demo")
    actual = [(col.name, col.type, col.nullable) for col in table_schema.columns]
    expected = [("id", "BIGINT", False)]
    if actual != expected:
        pytest.fail(f"RecordBatch iterable inference mismatch: {actual} != {expected}")


def test_inference_service_handles_polars_lazyframe() -> None:
    """Ensure Polars LazyFrames use the Arrow/Polars schema path."""
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "name": ["alpha", "beta"],
            "active": [True, False],
        }
    ).lazy()
    table_schema = table_schema_from_tabular(frame, table_key="analytics.polars_demo")
    actual = [(col.name, col.type) for col in table_schema.columns]
    expected = [("id", "BIGINT"), ("name", "VARCHAR"), ("active", "BOOLEAN")]
    if actual != expected:
        pytest.fail(f"Polars schema inference mismatch: {actual} != {expected}")
