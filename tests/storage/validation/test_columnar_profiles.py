"""Tests for columnar validation profiles."""

from __future__ import annotations

import pyarrow as pa
import pytest

from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.validation.pandera_schema import pandera_available
from codeintel.storage.validation.columnar import (
    ColumnarValidationContext,
    TableValidationError,
    validate_record_batch_reader,
    validate_table,
)


def test_schema_only_skips_data_checks() -> None:
    """Schema-only profile should skip data-level validation."""
    table_schema = TableSchema(
        schema="analytics",
        name="demo",
        columns=[
            Column("id", "BIGINT", nullable=False),
            Column("name", "VARCHAR"),
        ],
    )
    table = pa.table(
        {
            "id": [1, None],
            "name": ["a", "b"],
        }
    )

    schema_only_context = ColumnarValidationContext(
        table_schema=table_schema,
        validation_profile="schema-only",
    )
    validate_table(
        table_schema.table_key,
        table,
        context=schema_only_context,
    )

    light_context = ColumnarValidationContext(
        table_schema=table_schema,
        validation_profile="data-light",
    )
    with pytest.raises(TableValidationError):
        validate_table(
            table_schema.table_key,
            table,
            context=light_context,
        )


def test_data_strict_enforces_primary_key_uniqueness() -> None:
    """Data-strict profile should enforce primary key uniqueness via Pandera."""
    if not pandera_available():
        pytest.skip("Pandera + Polars required for uniqueness validation.")
    table_schema = TableSchema(
        schema="analytics",
        name="demo_unique",
        columns=[
            Column("id", "BIGINT", nullable=False),
            Column("name", "VARCHAR"),
        ],
        primary_key=("id",),
    )
    batch = pa.record_batch(
        [pa.array([1], type=pa.int64()), pa.array(["a"])],
        names=["id", "name"],
    )
    dup_batch = pa.record_batch(
        [pa.array([1], type=pa.int64()), pa.array(["b"])],
        names=["id", "name"],
    )

    reader = pa.RecordBatchReader.from_batches(batch.schema, [batch, dup_batch])
    light_context = ColumnarValidationContext(
        table_schema=table_schema,
        validation_profile="data-light",
    )
    light_reader = validate_record_batch_reader(
        table_schema.table_key,
        reader,
        context=light_context,
    )
    for _batch in light_reader:
        pass

    strict_reader = pa.RecordBatchReader.from_batches(batch.schema, [batch, dup_batch])
    strict_context = ColumnarValidationContext(
        table_schema=table_schema,
        validation_profile="data-strict",
    )
    with pytest.raises(TableValidationError):
        validate_record_batch_reader(
            table_schema.table_key,
            strict_reader,
            context=strict_context,
        )
