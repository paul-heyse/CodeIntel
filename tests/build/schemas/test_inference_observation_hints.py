"""Tests for inference-time observation hint merging."""

from __future__ import annotations

import pyarrow as pa
import pytest

from codeintel.build.schemas.inference_service import (
    SchemaObservationContext,
    observe_schema_from_batches,
)
from codeintel.build.schemas.observations import ColumnHint, SchemaHints
from codeintel.core.schemas.primitives import Column, TableSchema

pytestmark = pytest.mark.no_runtime_env


def test_observe_schema_merges_hints_without_overriding_types() -> None:
    """Ensure hints are merged without changing observed types."""
    schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
    batch = pa.record_batch([pa.array([1, 2])], schema=schema)
    declared_schema = TableSchema(
        schema="analytics",
        name="demo",
        columns=[Column("id", "VARCHAR", nullable=True, description="declared id")],
        description="declared table",
    )
    hints = SchemaHints(
        description="hint table",
        columns={"id": ColumnHint(nullable=True, description="hint id")},
    )
    context = SchemaObservationContext(declared_schema=declared_schema, schema_hints=hints)
    bundle = observe_schema_from_batches(
        batches=[batch],
        schema=schema,
        table_key="analytics.demo",
        context=context,
    )

    column = bundle.table_schema.columns[0]
    assert column.type == "BIGINT"
    assert column.nullable is True
    assert column.description == "hint id"


def test_observe_schema_uses_observed_nullability() -> None:
    """Ensure observed nullability overrides Arrow schema defaults."""
    schema = pa.schema([pa.field("id", pa.int64(), nullable=True)])
    batch = pa.record_batch([pa.array([1, 2])], schema=schema)
    bundle = observe_schema_from_batches(
        batches=[batch],
        schema=schema,
        table_key="analytics.demo",
    )

    column = bundle.table_schema.columns[0]
    assert column.nullable is False
