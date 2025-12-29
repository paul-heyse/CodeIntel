"""Tests for Arrow schema alignment helpers."""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

from codeintel.core.columnar.schema_alignment import align_reader_to_contract
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.schemas.primitives import Column, TableSchema


def _contract_schema() -> pa.Schema:
    table_schema = TableSchema(
        schema="analytics",
        name="alignment_contract",
        columns=[
            Column("id", "BIGINT", nullable=False),
            Column("name", "VARCHAR", nullable=True),
        ],
        primary_key=("id",),
    )
    return arrow_contract_for_table_schema(table_schema=table_schema)


def test_align_reader_retains_extras_and_preserves_columns() -> None:
    """Extra columns should be retained in _ci_extras with contract columns preserved."""
    contract_schema = _contract_schema()
    batch = pa.record_batch(
        [
            pa.array([1, 2]),
            pa.array(["alpha", "beta"]),
            pa.array(["x", None]),
        ],
        names=["id", "name", "extra"],
    )
    reader = pa.RecordBatchReader.from_batches(batch.schema, [batch])

    aligned_reader = align_reader_to_contract(
        reader,
        contract_schema,
        extras_policy="retain",
    )
    batches = list(aligned_reader)
    if len(batches) != 1:
        pytest.fail(f"Expected 1 batch, got {len(batches)}")

    aligned = batches[0]
    expected_names = ["id", "name", "_ci_extras"]
    if aligned.schema.names != expected_names:
        pytest.fail(f"Unexpected aligned schema names: {aligned.schema.names}")
    if aligned.column(0).to_pylist() != [1, 2]:
        pytest.fail("Aligned id column did not preserve values")
    if aligned.column(1).to_pylist() != ["alpha", "beta"]:
        pytest.fail("Aligned name column did not preserve values")

    extras = aligned.column(2).to_pylist()
    parsed = [json.loads(value) if value is not None else None for value in extras]
    if parsed[0] != {"extra": "x"}:
        pytest.fail(f"Unexpected extras payload: {parsed[0]}")
    if parsed[1] is not None:
        pytest.fail("Expected None extras payload when row has no extra values")


def test_align_reader_rejects_extras_policy() -> None:
    """Reject policy should raise when extra columns are present."""
    contract_schema = _contract_schema()
    batch = pa.record_batch(
        [
            pa.array([1]),
            pa.array(["alpha"]),
            pa.array(["extra"]),
        ],
        names=["id", "name", "extra"],
    )
    reader = pa.RecordBatchReader.from_batches(batch.schema, [batch])

    with pytest.raises(ValueError, match="Unexpected columns"):
        align_reader_to_contract(
            reader,
            contract_schema,
            extras_policy="reject",
        )
