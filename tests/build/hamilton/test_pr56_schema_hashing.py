"""PR-56: Core schema hashing primitives."""

from __future__ import annotations

import pytest

from codeintel.core.schemas.hashing import canonical_type, schema_hash
from codeintel.core.schemas.primitives import Column, TableSchema


def test_canonical_type_normalizes_timestamp_with_time_zone() -> None:
    """canonical_type should normalize common TIMESTAMPTZ aliases."""
    if canonical_type("timestamp with time zone") != "TIMESTAMPTZ":
        pytest.fail("Expected TIMESTAMPTZ normalization for 'timestamp with time zone'")
    if canonical_type(" TIMESTAMP   WITH  TIME   ZONE ") != "TIMESTAMPTZ":
        pytest.fail("Expected TIMESTAMPTZ normalization for irregular whitespace")
    if canonical_type("timestamptz") != "TIMESTAMPTZ":
        pytest.fail("Expected TIMESTAMPTZ normalization for 'timestamptz'")


def test_schema_hash_is_stable_for_identical_schemas() -> None:
    """schema_hash should be deterministic across identical TableSchema objects."""
    schema_a = TableSchema(
        schema="analytics",
        name="example",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("loc", "INTEGER"),
        ],
    )
    schema_b = TableSchema(
        schema="analytics",
        name="example",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
            Column("loc", "INTEGER"),
        ],
    )
    if schema_hash(schema_a) != schema_hash(schema_b):
        pytest.fail("Expected identical schemas to produce identical schema hashes")


def test_schema_hash_changes_when_column_order_changes() -> None:
    """schema_hash should change when column order changes."""
    schema_a = TableSchema(
        schema="analytics",
        name="example",
        columns=[
            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),
        ],
    )
    schema_b = TableSchema(
        schema="analytics",
        name="example",
        columns=[
            Column("commit", "VARCHAR", nullable=False),
            Column("repo", "VARCHAR", nullable=False),
        ],
    )
    if schema_hash(schema_a) == schema_hash(schema_b):
        pytest.fail("Expected column order changes to affect schema hash")
