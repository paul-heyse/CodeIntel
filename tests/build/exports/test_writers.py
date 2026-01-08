"""Tests for export writer helpers."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.exports.writers import maybe_dictionary_encode_table
from tests._helpers.assertions.expectation_assertions import expect_equal


def test_dictionary_encode_preserves_metadata_and_nullability() -> None:
    """_maybe_dictionary_encode_table should keep schema metadata and nullability."""
    field = pa.field(
        "name",
        pa.string(),
        nullable=False,
        metadata={b"field": b"meta"},
    )
    schema = pa.schema([field], metadata={b"schema": b"meta"})
    table = pa.Table.from_arrays([pa.array(["a", "b"])], schema=schema)

    encoded = maybe_dictionary_encode_table(table, ["name"])

    encoded_field = encoded.schema.field("name")
    expect_equal(encoded.schema.metadata, schema.metadata)
    expect_equal(encoded_field.metadata, field.metadata)
    expect_equal(encoded_field.nullable, field.nullable)
