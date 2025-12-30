"""Tests for export codecs registry."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal

import pyarrow as pa
import pytest

from codeintel.core.exports.codecs import (
    ExportCodecError,
    encode_batch,
    encode_ndjson_line,
    encode_reader,
    get_export_codec,
)

EXPECTED_DECIMAL = 42


def test_encode_ndjson_line_coerces_types() -> None:
    """Encode NDJSON lines with type coercion."""
    row = {
        "timestamp": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        "count": Decimal(str(EXPECTED_DECIMAL)),
        "tags": ["a", "b"],
    }
    line = encode_ndjson_line(row)
    assert line.endswith(b"\n")
    payload = json.loads(line.decode("utf-8"))
    assert payload["timestamp"].endswith("Z")
    assert payload["count"] == EXPECTED_DECIMAL
    assert payload["tags"] == ["a", "b"]


def test_encode_batch_and_reader_emit_ndjson() -> None:
    """Encode batches and readers into NDJSON lines."""
    batch = pa.record_batch(
        [pa.array([1, 2]), pa.array(["a", "b"])],
        names=["id", "name"],
    )
    chunks = list(encode_batch(batch, schema=batch.schema))
    payload = b"".join(chunks).decode("utf-8").strip().splitlines()
    assert [json.loads(line) for line in payload] == [
        {"id": 1, "name": "a"},
        {"id": 2, "name": "b"},
    ]

    reader = pa.RecordBatchReader.from_batches(batch.schema, [batch])
    reader_chunks = list(encode_reader(reader))
    reader_payload = b"".join(reader_chunks).decode("utf-8").strip().splitlines()
    assert [json.loads(line) for line in reader_payload] == [
        {"id": 1, "name": "a"},
        {"id": 2, "name": "b"},
    ]


def test_get_export_codec_unknown_raises() -> None:
    """Raise when requesting an unknown export codec."""
    with pytest.raises(ExportCodecError, match="not registered"):
        get_export_codec("missing")
