"""NDJSON streaming tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

from codeintel.serving.http.streaming import ndjson_stream
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_true,
)


def test_ndjson_stream_utf8_and_stringifies_types() -> None:
    """Ensure ndjson_stream emits UTF-8 with stringified values."""
    expected_ts = "2024-01-01T00:00:00Z"
    row: dict[str, object] = {
        "text": "naïve 🧪",
        "ts": datetime(2024, 1, 1, tzinfo=UTC),
        "uuid": UUID("12345678-1234-5678-1234-567812345678"),
        "bytes": b"hello",
    }

    line = next(ndjson_stream([row]))
    expect_true(line.endswith(b"\n"))

    decoded = line.decode("utf-8")
    expect_in("🧪", decoded)

    payload = json.loads(decoded)
    expect_equal(payload["text"], row["text"])
    expect_equal(payload["ts"], expected_ts)
    expect_equal(payload["uuid"], str(row["uuid"]))
    expect_equal(payload["bytes"], str(row["bytes"]))
