"""Tests for decision trace serialization."""

from __future__ import annotations

from datetime import UTC, datetime

from codeintel.build.hamilton.decision_trace import build_decision_trace_payload
from codeintel.build.manifest.records import CacheManifestEntry

DURATION_MS = 10.0
SIZE_BYTES = 100


def test_decision_trace_payload_preserves_entry_order() -> None:
    """Preserve entry order in decision trace payloads."""
    first = CacheManifestEntry(
        run_id="run-1",
        node_name="t__alpha",
        status="hit",
        recorded_at=datetime.now(tz=UTC),
        cache_key="alpha",
    )
    second = CacheManifestEntry(
        run_id="run-1",
        node_name="t__beta",
        status="miss",
        recorded_at=datetime.now(tz=UTC),
        cache_key="beta",
    )

    payload = build_decision_trace_payload([first, second])

    assert payload[0]["node_name"] == "t__alpha"
    assert payload[0]["index"] == 0
    assert payload[1]["node_name"] == "t__beta"
    assert payload[1]["index"] == 1


def test_decision_trace_payload_includes_cache_fields() -> None:
    """Include cache metadata in decision trace payloads."""
    entry = CacheManifestEntry(
        run_id="run-1",
        node_name="t__alpha",
        status="store",
        recorded_at=datetime.now(tz=UTC),
        cache_key="alpha",
        cache_version="v1",
        cache_path="/tmp/cache",
        duration_ms=DURATION_MS,
        size_bytes=SIZE_BYTES,
        target="alpha",
    )

    payload = build_decision_trace_payload([entry])
    record = payload[0]

    assert record["cache_key"] == "alpha"
    assert record["cache_version"] == "v1"
    assert record["cache_path"] == "/tmp/cache"
    assert record["duration_ms"] == DURATION_MS
    assert record["size_bytes"] == SIZE_BYTES
    assert record["target"] == "alpha"
