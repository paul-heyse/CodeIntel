"""Tests for cache log ingestion into DuckDB."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from codeintel.observability.cache_log_ingest import ingest_cache_log_jsonl

EXPECTED_EVENT_COUNT = 2


def _write_event_log(path: Path, events: list[dict[str, object]]) -> None:
    """Write a JSONL fixture to the given path."""
    lines = [json.dumps(event, sort_keys=True) for event in events]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_cache_log_ingest_inserts_events(tmp_path: Path) -> None:
    """Ingest JSONL events and verify persisted rows."""
    log_path = tmp_path / "cache_events.jsonl"
    _write_event_log(
        log_path,
        [
            {
                "run_id": "run-1",
                "node_name": "node_a",
                "event_type": "get_result",
                "actor": "result_store",
                "value": "dv-1",
                "timestamp": 1_700_000_000.0,
            },
            {
                "run_id": "run-1",
                "node_name": "node_b",
                "event_type": "execute_node",
                "actor": "adapter",
                "timestamp": 1_700_000_001.0,
            },
        ],
    )

    db_path = tmp_path / "cache.duckdb"
    result = ingest_cache_log_jsonl(duckdb_path=db_path, jsonl_paths=[log_path])
    assert result.inserted_events == EXPECTED_EVENT_COUNT
    assert result.run_ids == ("run-1",)

    con = duckdb.connect(str(db_path))
    try:
        row = con.execute("SELECT count(*) FROM observability.cache_events").fetchone()
        assert row is not None
        count = row[0]
        assert count == EXPECTED_EVENT_COUNT
    finally:
        con.close()
