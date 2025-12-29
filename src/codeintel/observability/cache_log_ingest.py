"""Ingest Hamilton cache JSONL logs into DuckDB."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import duckdb

if TYPE_CHECKING:
    from typing import TypedDict

    from duckdb import DuckDBPyConnection

    class _EventRow(TypedDict):
        event_id: str
        run_id: str
        node_name: str | None
        event_type: str | None
        data_version: str | None
        actor: str | None
        task_id: str | None
        message: str | None
        ts: float | None
        event_at: str | None
        source_file: str
        source_line: int
        raw_event: str


_CONFIG_ERROR_MESSAGE = "cache_dir or jsonl_paths must be provided"

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CacheLogIngestResult:
    """Summary of cache log ingestion."""

    inserted_events: int
    run_ids: tuple[str, ...]
    jsonl_files: tuple[str, ...]


class CacheLogIngestConfigError(ValueError):
    """Raised when cache log ingestion inputs are invalid."""

    def __init__(self) -> None:
        super().__init__(_CONFIG_ERROR_MESSAGE)


def ingest_cache_log_jsonl(
    *,
    duckdb_path: Path,
    cache_dir: Path | None = None,
    jsonl_paths: Sequence[Path] | None = None,
) -> CacheLogIngestResult:
    """Ingest cache event JSONL files into DuckDB.

    Parameters
    ----------
    duckdb_path
        DuckDB database file path.
    cache_dir
        Cache directory containing JSONL log files (recursive).
    jsonl_paths
        Optional explicit list of JSONL files to ingest.

    Returns
    -------
    CacheLogIngestResult
        Summary of ingested events and source files.
    """
    sources = _resolve_jsonl_paths(cache_dir=cache_dir, jsonl_paths=jsonl_paths)
    if not sources:
        return CacheLogIngestResult(inserted_events=0, run_ids=(), jsonl_files=())

    con = duckdb.connect(str(duckdb_path))
    try:
        _ensure_tables(con)
        inserted, run_ids = _ingest_events(con, sources)
    finally:
        con.close()

    return CacheLogIngestResult(
        inserted_events=inserted,
        run_ids=tuple(sorted(run_ids)),
        jsonl_files=tuple(str(path) for path in sources),
    )


def _resolve_jsonl_paths(
    *,
    cache_dir: Path | None,
    jsonl_paths: Sequence[Path] | None,
) -> list[Path]:
    if jsonl_paths:
        return sorted({path for path in jsonl_paths if path.exists()})
    if cache_dir is None:
        raise CacheLogIngestConfigError
    if not cache_dir.exists():
        return []
    return sorted(cache_dir.rglob("*.jsonl"))


def _ensure_tables(con: DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS observability;")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS observability.cache_events (
            event_id VARCHAR PRIMARY KEY,
            run_id VARCHAR NOT NULL,
            node_name VARCHAR,
            event_type VARCHAR,
            data_version VARCHAR,
            actor VARCHAR,
            task_id VARCHAR,
            message VARCHAR,
            ts DOUBLE,
            event_at TIMESTAMPTZ,
            source_file VARCHAR,
            source_line BIGINT,
            raw_event JSON
        );
        """
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cache_events_run
        ON observability.cache_events(run_id);
        """
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cache_events_node
        ON observability.cache_events(run_id, node_name);
        """
    )


def _ingest_events(
    con: DuckDBPyConnection,
    jsonl_files: Iterable[Path],
) -> tuple[int, set[str]]:
    inserted = 0
    run_ids: set[str] = set()
    for path in jsonl_files:
        inserted += _ingest_file(con, path, run_ids=run_ids)
    return inserted, run_ids


def _ingest_file(
    con: DuckDBPyConnection,
    path: Path,
    *,
    run_ids: set[str],
) -> int:
    inserted = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        log.warning("cache_log_ingest.read_failed path=%s error=%s", path, exc)
        return 0

    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        row = _event_row(event, path=path, line_no=line_no)
        if row is None:
            continue
        run_ids.add(row["run_id"])
        con.execute(
            """
            INSERT INTO observability.cache_events
              (event_id, run_id, node_name, event_type, data_version, actor, task_id, message,
               ts, event_at, source_file, source_line, raw_event)
            VALUES
              (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS JSON))
            ON CONFLICT(event_id) DO NOTHING
            """,
            [
                row["event_id"],
                row["run_id"],
                row["node_name"],
                row["event_type"],
                row["data_version"],
                row["actor"],
                row["task_id"],
                row["message"],
                row["ts"],
                row["event_at"],
                row["source_file"],
                row["source_line"],
                row["raw_event"],
            ],
        )
        inserted += con.rowcount
    return inserted


def _event_row(
    event: object,
    *,
    path: Path,
    line_no: int,
) -> _EventRow | None:
    if not isinstance(event, dict):
        return None
    run_id = _string_value(event.get("run_id"))
    if run_id is None:
        return None
    node_name = _string_value(event.get("node_name"))
    event_type = _string_value(event.get("event_type"))
    actor = _string_value(event.get("actor"))
    task_id = _string_value(event.get("task_id"))
    message = _string_value(event.get("msg"))
    timestamp = _float_value(event.get("timestamp"))
    event_at = _event_timestamp(timestamp)
    data_version = _string_value(event.get("value"))
    event_id = _event_id(
        _EventIdSeed(
            run_id=run_id,
            node_name=node_name,
            event_type=event_type,
            ts=timestamp,
            source_file=str(path),
            source_line=line_no,
        )
    )
    raw_event = json.dumps(event, sort_keys=True, default=str)
    return cast(
        "_EventRow",
        {
            "event_id": event_id,
            "run_id": run_id,
            "node_name": node_name,
            "event_type": event_type,
            "data_version": data_version,
            "actor": actor,
            "task_id": task_id,
            "message": message,
            "ts": timestamp,
            "event_at": event_at,
            "source_file": str(path),
            "source_line": line_no,
            "raw_event": raw_event,
        },
    )


@dataclass(frozen=True, slots=True)
class _EventIdSeed:
    run_id: str
    node_name: str | None
    event_type: str | None
    ts: float | None
    source_file: str
    source_line: int


def _event_id(
    seed: _EventIdSeed,
) -> str:
    parts = [
        seed.run_id,
        seed.node_name or "",
        seed.event_type or "",
        f"{seed.ts:.6f}" if seed.ts is not None else "",
        seed.source_file,
        str(seed.source_line),
    ]
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha1(raw, usedforsecurity=False).hexdigest()


def _string_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _float_value(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _event_timestamp(ts: float | None) -> str | None:
    if ts is None:
        return None
    dt = datetime.fromtimestamp(ts, tz=UTC)
    return dt.isoformat()


__all__ = [
    "CacheLogIngestConfigError",
    "CacheLogIngestResult",
    "ingest_cache_log_jsonl",
]
