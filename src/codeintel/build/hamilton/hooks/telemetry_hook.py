"""Hamilton execution hook for node-level telemetry.

This module provides a Hamilton adapter hook that records per-node
execution telemetry to build.run_nodes for profiling and debugging.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import duckdb
import polars as pl
import pyarrow as pa
from hamilton.lifecycle import base as lifecycle_base

from codeintel.build.hamilton.build_log import record_build_event
from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.records import NodeExecutionRecord

if TYPE_CHECKING:
    from hamilton.node import Node

    from codeintel.build.hamilton.run_writer import BuildRunWriter


log = logging.getLogger(__name__)


class NodeTelemetryHook(lifecycle_base.BasePreNodeExecute, lifecycle_base.BasePostNodeExecute):
    """Hamilton lifecycle hook for node telemetry.

    Records execution timing and status for each node to enable
    fine-grained build profiling.

    This hook implements the Hamilton adapter protocol by providing
    pre_node_execute and post_node_execute methods that are called
    by the Hamilton driver during execution.

    Parameters
    ----------
    run_id
        Build run identifier for grouping.
    writer
        Build run writer for persistence.
    """

    def __init__(
        self,
        run_id: str,
        writer: BuildRunWriter,
        output_path: Path | None = None,
    ) -> None:
        """Initialize telemetry hook.

        Parameters
        ----------
        run_id
            Build run identifier for grouping.
        writer
            Build run writer for persistence.
        output_path
            Optional JSONL output path for node telemetry.
        """
        self._run_id = run_id
        self._writer = writer
        self._output_path = output_path
        self._node_starts: dict[str, datetime] = {}
        self._records: list[NodeExecutionRecord] = []
        self._last_flushed: list[NodeExecutionRecord] = []
        self._lock = threading.Lock()

    def pre_node_execute(
        self,
        *,
        node_: Node,
        **context: object,
    ) -> None:
        """Record node execution start.

        Parameters
        ----------
        node_
            Node being executed.
        context
            Additional Hamilton lifecycle context (ignored).
        """
        _ = context
        started_at = datetime.now(tz=UTC)
        with self._lock:
            self._node_starts[node_.name] = started_at

    def post_node_execute(
        self,
        *,
        node_: Node,
        success: bool,
        error: Exception | None,
        **context: object,
    ) -> None:
        """Record node execution completion.

        Parameters
        ----------
        node_
            Node that was executed.
        success
            Whether the node execution succeeded.
        error
            Exception if the node failed.
        context
            Additional Hamilton lifecycle context (ignored).
        """
        _ = context

        node_tags = node_.tags if isinstance(node_.tags, dict) else None

        completed_at = datetime.now(tz=UTC)
        node_name = node_.name
        with self._lock:
            started_at = self._node_starts.pop(node_name, completed_at)
        duration_ms = (completed_at - started_at).total_seconds() * 1000

        target_raw = node_tags.get(ht.TAG_TARGET) if node_tags else None
        target = target_raw if isinstance(target_raw, str) else None
        node_type_raw = node_tags.get(ht.TAG_NODE_TYPE) if node_tags else None
        node_type = node_type_raw if isinstance(node_type_raw, str) else None
        table_key_raw = node_tags.get(ht.TAG_TABLE_KEY) if node_tags else None
        table_key = table_key_raw if isinstance(table_key_raw, str) else None

        if not success:
            exception_type = type(error).__name__ if error else None
            error_message = str(error) if error else None
            record_build_event(
                "build.node.error",
                node_name=node_name,
                target=target,
                table_key=table_key,
                exception_type=exception_type,
                error=error_message,
            )
            log.error(
                "build.node.error run_id=%s node_name=%s target=%s table_key=%s exception_type=%s "
                "error=%s",
                self._run_id,
                node_name,
                target,
                table_key,
                exception_type,
                error_message,
            )

        record = NodeExecutionRecord(
            run_id=self._run_id,
            node_name=node_name,
            target=target,
            node_type=node_type,
            status="succeeded" if success else "failed",
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            error=str(error) if error else None,
            tags=node_tags if node_tags else None,
        )

        with self._lock:
            self._records.append(record)

    def flush(self) -> int:
        """Persist all recorded telemetry and clear buffer.

        Returns
        -------
        int
            Number of records persisted.
        """
        with self._lock:
            if not self._records:
                return 0
            records = list(self._records)
            self._records.clear()
            self._last_flushed = records

        saved = self._writer.save_run_nodes(self._run_id, records)
        if self._output_path is not None:
            try:
                _write_node_telemetry(self._output_path, records)
            except (OSError, TypeError, ValueError) as exc:
                log.warning(
                    "build.node.telemetry_write_failed run_id=%s error=%s",
                    self._run_id,
                    exc,
                )
        return saved

    def last_flushed_records(self) -> tuple[NodeExecutionRecord, ...]:
        """Return the most recently flushed records.

        Returns
        -------
        tuple[NodeExecutionRecord, ...]
            Records most recently flushed from the buffer.
        """
        with self._lock:
            return tuple(self._last_flushed)


class NodeIOTelemetryHook(lifecycle_base.BasePreNodeExecute, lifecycle_base.BasePostNodeExecute):
    """Hamilton lifecycle hook for per-node input/output telemetry.

    Persists a JSONL record per node containing input/output summaries. The log
    is flushed on every node completion to preserve data even for partial runs.
    """

    def __init__(
        self,
        run_id: str,
        *,
        output_path: Path,
    ) -> None:
        """Initialize the input/output telemetry hook."""
        self._run_id = run_id
        self._output_path = output_path
        self._node_starts: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def pre_node_execute(
        self,
        *,
        node_: Node,
        **context: object,
    ) -> None:
        """Record node start time for IO telemetry."""
        _ = context
        started_at = datetime.now(tz=UTC)
        with self._lock:
            self._node_starts[node_.name] = started_at

    def post_node_execute(
        self,
        *,
        node_: Node,
        success: bool,
        error: Exception | None,
        **context: object,
    ) -> None:
        """Log input/output summaries for a node execution."""
        node_tags = node_.tags if isinstance(node_.tags, dict) else None
        run_id = _context_string(context, "run_id") or self._run_id
        kwargs = _context_mapping(context, "kwargs")
        result = context.get("result")
        task_id = _context_string(context, "task_id")

        node_name = node_.name
        completed_at = datetime.now(tz=UTC)
        with self._lock:
            started_at = self._node_starts.pop(node_name, completed_at)
        duration_ms = (completed_at - started_at).total_seconds() * 1000

        payload = {
            "run_id": run_id,
            "node_name": node_name,
            "target": _tag_string(node_tags, ht.TAG_TARGET),
            "node_type": _tag_string(node_tags, ht.TAG_NODE_TYPE),
            "table_key": _tag_string(node_tags, ht.TAG_TABLE_KEY),
            "status": "succeeded" if success else "failed",
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_ms": duration_ms,
            "task_id": task_id,
            "error": str(error) if error else None,
            "inputs": _summarize_inputs(kwargs),
            "output": _summarize_value(result),
            "tags": _normalize_tags(node_tags),
        }
        _append_jsonl(self._output_path, payload)


def _write_node_telemetry(path: Path, records: list[NodeExecutionRecord]) -> None:
    payloads = [_node_record_payload(record) for record in records]
    lines = [
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        for payload in payloads
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _node_record_payload(record: NodeExecutionRecord) -> dict[str, object]:
    payload = asdict(record)
    started_at = record.started_at.isoformat()
    payload["started_at"] = started_at
    completed_at = record.completed_at.isoformat() if record.completed_at else None
    payload["completed_at"] = completed_at
    tags = record.tags
    if tags:
        payload["tags"] = {str(key): _normalize_tag_value(value) for key, value in tags.items()}
    return payload


def _normalize_tag_value(value: object) -> object:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _tag_string(tags: Mapping[str, object] | None, key: str) -> str | None:
    if not tags:
        return None
    value = tags.get(key)
    return value if isinstance(value, str) else None


def _context_string(context: dict[str, object], key: str) -> str | None:
    value = context.get(key)
    return value if isinstance(value, str) else None


def _context_mapping(context: dict[str, object], key: str) -> dict[str, object]:
    value = context.get(key)
    return value if isinstance(value, dict) else {}


def _normalize_tags(tags: Mapping[str, object] | None) -> dict[str, object] | None:
    if not tags:
        return None
    return {str(key): _normalize_tag_value(value) for key, value in tags.items()}


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    line = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def _summarize_inputs(kwargs: dict[str, object]) -> dict[str, object]:
    summaries: dict[str, object] = {}
    for key, value in kwargs.items():
        summaries[key] = _summarize_value(value)
    return summaries


def _summarize_value(value: object) -> dict[str, object] | None:
    summary: dict[str, object] = {"type": "None", "empty": True}
    if value is not None:
        summary = {"type": type(value).__name__}
        if isinstance(value, pa.Table):
            summary.update(_summarize_arrow_table(value))
        elif isinstance(value, pa.RecordBatch):
            batch = cast("pa.RecordBatch", value)
            summary.update(
                {
                    "row_count": batch.num_rows,
                    "column_count": batch.num_columns,
                    "columns": list(batch.schema.names),
                    "schema": {
                        name: str(batch.schema.field(name).type) for name in batch.schema.names
                    },
                    "empty": batch.num_rows == 0 or batch.num_columns == 0,
                }
            )
        elif isinstance(value, pa.RecordBatchReader):
            reader = cast("pa.RecordBatchReader", value)
            summary.update(
                {
                    "row_count": None,
                    "column_count": len(reader.schema.names),
                    "columns": list(reader.schema.names),
                    "schema": {
                        name: str(reader.schema.field(name).type) for name in reader.schema.names
                    },
                    "empty": None,
                    "streaming": True,
                }
            )
        elif isinstance(value, pl.DataFrame):
            summary.update(_summarize_polars_frame(value))
        elif isinstance(value, pl.LazyFrame):
            summary.update(_summarize_polars_lazy(value))
        elif isinstance(value, duckdb.DuckDBPyRelation):
            summary.update(_summarize_duckdb_relation(value))
        elif isinstance(value, dict):
            summary.update(_summarize_mapping(value))
        elif isinstance(value, (list, tuple)):
            summary.update(_summarize_sequence(value))
    return summary


def _summarize_arrow_table(table: pa.Table) -> dict[str, object]:
    columns = list(table.column_names)
    schema = {name: str(table.schema.field(name).type) for name in columns}
    null_counts = {name: table.column(name).null_count for name in columns}
    return {
        "row_count": table.num_rows,
        "column_count": table.num_columns,
        "columns": columns,
        "schema": schema,
        "null_counts": null_counts,
        "empty": table.num_rows == 0 or table.num_columns == 0,
    }


def _summarize_polars_frame(frame: pl.DataFrame) -> dict[str, object]:
    schema = dict(zip(frame.columns, (str(dtype) for dtype in frame.dtypes), strict=False))
    return {
        "row_count": frame.height,
        "column_count": frame.width,
        "columns": list(frame.columns),
        "schema": schema,
        "empty": frame.height == 0 or frame.width == 0,
    }


def _summarize_polars_lazy(frame: pl.LazyFrame) -> dict[str, object]:
    try:
        schema = frame.collect_schema()
        columns = list(schema.names())
        types = [str(dtype) for dtype in schema.dtypes()]
    except (AttributeError, pl.exceptions.PolarsError, ValueError):
        columns = []
        types = []
    return {
        "row_count": None,
        "column_count": len(columns),
        "columns": columns,
        "schema": dict(zip(columns, types, strict=False)),
        "empty": None,
        "lazy": True,
    }


def _summarize_duckdb_relation(rel: duckdb.DuckDBPyRelation) -> dict[str, object]:
    try:
        columns = list(rel.columns)
    except (AttributeError, TypeError):
        columns = []
    return {
        "row_count": None,
        "column_count": len(columns),
        "columns": columns,
        "empty": None,
        "relation": True,
    }


def _summarize_mapping(values: dict[object, object]) -> dict[str, object]:
    keys = list(values.keys())
    sample = [str(key) for key in keys[:5]]
    return {"length": len(values), "sample_keys": sample}


def _summarize_sequence(values: list[object] | tuple[object, ...]) -> dict[str, object]:
    sample = values[:5]
    return {
        "length": len(values),
        "sample_types": [type(item).__name__ for item in sample],
    }


__all__ = [
    "NodeExecutionRecord",
    "NodeIOTelemetryHook",
    "NodeTelemetryHook",
]
