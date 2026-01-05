"""Hamilton execution hook for node-level telemetry.

This module provides a Hamilton adapter hook that records per-node
execution telemetry to build.run_nodes for profiling and debugging.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

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


__all__ = [
    "NodeExecutionRecord",
    "NodeTelemetryHook",
]
