"""Hamilton execution hook for node-level telemetry.

This module provides a Hamilton adapter hook that records per-node
execution telemetry to build.run_nodes for profiling and debugging.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from hamilton.lifecycle import base as lifecycle_base

from codeintel.build.hamilton import tags as ht
from codeintel.hamilton.records import NodeExecutionRecord

if TYPE_CHECKING:
    from hamilton.node import Node

    from codeintel.storage.gateway.protocol import StorageGateway


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
    gateway
        Storage gateway for persistence.
    """

    def __init__(self, run_id: str, gateway: StorageGateway) -> None:
        """Initialize telemetry hook.

        Parameters
        ----------
        run_id
            Build run identifier for grouping.
        gateway
            Storage gateway for persistence.
        """
        self._run_id = run_id
        self._gateway = gateway
        self._node_starts: dict[str, datetime] = {}
        self._records: list[NodeExecutionRecord] = []
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

        return self._gateway.build.save_run_nodes(self._run_id, records)


__all__ = [
    "NodeExecutionRecord",
    "NodeTelemetryHook",
]
