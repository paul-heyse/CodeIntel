"""Hamilton execution hook for node-level telemetry.

This module provides a Hamilton adapter hook that records per-node
execution telemetry to build.run_nodes for profiling and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

try:
    from hamilton.lifecycle.base import BasePostNodeExecute as _BasePostNodeExecute
    from hamilton.lifecycle.base import BasePreNodeExecute as _BasePreNodeExecute
except ImportError:
    _BasePreNodeExecute = object
    _BasePostNodeExecute = object

BasePreNodeExecute: type = _BasePreNodeExecute
BasePostNodeExecute: type = _BasePostNodeExecute

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway


@dataclass
class NodeExecutionRecord:
    """Record of a single node execution.

    Attributes
    ----------
    run_id
        Parent run identifier.
    node_name
        Hamilton node name.
    target
        Parent target if applicable.
    node_kind
        Node kind: compute, materialize, tool, etc.
    status
        Execution status: succeeded, failed, skipped.
    started_at
        When node execution started.
    completed_at
        When node execution completed.
    duration_ms
        Execution duration in milliseconds.
    error
        Error message if execution failed.
    tags
        Hamilton tags from node.
    """

    run_id: str
    node_name: str
    target: str | None
    node_kind: str | None
    status: str
    started_at: datetime
    completed_at: datetime | None
    duration_ms: float | None
    error: str | None
    tags: dict[str, object] | None


class NodeTelemetryHook(BasePreNodeExecute, BasePostNodeExecute):
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

    def pre_node_execute(
        self,
        *,
        node_name: str,
        **kwargs: object,
    ) -> None:
        """Record node execution start.

        Parameters
        ----------
        node_name
            Name of the node being executed.
        **kwargs
            Additional keyword arguments from Hamilton.
        """
        _ = kwargs
        self._node_starts[node_name] = datetime.now(tz=UTC)

    def post_node_execute(
        self,
        *,
        node_name: str,
        **kwargs: object,
    ) -> None:
        """Record node execution completion.

        Parameters
        ----------
        node_name
            Name of the node that was executed.
        **kwargs
            Additional keyword arguments from Hamilton.
        """
        node_tags_raw = kwargs.get("node_tags")
        node_tags = cast("dict[str, object] | None", node_tags_raw) if isinstance(node_tags_raw, dict) else None
        error_raw = kwargs.get("error")
        error = error_raw if isinstance(error_raw, Exception) else None
        success_raw = kwargs.get("success")
        success = success_raw if isinstance(success_raw, bool) else False

        completed_at = datetime.now(tz=UTC)
        started_at = self._node_starts.pop(node_name, completed_at)
        duration_ms = (completed_at - started_at).total_seconds() * 1000

        target_raw = node_tags.get("target") if node_tags else None
        target = target_raw if isinstance(target_raw, str) else None
        node_kind_raw = node_tags.get("node_kind") if node_tags else None
        node_kind = node_kind_raw if isinstance(node_kind_raw, str) else None

        record = NodeExecutionRecord(
            run_id=self._run_id,
            node_name=node_name,
            target=target,
            node_kind=node_kind,
            status="succeeded" if success else "failed",
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            error=str(error) if error else None,
            tags=node_tags if node_tags else None,
        )

        self._records.append(record)

    def flush(self) -> int:
        """Persist all recorded telemetry and clear buffer.

        Returns
        -------
        int
            Number of records persisted.
        """
        if not self._records:
            return 0

        count = self._gateway.build.save_run_nodes(self._run_id, self._records)
        self._records.clear()
        return count


__all__ = [
    "NodeExecutionRecord",
    "NodeTelemetryHook",
]
