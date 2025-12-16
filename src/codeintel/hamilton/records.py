"""Shared Hamilton execution record dataclasses.

These records are persisted to DuckDB tables by the storage layer, so they
must be importable without creating a storage→build dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping
else:
    from collections.abc import Mapping


class DatasetRefProtocol(Protocol):
    """Minimal protocol for dataset references used in telemetry records."""

    table_key: str


class ArtifactRefProtocol(Protocol):
    """Minimal protocol for artifact references used in telemetry records."""

    name: str


@dataclass(frozen=True)
class TargetRunRecord:
    """Record of a Hamilton node execution for a target."""

    target: str
    plugin_name: str
    status: str
    input_hash: str | None
    options_hash: str | None = None
    duration_ms: float = 0.0
    row_counts: Mapping[str, int] = field(default_factory=dict)
    error: str | None = None
    datasets: tuple[DatasetRefProtocol, ...] = ()
    artifacts: tuple[ArtifactRefProtocol, ...] = ()

    @property
    def success(self) -> bool:
        """Return True if execution succeeded."""
        return self.status == "succeeded"

    @property
    def skipped(self) -> bool:
        """Return True if execution was skipped."""
        return self.status == "skipped"

    def get_dataset(self, table_key: str) -> DatasetRefProtocol | None:
        """Return the dataset ref for table_key when present."""
        for ds in self.datasets:
            if ds.table_key == table_key:
                return ds
        return None


@dataclass
class NodeExecutionRecord:
    """Record of a single node execution."""

    run_id: str
    node_name: str
    target: str | None
    node_type: str | None
    status: str
    started_at: datetime
    completed_at: datetime | None
    duration_ms: float | None
    error: str | None
    tags: Mapping[str, object] | None


__all__ = [
    "NodeExecutionRecord",
    "TargetRunRecord",
]
