"""Data structures for graph runtime planning and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from codeintel.analytics.graphs.contracts import PluginContractResult
from codeintel.analytics.graphs.plugins import GraphMetricPluginSkip
from codeintel.config.steps_graphs import GraphRunScope


@dataclass(frozen=True)
class GraphPluginRunRecord:
    """Capture execution telemetry for a single graph plugin."""

    name: str
    stage: str
    severity: Literal["fatal", "soft_fail", "skip_on_error"]
    status: Literal["succeeded", "failed", "skipped"]
    attempts: int
    timeout_ms: int | None
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    partial: bool
    run_id: str
    error: str | None = None
    options: object | None = None
    input_hash: str | None = None
    options_hash: str | None = None
    version_hash: str | None = None
    skipped_reason: str | None = None
    row_counts: dict[str, int] | None = None
    contracts: tuple[PluginContractResult, ...] = ()
    requires_isolation: bool = False
    isolation_kind: str | None = None
    policy_fail_fast: bool = False


@dataclass(frozen=True)
class GraphPluginRunOptions:
    """Optional controls for plugin execution."""

    plugin_options: dict[str, dict[str, object]] | None = None
    manifest_path: Path | None = None
    scope: GraphRunScope | None = None
    dry_run: bool | None = None


@dataclass(frozen=True)
class GraphPluginRunReport:
    """Aggregate report for a batch of graph plugin executions."""

    repo: str
    commit: str
    records: tuple[GraphPluginRunRecord, ...]
    scope: GraphRunScope
    run_id: str
    plan_id: str
    ordered_plugins: tuple[str, ...]
    skipped_plugins: tuple[GraphMetricPluginSkip, ...]
    dep_graph: dict[str, tuple[str, ...]]


__all__ = [
    "GraphPluginRunOptions",
    "GraphPluginRunRecord",
    "GraphPluginRunReport",
]
