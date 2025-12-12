"""Shared manifest schema for analytics runtimes.

This module provides manifest types for analytics runs, using core plugin
types where possible and extending them for analytics-specific needs.

Core Type Mappings
------------------
- ``PluginExecutionRecord`` from core is used directly for execution records
- ``PluginSkip`` from core is extended with analytics-specific fields
- ``BaseExecutionReport`` from core is extended for analytics run reports
- ``AnalyticsScope`` is analytics-specific (time windows, labels)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.core.plugins.registry.base import PluginSkip
from codeintel.core.plugins.types.report import BaseExecutionReport
from codeintel.core.plugins.types.result import PluginExecutionRecord, PluginStatus

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class AnalyticsScope:
    """Describe the scope of an analytics run.

    This is analytics-specific, supporting time-windowed queries
    and label-based filtering not found in core plugin types.

    Attributes
    ----------
    paths
        File paths included in the scope.
    modules
        Module names included in the scope.
    time_window
        Optional start/end datetime for time-bounded analysis.
    labels
        Key-value labels for filtering.
    """

    paths: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    time_window: tuple[datetime, datetime] | None = None
    labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalyticsSkippedStep(PluginSkip):
    """Extended skip metadata with analytics-specific kind field.

    Extend core ``PluginSkip`` with an optional ``kind`` field for
    categorizing skipped analytics steps.

    Attributes
    ----------
    name
        Step name (inherited from PluginSkip).
    reason
        Reason for skipping (inherited from PluginSkip).
    kind
        Optional step kind/category.
    """

    kind: str | None = None


@dataclass(frozen=True)
class AnalyticsPlanInfo:
    """Planning metadata for an analytics run.

    This is a data-only view of a plan, containing step names rather than
    actual plugin instances. Used for manifest serialization.

    Attributes
    ----------
    plan_id
        Unique identifier for the plan.
    ordered_steps
        Step names in execution order.
    skipped_steps
        Steps excluded from execution with reasons.
    dep_graph
        Dependency relationships between steps.
    """

    plan_id: str | None = None
    ordered_steps: tuple[str, ...] = ()
    skipped_steps: tuple[AnalyticsSkippedStep, ...] = ()
    dep_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalyticsRunReport(BaseExecutionReport):
    """Manifest-ready view of an analytics run.

    Extend ``BaseExecutionReport`` with analytics-specific fields for
    repository context, scope, plan info, and tags.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    scope
        Analytics scope describing what was analyzed.
    plan
        Planning information for the run.
    tags
        Key-value tags for categorization.
    """

    repo: str = ""
    commit: str = ""
    scope: AnalyticsScope = field(default_factory=AnalyticsScope)
    plan: AnalyticsPlanInfo = field(default_factory=AnalyticsPlanInfo)
    tags: Mapping[str, str] = field(default_factory=dict)


def encode_manifest(report: AnalyticsRunReport) -> dict[str, object]:
    """Encode an AnalyticsRunReport into a JSON-serializable manifest payload.

    Parameters
    ----------
    report
        The analytics run report to encode.

    Returns
    -------
    dict[str, object]
        Manifest payload ready for persistence.
    """
    return {
        "repo": report.repo,
        "commit": report.commit,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "run_id": report.run_id,
        "plan": {
            "plan_id": report.plan.plan_id,
            "ordered_steps": list(report.plan.ordered_steps),
            "skipped_steps": [
                {
                    "name": skipped.name,
                    "reason": skipped.reason,
                    "kind": skipped.kind,
                }
                for skipped in report.plan.skipped_steps
            ],
            "dep_graph": {name: list(deps) for name, deps in report.plan.dep_graph.items()},
        },
        "scope": {
            "paths": list(report.scope.paths),
            "modules": list(report.scope.modules),
            "time_window": (
                (
                    report.scope.time_window[0].isoformat(),
                    report.scope.time_window[1].isoformat(),
                )
                if report.scope.time_window is not None
                else None
            ),
            "labels": dict(report.scope.labels),
        },
        "tags": dict(report.tags),
        "records": [_encode_record(record) for record in report.records],
    }


def _encode_record(record: PluginExecutionRecord) -> dict[str, object]:
    """Encode a single PluginExecutionRecord into manifest-friendly form.

    Parameters
    ----------
    record
        The execution record to encode.

    Returns
    -------
    dict[str, object]
        Serialized record payload.
    """
    kind = record.meta.get("kind", "") if record.meta else ""

    payload: dict[str, object] = {
        "name": record.plugin_name,
        "kind": kind,
        "status": record.status,
        "attempts": record.attempts,
        "started_at": record.started_at.isoformat(),
        "ended_at": record.ended_at.isoformat(),
        "duration_ms": record.duration_ms,
        "partial": record.partial,
        "error": record.error,
        "meta": dict(record.meta) if record.meta else {},
    }
    contracts = record.meta.get("contracts") if isinstance(record.meta, dict) else None
    if contracts is not None:
        payload["contracts"] = contracts
    return payload


__all__ = [
    "AnalyticsPlanInfo",
    "AnalyticsRunReport",
    "AnalyticsScope",
    "AnalyticsSkippedStep",
    "PluginExecutionRecord",
    "PluginStatus",
    "encode_manifest",
]
