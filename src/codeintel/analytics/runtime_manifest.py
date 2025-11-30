"""Shared manifest schema for analytics runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


@dataclass(frozen=True)
class AnalyticsScope:
    """Describe the scope of an analytics run."""

    paths: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    time_window: tuple[datetime, datetime] | None = None
    labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalyticsSkippedStep:
    """Reasoned skip for a planned step."""

    name: str
    reason: str
    kind: str | None = None


@dataclass(frozen=True)
class AnalyticsPlanInfo:
    """Planning metadata for an analytics run."""

    plan_id: str | None = None
    ordered_steps: tuple[str, ...] = ()
    skipped_steps: tuple[AnalyticsSkippedStep, ...] = ()
    dep_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


AnalyticsStatus = Literal["succeeded", "failed", "skipped"]


@dataclass(frozen=True)
class AnalyticsRunRecord:
    """Execution record for a single step in a run."""

    name: str
    kind: str
    status: AnalyticsStatus
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    attempts: int = 1
    partial: bool = False
    error: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalyticsRunReport:
    """Manifest-ready view of an analytics run."""

    repo: str
    commit: str
    run_id: str
    scope: AnalyticsScope
    records: tuple[AnalyticsRunRecord, ...]
    plan: AnalyticsPlanInfo
    tags: Mapping[str, str] = field(default_factory=dict)


def encode_manifest(report: AnalyticsRunReport) -> dict[str, object]:
    """
    Encode an AnalyticsRunReport into a JSON-serializable manifest payload.

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
        "records": [
            _encode_record(record)
            for record in report.records
        ],
    }


def _encode_record(record: AnalyticsRunRecord) -> dict[str, object]:
    """
    Encode a single AnalyticsRunRecord into manifest-friendly form.

    Returns
    -------
    dict[str, object]
        Serialized record payload.
    """
    payload: dict[str, object] = {
        "name": record.name,
        "kind": record.kind,
        "status": record.status,
        "attempts": record.attempts,
        "started_at": record.started_at.isoformat(),
        "ended_at": record.ended_at.isoformat(),
        "duration_ms": record.duration_ms,
        "partial": record.partial,
        "error": record.error,
        "meta": dict(record.meta),
    }
    contracts = record.meta.get("contracts") if isinstance(record.meta, dict) else None
    if contracts is not None:
        payload["contracts"] = contracts
    return payload


__all__ = [
    "AnalyticsPlanInfo",
    "AnalyticsRunRecord",
    "AnalyticsRunReport",
    "AnalyticsScope",
    "AnalyticsSkippedStep",
    "AnalyticsStatus",
    "encode_manifest",
]
