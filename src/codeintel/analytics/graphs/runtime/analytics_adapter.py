"""Adapters between graph runtime reports and the generic analytics manifest."""

from __future__ import annotations

from typing import Any, Literal, cast

from codeintel.analytics.graphs.contracts import PluginContractResult
from codeintel.analytics.graphs.plugins import GraphMetricPluginSkip
from codeintel.analytics.graphs.runtime.model import (
    GraphPluginRunRecord,
    GraphPluginRunReport,
)
from codeintel.analytics.runtime_manifest import (
    AnalyticsPlanInfo,
    AnalyticsRunRecord,
    AnalyticsRunReport,
    AnalyticsScope,
    AnalyticsSkippedStep,
    AnalyticsStatus,
)
from codeintel.config.steps_graphs import GraphRunScope


def _parse_contracts(meta: dict[str, object]) -> tuple[PluginContractResult, ...]:
    raw = meta.get("contracts")
    if not isinstance(raw, (tuple, list)):
        return ()
    parsed: list[PluginContractResult] = []
    allowed_statuses = {"passed", "failed", "soft_failed"}
    for contract in raw:
        if isinstance(contract, PluginContractResult):
            parsed.append(contract)
            continue
        if isinstance(contract, dict):
            name = contract.get("name")
            status = contract.get("status")
            message = contract.get("message")
            if isinstance(name, str) and isinstance(status, str):
                normalized_status = cast(
                    "Literal['passed','failed','soft_failed']",
                    status if status in allowed_statuses else "failed",
                )
                parsed.append(
                    PluginContractResult(
                        name=name,
                        status=normalized_status,
                        message=message if isinstance(message, str) else None,
                    )
                )
    return tuple(parsed)


def _scope_from_graph(scope: GraphRunScope) -> AnalyticsScope:
    """
    Convert GraphRunScope to the shared AnalyticsScope.

    Returns
    -------
    AnalyticsScope
        Scope adapted for the generic analytics manifest.
    """
    return AnalyticsScope(
        paths=scope.paths,
        modules=scope.modules,
        time_window=scope.time_window,
        labels={"runtime": "graph"},
    )


def _plan_from_graph(report: GraphPluginRunReport) -> AnalyticsPlanInfo:
    """
    Convert GraphPluginRunReport plan metadata to AnalyticsPlanInfo.

    Returns
    -------
    AnalyticsPlanInfo
        Plan metadata compatible with analytics manifests.
    """
    return AnalyticsPlanInfo(
        plan_id=report.plan_id,
        ordered_steps=report.ordered_plugins,
        skipped_steps=tuple(
            AnalyticsSkippedStep(
                name=skipped.name,
                reason=skipped.reason,
                kind="graph_plugin",
            )
            for skipped in report.skipped_plugins
        ),
        dep_graph=report.dep_graph,
    )


def _meta_from_graph_record(rec: GraphPluginRunRecord) -> dict[str, Any]:
    """
    Pack graph-specific metadata into the generic record meta field.

    Returns
    -------
    dict[str, Any]
        Mapping of graph-specific attributes.
    """
    return {
        "stage": rec.stage,
        "severity": rec.severity,
        "timeout_ms": rec.timeout_ms,
        "input_hash": rec.input_hash,
        "options_hash": rec.options_hash,
        "version_hash": rec.version_hash,
        "skipped_reason": rec.skipped_reason,
        "row_counts": rec.row_counts,
        "requires_isolation": rec.requires_isolation,
        "isolation_kind": rec.isolation_kind,
        "policy_fail_fast": rec.policy_fail_fast,
        "options_present": rec.options is not None,
        "contracts": [
            {
                "name": contract.name,
                "status": contract.status,
                "message": contract.message,
            }
            for contract in rec.contracts
        ],
    }


def graph_run_to_analytics(report: GraphPluginRunReport) -> AnalyticsRunReport:
    """
    Convert a graph runtime report into the generic analytics manifest model.

    Returns
    -------
    AnalyticsRunReport
        Generic manifest-ready report.
    """
    scope = _scope_from_graph(report.scope)
    plan = _plan_from_graph(report)
    records = tuple(
        AnalyticsRunRecord(
            name=rec.name,
            kind="graph_plugin",
            status=rec.status,
            started_at=rec.started_at,
            ended_at=rec.ended_at,
            duration_ms=rec.duration_ms,
            attempts=rec.attempts,
            partial=rec.partial,
            error=rec.error,
            meta=_meta_from_graph_record(rec),
        )
        for rec in report.records
    )
    return AnalyticsRunReport(
        repo=report.repo,
        commit=report.commit,
        run_id=report.run_id,
        scope=scope,
        records=records,
        plan=plan,
        tags={"runtime": "graph"},
    )


def _graph_record_from_analytics(rec: AnalyticsRunRecord, run_id: str) -> GraphPluginRunRecord:
    """
    Convert an AnalyticsRunRecord back into a graph runtime record.

    Returns
    -------
    GraphPluginRunRecord
        Record aligned to graph runtime telemetry schema.
    """
    meta = rec.meta if isinstance(rec.meta, dict) else {}
    stage = str(meta.get("stage") or rec.kind)
    severity_raw = meta.get("severity")
    severity_allowed = {"fatal", "soft_fail", "skip_on_error"}
    severity = cast(
        "Literal['fatal','soft_fail','skip_on_error']",
        severity_raw
        if isinstance(severity_raw, str) and severity_raw in severity_allowed
        else "fatal",
    )
    options_hash = meta.get("options_hash")
    version_hash = meta.get("version_hash")
    input_hash = meta.get("input_hash")
    timeout_ms = meta.get("timeout_ms")
    skipped_reason = meta.get("skipped_reason")
    contracts = _parse_contracts(meta)
    status: AnalyticsStatus = rec.status

    return GraphPluginRunRecord(
        name=rec.name,
        stage=stage,
        severity=severity,
        status=status,
        attempts=rec.attempts,
        timeout_ms=timeout_ms if isinstance(timeout_ms, int) else None,
        started_at=rec.started_at,
        ended_at=rec.ended_at,
        duration_ms=rec.duration_ms,
        partial=rec.partial,
        run_id=run_id,
        error=rec.error,
        options=None,
        input_hash=input_hash if isinstance(input_hash, str) else None,
        options_hash=options_hash if isinstance(options_hash, str) else None,
        version_hash=version_hash if isinstance(version_hash, str) else None,
        skipped_reason=skipped_reason if isinstance(skipped_reason, str) else None,
        row_counts=meta.get("row_counts") if isinstance(meta.get("row_counts"), dict) else None,
        contracts=contracts,
        requires_isolation=bool(meta.get("requires_isolation", False)),
        isolation_kind=meta.get("isolation_kind")
        if isinstance(meta.get("isolation_kind"), str)
        else None,
        policy_fail_fast=bool(meta.get("policy_fail_fast", False)),
    )


def analytics_to_graph_run(report: AnalyticsRunReport) -> GraphPluginRunReport:
    """
    Convert an AnalyticsRunReport back into a GraphPluginRunReport.

    Returns
    -------
    GraphPluginRunReport
        Graph runtime-compatible run report.
    """
    scope = GraphRunScope(
        paths=tuple(report.scope.paths),
        modules=tuple(report.scope.modules),
        time_window=report.scope.time_window,
    )
    skipped = tuple(
        GraphMetricPluginSkip(name=step.name, reason="disabled")
        for step in report.plan.skipped_steps
    )
    records = tuple(_graph_record_from_analytics(rec, report.run_id) for rec in report.records)

    return GraphPluginRunReport(
        repo=report.repo,
        commit=report.commit,
        records=records,
        scope=scope,
        run_id=report.run_id,
        plan_id=report.plan.plan_id or "",
        ordered_plugins=tuple(report.plan.ordered_steps),
        skipped_plugins=skipped,
        dep_graph=dict(report.plan.dep_graph),
    )


__all__ = [
    "analytics_to_graph_run",
    "graph_run_to_analytics",
]
