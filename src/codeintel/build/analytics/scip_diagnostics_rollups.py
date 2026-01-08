"""SCIP diagnostics rollup helpers for post-run analytics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

SCIP_DIAGNOSTICS_TABLE_KEY = "core.scip_diagnostics"
SCIP_DIAGNOSTICS_SUMMARY_TABLE_KEY = "analytics.scip_diagnostics_summary"
SCIP_DIAGNOSTICS_BY_FILE_TABLE_KEY = "analytics.scip_diagnostics_by_file"
SCIP_DIAGNOSTICS_TOP_MESSAGES_TABLE_KEY = "analytics.scip_diagnostics_top_messages"


@dataclass(frozen=True, slots=True)
class ScipDiagnosticsRollups:
    """Computed rollup rows for SCIP diagnostics outputs."""

    summary_rows: list[dict[str, object]]
    by_file_rows: list[dict[str, object]]
    top_message_rows: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class _ScipDiagnosticsCounts:
    summary_counts: dict[tuple[str, str], int]
    file_counts: dict[tuple[str, str, str], int]
    message_counts: dict[tuple[str, str, str, str], int]


def build_scip_diagnostics_rollups(
    *,
    repo: str,
    commit: str,
    rows: Sequence[Mapping[str, object]],
) -> ScipDiagnosticsRollups:
    """Build rollup rows for SCIP diagnostics datasets.

    Returns
    -------
    ScipDiagnosticsRollups
        Rollup rows for summary, by-file, and top-message datasets.
    """
    if not rows:
        return ScipDiagnosticsRollups([], [], [])
    counts = _collect_diagnostics_counts(rows)
    created_at = datetime.now(tz=UTC)
    return ScipDiagnosticsRollups(
        summary_rows=_summary_rows(repo, commit, counts, created_at),
        by_file_rows=_by_file_rows(repo, commit, counts, created_at),
        top_message_rows=_top_message_rows(repo, commit, counts, created_at),
    )


def _normalize_text(value: object | None, *, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _collect_diagnostics_counts(
    rows: Sequence[Mapping[str, object]],
) -> _ScipDiagnosticsCounts:
    summary_counts: dict[tuple[str, str], int] = {}
    file_counts: dict[tuple[str, str, str], int] = {}
    message_counts: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        severity = _normalize_text(row.get("severity"))
        source = _normalize_text(row.get("source"))
        code = _normalize_text(row.get("code"))
        message = _normalize_text(row.get("message"))
        rel_path = _normalize_text(row.get("rel_path"))
        summary_counts[severity, source] = summary_counts.get((severity, source), 0) + 1
        file_counts[rel_path, severity, source] = (
            file_counts.get((rel_path, severity, source), 0) + 1
        )
        message_counts[severity, source, code, message] = (
            message_counts.get((severity, source, code, message), 0) + 1
        )
    return _ScipDiagnosticsCounts(
        summary_counts=summary_counts,
        file_counts=file_counts,
        message_counts=message_counts,
    )


def _summary_rows(
    repo: str,
    commit: str,
    counts: _ScipDiagnosticsCounts,
    created_at: datetime,
) -> list[dict[str, object]]:
    return [
        {
            "repo": repo,
            "commit": commit,
            "severity": severity,
            "source": source,
            "diagnostic_count": count,
            "created_at": created_at,
        }
        for (severity, source), count in sorted(counts.summary_counts.items())
    ]


def _by_file_rows(
    repo: str,
    commit: str,
    counts: _ScipDiagnosticsCounts,
    created_at: datetime,
) -> list[dict[str, object]]:
    return [
        {
            "repo": repo,
            "commit": commit,
            "rel_path": rel_path,
            "severity": severity,
            "source": source,
            "diagnostic_count": count,
            "created_at": created_at,
        }
        for (rel_path, severity, source), count in sorted(counts.file_counts.items())
    ]


def _top_message_rows(
    repo: str,
    commit: str,
    counts: _ScipDiagnosticsCounts,
    created_at: datetime,
) -> list[dict[str, object]]:
    return [
        {
            "repo": repo,
            "commit": commit,
            "severity": severity,
            "source": source,
            "code": code,
            "message": message,
            "diagnostic_count": count,
            "created_at": created_at,
        }
        for (severity, source, code, message), count in sorted(counts.message_counts.items())
    ]


__all__ = [
    "SCIP_DIAGNOSTICS_BY_FILE_TABLE_KEY",
    "SCIP_DIAGNOSTICS_SUMMARY_TABLE_KEY",
    "SCIP_DIAGNOSTICS_TABLE_KEY",
    "SCIP_DIAGNOSTICS_TOP_MESSAGES_TABLE_KEY",
    "ScipDiagnosticsRollups",
    "build_scip_diagnostics_rollups",
]
