"""SCIP diagnostics analytics rollups."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.contracts.ref import contract_ref_for_table
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    MultiTableTargetContext,
    TableTargetTableContext,
    attach_table_target_template,
    build_multi_table_target_spec_from_contexts,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.scoping import collect_scoped_rows
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCIP_DIAGNOSTICS_TARGET_NAME = "scip_diagnostics"
SCIP_DIAGNOSTICS_SUMMARY_TABLE_KEY = "analytics.scip_diagnostics_summary"
SCIP_DIAGNOSTICS_BY_FILE_TABLE_KEY = "analytics.scip_diagnostics_by_file"
SCIP_DIAGNOSTICS_TOP_MESSAGES_TABLE_KEY = "analytics.scip_diagnostics_top_messages"

SCIP_DIAGNOSTICS_SUMMARY_CONTRACT = contract_ref_for_table(
    table_key=SCIP_DIAGNOSTICS_SUMMARY_TABLE_KEY,
    target_name=SCIP_DIAGNOSTICS_TARGET_NAME,
    input_name="scip_diagnostics_summary__base",
    required_cols=(),
    clip_column=None,
)
SCIP_DIAGNOSTICS_BY_FILE_CONTRACT = contract_ref_for_table(
    table_key=SCIP_DIAGNOSTICS_BY_FILE_TABLE_KEY,
    target_name=SCIP_DIAGNOSTICS_TARGET_NAME,
    input_name="scip_diagnostics_by_file__base",
    required_cols=(),
    clip_column=None,
)
SCIP_DIAGNOSTICS_TOP_MESSAGES_CONTRACT = contract_ref_for_table(
    table_key=SCIP_DIAGNOSTICS_TOP_MESSAGES_TABLE_KEY,
    target_name=SCIP_DIAGNOSTICS_TARGET_NAME,
    input_name="scip_diagnostics_top_messages__base",
    required_cols=(),
    clip_column=None,
)


@dataclass(frozen=True, slots=True)
class ScipDiagnosticsRollups:
    summary_rows: Sequence[Mapping[str, object]]
    by_file_rows: Sequence[Mapping[str, object]]
    top_message_rows: Sequence[Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class _ScipDiagnosticsCounts:
    summary_counts: dict[tuple[str, str], int]
    file_counts: dict[tuple[str, str, str], int]
    message_counts: dict[tuple[str, str, str, str], int]


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
            file_counts.get(
                (rel_path, severity, source),
                0,
            )
            + 1
        )
        message_counts[severity, source, code, message] = (
            message_counts.get(
                (severity, source, code, message),
                0,
            )
            + 1
        )
    return _ScipDiagnosticsCounts(
        summary_counts=summary_counts,
        file_counts=file_counts,
        message_counts=message_counts,
    )


def _summary_rows(
    env: BuildEnv,
    counts: _ScipDiagnosticsCounts,
    created_at: datetime,
) -> list[dict[str, object]]:
    return [
        {
            "repo": env.repo,
            "commit": env.commit,
            "severity": severity,
            "source": source,
            "diagnostic_count": count,
            "created_at": created_at,
        }
        for (severity, source), count in sorted(counts.summary_counts.items())
    ]


def _by_file_rows(
    env: BuildEnv,
    counts: _ScipDiagnosticsCounts,
    created_at: datetime,
) -> list[dict[str, object]]:
    return [
        {
            "repo": env.repo,
            "commit": env.commit,
            "rel_path": rel_path,
            "severity": severity,
            "source": source,
            "diagnostic_count": count,
            "created_at": created_at,
        }
        for (rel_path, severity, source), count in sorted(counts.file_counts.items())
    ]


def _top_message_rows(
    env: BuildEnv,
    counts: _ScipDiagnosticsCounts,
    created_at: datetime,
) -> list[dict[str, object]]:
    return [
        {
            "repo": env.repo,
            "commit": env.commit,
            "severity": severity,
            "source": source,
            "code": code,
            "message": message,
            "diagnostic_count": count,
            "created_at": created_at,
        }
        for (severity, source, code, message), count in sorted(counts.message_counts.items())
    ]


def scip_diagnostics__rollups(
    env: BuildEnv,
    q__core__scip_diagnostics: InferableTabularInput,
) -> ScipDiagnosticsRollups:
    """Compute diagnostics rollups from core.scip_diagnostics rows.

    Returns
    -------
    ScipDiagnosticsRollups
        Rollup row collections for summary, by-file, and top-message tables.
    """
    scope = SnapshotScope.from_snapshot(env.snapshot)
    rows = collect_scoped_rows(
        q__core__scip_diagnostics,
        (
            "repo",
            "commit",
            "rel_path",
            "severity",
            "source",
            "code",
            "message",
        ),
        scope=scope,
    )
    if not rows:
        return ScipDiagnosticsRollups((), (), ())
    counts = _collect_diagnostics_counts(rows)
    created_at = datetime.now(tz=UTC)
    summary_rows = _summary_rows(env, counts, created_at)
    by_file_rows = _by_file_rows(env, counts, created_at)
    top_message_rows = _top_message_rows(env, counts, created_at)
    return ScipDiagnosticsRollups(summary_rows, by_file_rows, top_message_rows)


def scip_diagnostics_summary__base(
    scip_diagnostics__rollups: ScipDiagnosticsRollups,
) -> pa.Table:
    """Return rows for analytics.scip_diagnostics_summary.

    Returns
    -------
    pa.Table
        Table data for the summary rollup.
    """
    if not scip_diagnostics__rollups.summary_rows:
        return empty_table_for_table(SCIP_DIAGNOSTICS_SUMMARY_TABLE_KEY)
    reader, _ = table_for_rows(
        SCIP_DIAGNOSTICS_SUMMARY_TABLE_KEY,
        scip_diagnostics__rollups.summary_rows,
    )
    return reader


def scip_diagnostics_by_file__base(
    scip_diagnostics__rollups: ScipDiagnosticsRollups,
) -> pa.Table:
    """Return rows for analytics.scip_diagnostics_by_file.

    Returns
    -------
    pa.Table
        Table data for the by-file rollup.
    """
    if not scip_diagnostics__rollups.by_file_rows:
        return empty_table_for_table(SCIP_DIAGNOSTICS_BY_FILE_TABLE_KEY)
    reader, _ = table_for_rows(
        SCIP_DIAGNOSTICS_BY_FILE_TABLE_KEY,
        scip_diagnostics__rollups.by_file_rows,
    )
    return reader


def scip_diagnostics_top_messages__base(
    scip_diagnostics__rollups: ScipDiagnosticsRollups,
) -> pa.Table:
    """Return rows for analytics.scip_diagnostics_top_messages.

    Returns
    -------
    pa.Table
        Table data for the top-message rollup.
    """
    if not scip_diagnostics__rollups.top_message_rows:
        return empty_table_for_table(SCIP_DIAGNOSTICS_TOP_MESSAGES_TABLE_KEY)
    reader, _ = table_for_rows(
        SCIP_DIAGNOSTICS_TOP_MESSAGES_TABLE_KEY,
        scip_diagnostics__rollups.top_message_rows,
    )
    return reader


_MODULE = sys.modules[__name__]
_SCIP_DIAGNOSTICS_TABLE_CONTEXTS = (
    TableTargetTableContext.from_contract_ref(
        contract_ref=SCIP_DIAGNOSTICS_SUMMARY_CONTRACT,
        node_name="scip_diagnostics_summary__table",
    ),
    TableTargetTableContext.from_contract_ref(
        contract_ref=SCIP_DIAGNOSTICS_BY_FILE_CONTRACT,
        node_name="scip_diagnostics_by_file__table",
    ),
    TableTargetTableContext.from_contract_ref(
        contract_ref=SCIP_DIAGNOSTICS_TOP_MESSAGES_CONTRACT,
        node_name="scip_diagnostics_top_messages__table",
    ),
)
_SCIP_DIAGNOSTICS_TABLE_TARGET_SPEC = build_multi_table_target_spec_from_contexts(
    context=MultiTableTargetContext(
        domain="analytics",
        target_name=SCIP_DIAGNOSTICS_TARGET_NAME,
        tables=(),
        table_materializations_node="scip_diagnostics__table_materializations",
        anchor_node_name="t__scip_diagnostics",
        default_input_type=pa.Table,
    ),
    table_contexts=_SCIP_DIAGNOSTICS_TABLE_CONTEXTS,
)
attach_table_target_template(_MODULE, spec=_SCIP_DIAGNOSTICS_TABLE_TARGET_SPEC)
scip_diagnostics_summary__table = _MODULE.scip_diagnostics_summary__table
scip_diagnostics_by_file__table = _MODULE.scip_diagnostics_by_file__table
scip_diagnostics_top_messages__table = _MODULE.scip_diagnostics_top_messages__table
scip_diagnostics__table_materializations = _MODULE.scip_diagnostics__table_materializations
t__scip_diagnostics = _MODULE.t__scip_diagnostics


__all__ = [
    "scip_diagnostics__rollups",
    "scip_diagnostics__table_materializations",
    "scip_diagnostics_by_file__base",
    "scip_diagnostics_by_file__table",
    "scip_diagnostics_summary__base",
    "scip_diagnostics_summary__table",
    "scip_diagnostics_top_messages__base",
    "scip_diagnostics_top_messages__table",
    "t__scip_diagnostics",
]
