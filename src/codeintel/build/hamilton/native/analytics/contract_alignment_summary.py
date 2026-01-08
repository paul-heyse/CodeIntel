"""Contract alignment diagnostics summary table."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pyarrow as pa

from codeintel.build.contracts.ref import contract_ref_for_table
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    TableTargetContext,
    attach_table_target_template,
    build_single_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.scoping import collect_scoped_rows
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.execution.ids import RUN_PREFIX_ANALYTICS, new_run_id
from codeintel.core.query_results import coerce_optional_int

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

CONTRACT_ALIGNMENT_SUMMARY_TARGET_NAME = "contract_alignment_summary"
CONTRACT_ALIGNMENT_SUMMARY_TABLE_KEY = "analytics.contract_alignment_summary"
CONTRACT_ALIGNMENT_SUMMARY_CONTRACT = contract_ref_for_table(
    table_key=CONTRACT_ALIGNMENT_SUMMARY_TABLE_KEY,
    target_name=CONTRACT_ALIGNMENT_SUMMARY_TARGET_NAME,
    input_name="contract_alignment_summary__base",
    required_cols=(),
    clip_column=None,
)


@dataclass(slots=True)
class _AlignmentSummary:
    target_names: set[str] = field(default_factory=set)
    table_keys: set[str] = field(default_factory=set)
    issue_count: int = 0
    missing_total: int = 0
    extra_total: int = 0
    coerced_total: int = 0


def _coerce_count(value: object | None, *, ctx: str) -> int:
    if value is None:
        return 0
    coerced = coerce_optional_int(value, ctx=ctx)
    if coerced is None:
        return 0
    return coerced


def _resolve_run_id(env: BuildEnv) -> str:
    run_context = env.run_context
    if run_context is None:
        return new_run_id(RUN_PREFIX_ANALYTICS)
    return run_context.run_id


def contract_alignment_summary__base(
    env: BuildEnv,
    q__build__contract_alignment_issues: InferableTabularInput,
) -> pa.Table:
    """Aggregate contract alignment issues into run-level counts.

    Returns
    -------
    pa.Table
        Reader with contract alignment summary rows.
    """
    scope = SnapshotScope.from_snapshot(env.snapshot)
    rows = collect_scoped_rows(
        q__build__contract_alignment_issues,
        (
            "repo",
            "commit",
            "run_id",
            "target_name",
            "table_key",
            "missing_count",
            "extra_count",
            "coerced_count",
        ),
        scope=scope,
    )
    if not rows:
        created_at = datetime.now(tz=UTC)
        output_rows = [
            {
                "repo": env.repo,
                "commit": env.commit,
                "run_id": _resolve_run_id(env),
                "issue_count": 0,
                "target_count": 0,
                "table_count": 0,
                "missing_total": 0,
                "extra_total": 0,
                "coerced_total": 0,
                "created_at": created_at,
            }
        ]
        reader, _ = table_for_rows(CONTRACT_ALIGNMENT_SUMMARY_TABLE_KEY, output_rows)
        return reader

    summaries: dict[str, _AlignmentSummary] = {}
    for row in rows:
        run_id_value = row.get("run_id")
        if run_id_value is None:
            continue
        run_id = str(run_id_value)
        summary = summaries.get(run_id)
        if summary is None:
            summary = _AlignmentSummary()
            summaries[run_id] = summary
        summary.issue_count += 1
        target_name = row.get("target_name")
        if target_name is not None:
            summary.target_names.add(str(target_name))
        table_key = row.get("table_key")
        if table_key is not None:
            summary.table_keys.add(str(table_key))
        summary.missing_total += _coerce_count(row.get("missing_count"), ctx="missing_count")
        summary.extra_total += _coerce_count(row.get("extra_count"), ctx="extra_count")
        summary.coerced_total += _coerce_count(row.get("coerced_count"), ctx="coerced_count")

    if not summaries:
        return empty_table_for_table(CONTRACT_ALIGNMENT_SUMMARY_TABLE_KEY)

    created_at = datetime.now(tz=UTC)
    output_rows = [
        {
            "repo": env.repo,
            "commit": env.commit,
            "run_id": run_id,
            "issue_count": summary.issue_count,
            "target_count": len(summary.target_names),
            "table_count": len(summary.table_keys),
            "missing_total": summary.missing_total,
            "extra_total": summary.extra_total,
            "coerced_total": summary.coerced_total,
            "created_at": created_at,
        }
        for run_id, summary in sorted(summaries.items())
    ]
    reader, _ = table_for_rows(CONTRACT_ALIGNMENT_SUMMARY_TABLE_KEY, output_rows)
    return reader


_MODULE = sys.modules[__name__]
_CONTRACT_ALIGNMENT_SUMMARY_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext.from_contract_ref(
        contract_ref=CONTRACT_ALIGNMENT_SUMMARY_CONTRACT,
        input_type=pa.Table,
    )
)
attach_table_target_template(_MODULE, spec=_CONTRACT_ALIGNMENT_SUMMARY_TABLE_TARGET_SPEC)
contract_alignment_summary__table = _MODULE.contract_alignment_summary__table
contract_alignment_summary__table_materializations = (
    _MODULE.contract_alignment_summary__table_materializations
)
t__contract_alignment_summary = _MODULE.t__contract_alignment_summary


__all__ = [
    "contract_alignment_summary__base",
    "contract_alignment_summary__table",
    "t__contract_alignment_summary",
]
