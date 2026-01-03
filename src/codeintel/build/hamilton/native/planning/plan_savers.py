"""Saver nodes for plan/explain outputs."""

from __future__ import annotations

import json
from collections.abc import Iterable

import polars as pl

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns.savers import (
    ArtifactSaveSpec,
    RelationTableSaveSpec,
    SaverContext,
    save_artifact,
    save_relation_table,
)
from codeintel.build.hamilton.native.planning.plan_targets import (
    CI_PLAN_EXPLAIN_ARTIFACT,
    CI_PLAN_JSON_ARTIFACT,
    CI_PLAN_TARGET_NAME,
    PLAN_DOMAIN,
)
from codeintel.build.planning.model import PLAN_SCHEMA_VERSION, BuildPlan
from codeintel.build.tabular.frames import lazyframe_for_table_columns
from codeintel.core.columnar.rows import columnar_buffer_for_table_key
from codeintel.core.schemas.tables.ci_plan_entries import CI_PLAN_ENTRIES_TABLE_KEY


def _plan_row_mappings(
    *,
    plan: BuildPlan,
    run_id: str,
) -> Iterable[dict[str, object]]:
    for entry in plan.entries:
        yield {
            "run_id": run_id,
            "created_at_utc": plan.created_at_utc,
            "requested_targets": list(plan.request.requested_targets),
            "target": entry.target,
            "domain": entry.domain,
            "action": entry.predicted_action,
            "cache_hit_ratio": entry.cache_hit_ratio,
            "block_reasons": list(entry.block_reasons),
            "miss_nodes": list(entry.miss_nodes),
            "reads": list(entry.reads),
            "writes_tables": list(entry.writes_tables),
            "writes_artifacts": list(entry.writes_artifacts),
            "build_fingerprint": plan.build_fingerprint,
            "plan_schema_version": PLAN_SCHEMA_VERSION,
        }


def _plan_markdown(plan: BuildPlan) -> str:
    action_counts = {
        "reuse": 0,
        "compute": 0,
        "blocked": 0,
    }
    for entry in plan.entries:
        action_counts[entry.predicted_action] += 1

    lines = [
        "# Build Plan",
        "",
        f"- Created at: {plan.created_at_utc}",
        f"- Requested targets: {', '.join(plan.request.requested_targets) or 'none'}",
        f"- Closure size: {len(plan.closure)}",
        "- Predicted actions:",
        f"  - reuse: {action_counts['reuse']}",
        f"  - compute: {action_counts['compute']}",
        f"  - blocked: {action_counts['blocked']}",
        "",
        "## Targets",
        "",
        "| target | domain | action | cache_hit_ratio | block_reasons |",
        "| --- | --- | --- | --- | --- |",
    ]

    for entry in plan.entries:
        ratio = ""
        if entry.cache_hit_ratio is not None:
            ratio = f"{entry.cache_hit_ratio:.2f}"
        reasons = ", ".join(entry.block_reasons)
        lines.append(
            f"| {entry.target} | {entry.domain} | {entry.predicted_action} | {ratio} | {reasons} |"
        )

    return "\n".join(lines) + "\n"


@save_artifact(
    context=SaverContext(domain=PLAN_DOMAIN, target=CI_PLAN_TARGET_NAME),
    spec=ArtifactSaveSpec(
        artifact_name=CI_PLAN_JSON_ARTIFACT,
        path_template="{build_dir}/plans/ci_plan.json",
        output_role="contract",
    ),
)
def m__ci_plan_json(plan: BuildPlan) -> str:
    """Serialize the plan as JSON.

    Returns
    -------
    str
        Newline-terminated JSON payload.
    """
    return json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n"


@save_artifact(
    context=SaverContext(domain=PLAN_DOMAIN, target=CI_PLAN_TARGET_NAME),
    spec=ArtifactSaveSpec(
        artifact_name=CI_PLAN_EXPLAIN_ARTIFACT,
        path_template="{build_dir}/plans/ci_plan.explain.md",
        output_role="contract",
    ),
)
def m__ci_plan_explain_md(plan: BuildPlan) -> str:
    """Render a human-readable plan summary.

    Returns
    -------
    str
        Markdown summary of the plan.
    """
    return _plan_markdown(plan)


@save_relation_table(
    context=SaverContext(domain=PLAN_DOMAIN, target=CI_PLAN_TARGET_NAME),
    spec=RelationTableSaveSpec(
        table_key=CI_PLAN_ENTRIES_TABLE_KEY,
        output_role="contract",
    ),
)
def m__ci_plan_entries(env: BuildEnv, plan: BuildPlan) -> pl.LazyFrame:
    """Materialize plan entries as a lazy frame.

    Returns
    -------
    pl.LazyFrame
        Lazy frame for plan entries.
    """
    run_context = env.run_context
    run_id = run_context.run_id if run_context is not None else "unknown"
    buffer = columnar_buffer_for_table_key(CI_PLAN_ENTRIES_TABLE_KEY)
    for row in _plan_row_mappings(plan=plan, run_id=run_id):
        buffer.append(row)
    return lazyframe_for_table_columns(CI_PLAN_ENTRIES_TABLE_KEY, buffer.data)


__all__ = [
    "m__ci_plan_entries",
    "m__ci_plan_explain_md",
    "m__ci_plan_json",
]
