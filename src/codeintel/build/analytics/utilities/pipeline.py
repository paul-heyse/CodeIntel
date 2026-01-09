"""Shared pipeline runner for analytics QuerySpec execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_columnar_context,
)
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table
from codeintel.core.columnar.plan_builder import (
    SchemaPlanDefaultsRequest,
    build_plan_from_query_spec,
    plan_from_schema_defaults,
)
from codeintel.core.columnar.plan_ops import QueryPlanOptions
from codeintel.core.columnar.queryspec import QuerySpec
from codeintel.core.columnar.run_manifest import (
    RunManifestOptions,
    run_manifest_options_for_context,
)
from codeintel.core.columnar.streaming import scan_telemetry_for_queryspec
from codeintel.core.execution.context import ExecutionContext as RuntimeExecutionContext
from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    from codeintel.build.tabular.finalize_ops import FinalizeResult

type QuerySource = ds.Dataset | pa.Table


@dataclass(frozen=True, slots=True)
class AnalyticsPipelineRunRequest:
    """Inputs required to execute an analytics pipeline."""

    source: QuerySource
    spec: QuerySpec
    table_key: str
    ctx: ExecutionContext | RuntimeExecutionContext
    options: QueryPlanOptions | None = None
    manifest_dir: Path | None = None
    manifest_options: RunManifestOptions | None = None


def run_analytics_pipeline(request: AnalyticsPipelineRunRequest) -> FinalizeResult:
    """Execute a QuerySpec and finalize results for analytics outputs.

    Returns
    -------
    FinalizeResult
        Finalize artifacts for the table key.
    """
    scan_telemetry = None
    resolved_ctx = resolve_columnar_context(request.ctx)
    if isinstance(request.source, ds.Dataset):
        plan = plan_from_schema_defaults(
            schema_service=get_schema_service(),
            request=SchemaPlanDefaultsRequest(
                table_key=request.table_key,
                dataset=request.source,
                predicate=request.spec.predicate,
                columns=request.spec.scan_columns(provenance=False),
                options=request.options,
                ctx=resolved_ctx,
            ),
        )
        scan_telemetry = scan_telemetry_for_queryspec(request.source, spec=request.spec)
    else:
        plan = build_plan_from_query_spec(
            table=request.source,
            spec=request.spec,
            ctx=resolved_ctx,
        )
    finalize = finalize_spec_for_table(
        request.table_key,
        mode="tolerant",
        emit_artifacts=True,
    )
    pipeline_options = PipelineRunOptions(
        ctx=resolved_ctx,
        manifest_dir=request.manifest_dir,
        manifest_options=run_manifest_options_for_context(
            ctx=resolved_ctx,
            ordering=plan.ordering,
            scan_telemetry=scan_telemetry,
            options=request.manifest_options,
        ),
        scan_telemetry=scan_telemetry,
    )
    return run_pipeline(
        plan=ExecutionPlan.from_plan(plan),
        finalize=finalize,
        options=pipeline_options,
    )


__all__ = [
    "AnalyticsPipelineRunRequest",
    "QuerySource",
    "run_analytics_pipeline",
]
