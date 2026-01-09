"""Ingestion plan surface for QuerySpec-driven Acero execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.execution_context import ExecutionContext, resolve_execution_context
from codeintel.core.columnar.plan_builder import build_plan_from_query_spec
from codeintel.core.columnar.plan_ops import (
    Plan,
    QueryPlanOptions,
    build_query_plan_for_context,
    query_plan_options_for_context,
)
from codeintel.core.columnar.queryspec import QuerySpec
from codeintel.ingestion.compute.queryspecs import build_ingest_query_spec


@dataclass(frozen=True, slots=True)
class IngestQuery:
    """Query inputs for ingestion plans."""

    table_key: str
    columns: Sequence[str] | None = None
    repo: str | None = None
    commit: str | None = None
    rel_path: str | None = None

    def to_query_spec(
        self,
        *,
        available_columns: Sequence[str] | None = None,
    ) -> QuerySpec:
        """Build a QuerySpec for this ingestion query.

        Returns
        -------
        QuerySpec
            Query specification built from the ingestion query.
        """
        return build_ingest_query_spec(
            self.table_key,
            columns=self.columns,
            repo=self.repo,
            commit=self.commit,
            rel_path=self.rel_path,
            available_columns=available_columns,
        )


def ingest_plan_for_table(
    table: pa.Table,
    *,
    query: IngestQuery,
    ctx: ExecutionContext | None = None,
    options: QueryPlanOptions | None = None,
) -> Plan:
    """Build an ingestion plan for an in-memory table.

    Parameters
    ----------
    table
        Input Arrow table.
    query
        Query settings for table scoping and projection.
    ctx
        Optional execution context for runtime profile defaults.
    options
        Optional query plan options overriding defaults.

    Returns
    -------
    Plan
        Plan with filter/project nodes derived from QuerySpec.
    """
    resolved_ctx = resolve_execution_context(ctx)
    spec = query.to_query_spec(available_columns=table.column_names)
    resolved = query_plan_options_for_context(ctx=resolved_ctx, options=options)
    plan = build_plan_from_query_spec(
        table=table,
        spec=spec,
        ctx=resolved_ctx,
    )
    if resolved.order_by is not None:
        plan = plan.order_by(sort_keys=resolved.order_by)
    return plan


def ingest_plan_for_dataset(
    dataset: ds.Dataset,
    *,
    query: IngestQuery,
    ctx: ExecutionContext | None = None,
    options: QueryPlanOptions | None = None,
) -> Plan:
    """Build an ingestion plan for a dataset scan.

    Parameters
    ----------
    dataset
        Dataset to scan.
    query
        Query settings for dataset scoping and projection.
    ctx
        Optional execution context for runtime profile defaults.
    options
        Optional query plan options overriding defaults.

    Returns
    -------
    Plan
        Plan compiled from a QuerySpec and execution context.
    """
    resolved_ctx = resolve_execution_context(ctx)
    spec = query.to_query_spec(available_columns=dataset.schema.names)
    return build_query_plan_for_context(dataset, spec=spec, ctx=resolved_ctx, options=options)


def ingest_reader_for_plan(
    plan: Plan,
    *,
    ctx: ExecutionContext | None = None,
) -> pa.RecordBatchReader:
    """Return a reader for an ingestion plan.

    Parameters
    ----------
    plan
        Ingestion plan to execute.
    ctx
        Optional execution context for runtime profile defaults.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader for the plan output.
    """
    resolved_ctx = resolve_execution_context(ctx)
    return ExecutionPlan.from_plan(plan).to_reader(ctx=resolved_ctx)


def ingest_reader_for_table(
    table: pa.Table,
    *,
    query: IngestQuery,
    ctx: ExecutionContext | None = None,
    options: QueryPlanOptions | None = None,
) -> pa.RecordBatchReader:
    """Return a reader for an ingestion table plan.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader for the plan output.
    """
    plan = ingest_plan_for_table(
        table,
        query=query,
        ctx=ctx,
        options=options,
    )
    return ingest_reader_for_plan(plan, ctx=ctx)


def ingest_reader_for_dataset(
    dataset: ds.Dataset,
    *,
    query: IngestQuery,
    ctx: ExecutionContext | None = None,
    options: QueryPlanOptions | None = None,
) -> pa.RecordBatchReader:
    """Return a reader for an ingestion dataset plan.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader for the plan output.
    """
    plan = ingest_plan_for_dataset(
        dataset,
        query=query,
        ctx=ctx,
        options=options,
    )
    return ingest_reader_for_plan(plan, ctx=ctx)


__all__ = [
    "IngestQuery",
    "ingest_plan_for_dataset",
    "ingest_plan_for_table",
    "ingest_reader_for_dataset",
    "ingest_reader_for_plan",
    "ingest_reader_for_table",
]
