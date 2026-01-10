"""Tabular helpers for building function catalog providers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import pyarrow as pa

from codeintel.build.analytics.utilities.snapshot import (
    SnapshotContext,
    snapshot_plan,
    snapshot_table,
)
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.finalize_ops import finalize_reader, finalize_spec_for_table
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_columnar_context,
    resolve_execution_context,
)
from codeintel.core.execution.context import ExecutionContext as RuntimeExecutionContext
from codeintel.storage.catalog import CatalogService, build_function_catalog_from_rows

_FUNCTION_KINDS = {"function", "method"}
_INTERNAL_PLAN_TABLE_KEY = "internal.plan_materialize"
type RowSource = pa.Table | pa.RecordBatchReader


@dataclass(frozen=True, slots=True)
class CatalogScope:
    """Snapshot scope details for catalog helpers."""

    repo: str | None = None
    commit: str | None = None
    ctx: ExecutionContext | RuntimeExecutionContext | None = None


@dataclass(frozen=True, slots=True)
class CatalogProviderRequest:
    """Inputs required to build a catalog provider."""

    goids_frame: RowSource
    modules_frame: RowSource
    module_map_override: Mapping[str, str] | None = None
    scope: CatalogScope | None = None


@dataclass(frozen=True, slots=True)
class SnapshotSourceRequest:
    """Inputs for snapshotting an in-memory table source."""

    repo: str | None
    commit: str | None
    columns: Sequence[str]
    table_key: str | None
    ctx: ExecutionContext | RuntimeExecutionContext | None


def module_map_from_frame(
    modules_frame: RowSource,
    scope: CatalogScope | None = None,
) -> dict[str, str]:
    """Build module mapping from core.modules frame.

    Returns
    -------
    dict[str, str]
        Mapping of file path to module name.
    """
    module_map: dict[str, str] = {}
    resolved_scope = scope or CatalogScope()
    source = _snapshot_source(
        modules_frame,
        request=SnapshotSourceRequest(
            repo=resolved_scope.repo,
            commit=resolved_scope.commit,
            columns=("path", "module"),
            table_key="core.modules",
            ctx=resolved_scope.ctx,
        ),
    )
    for row in _iter_rows_from_source(source):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_map[path] = module
    return module_map


def catalog_provider_from_frames(
    request: CatalogProviderRequest,
) -> CatalogService:
    """Build a CatalogService from goids and modules frames.

    Returns
    -------
    CatalogService
        Catalog provider backed by the input frames.
    """
    resolved_scope = request.scope or CatalogScope()
    module_map = dict(
        request.module_map_override
        or module_map_from_frame(request.modules_frame, scope=resolved_scope)
    )
    rows: list[dict[str, object]] = []
    source = _goids_source(
        request.goids_frame,
        repo=resolved_scope.repo,
        commit=resolved_scope.commit,
        ctx=resolved_scope.ctx,
    )
    for row in _iter_rows_from_source(source):
        kind = row.get("kind")
        if kind is not None and str(kind) not in _FUNCTION_KINDS:
            continue
        rows.append(
            {
                "goid_h128": row.get("goid_h128"),
                "rel_path": row.get("rel_path"),
                "qualname": row.get("qualname"),
                "start_line": row.get("start_line"),
                "end_line": row.get("end_line"),
                "urn": row.get("urn"),
            }
        )
    catalog = build_function_catalog_from_rows(rows, module_by_path=module_map)
    return CatalogService(catalog)


def _iter_rows_from_source(source: RowSource) -> Iterable[dict[str, object]]:
    if isinstance(source, pa.Table):
        yield from iter_rows(source)
        return
    for batch in source:
        yield from iter_rows(batch)


def _snapshot_source(
    source: RowSource,
    *,
    request: SnapshotSourceRequest,
) -> RowSource:
    if not isinstance(source, pa.Table):
        return source
    if not set(request.columns).issubset(source.column_names):
        return source
    return snapshot_table(
        source,
        columns=request.columns,
        context=SnapshotContext(
            repo=request.repo,
            commit=request.commit,
            ctx=request.ctx,
            table_key=request.table_key,
        ),
    )


def _goids_source(
    source: RowSource,
    *,
    repo: str | None,
    commit: str | None,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> RowSource:
    if not isinstance(source, pa.Table):
        return source
    required = (
        "goid_h128",
        "rel_path",
        "qualname",
        "start_line",
        "end_line",
        "urn",
        "kind",
    )
    if not set(required).issubset(source.column_names):
        return source
    plan = snapshot_plan(
        source,
        columns=required,
        context=SnapshotContext(
            repo=repo,
            commit=commit,
            ctx=ctx,
            table_key="core.goids",
        ),
    )
    plan = plan.filter(E.in_("kind", tuple(_FUNCTION_KINDS)))
    execution_ctx = resolve_execution_context(resolve_columnar_context(ctx))
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    result = finalize_reader(
        reader,
        spec=finalize_spec_for_table(
            _INTERNAL_PLAN_TABLE_KEY,
            mode="tolerant",
            ordering=plan.ordering,
        ),
    )
    return result.good


__all__ = [
    "CatalogProviderRequest",
    "CatalogScope",
    "catalog_provider_from_frames",
    "module_map_from_frame",
]
