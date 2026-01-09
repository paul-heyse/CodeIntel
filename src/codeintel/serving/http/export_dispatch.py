"""HTTP export dispatch helpers.

This module centralizes the mapping from export format -> response builder so
HTTP route handlers stay thin and consistent.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anyio
from anyio import to_thread
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from starlette.responses import FileResponse

from codeintel.core.columnar.arrowdsl import ExecutionContext
from codeintel.core.columnar.finalize_ops import FinalizeDedupe, finalize_spec_for_table
from codeintel.core.columnar.iter import iter_rows
from codeintel.core.columnar.kernels import SortKey
from codeintel.core.constants import DEFAULT_ARROW_PROVENANCE_COLUMNS
from codeintel.serving.export.engine import (
    ExportDelivery,
    ExportPlan,
    build_export_plan,
    write_export_file,
)
from codeintel.serving.http.streaming import (
    NdjsonBatchResponseOptions,
    ndjson_response_from_batches,
)
from codeintel.serving.metrics import QueryMetrics, log_query_metrics
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.models import SemanticExportRequest, SemanticViewDescriptionResponse

if TYPE_CHECKING:
    from collections.abc import Callable

    from starlette.responses import Response

    from codeintel.core.columnar.finalize_ops import FinalizeResult
    from codeintel.serving.operations.cancellation import CancelCheck


@dataclass(frozen=True, slots=True)
class ExportDispatchResult:
    """Result of dispatching an export request for HTTP."""

    response: Response
    metrics_row_count: int | None


LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExportMetricsContext:
    """Context required to emit export metrics."""

    view_id: str
    correlation_id: str
    query_hash: str
    schema_hash: str | None
    ast_fingerprint: str | None = None
    sql_fingerprint: str | None = None

    def to_metrics(self, *, row_count: int, duration_ms: float) -> QueryMetrics:
        """Build QueryMetrics record for the export response.

        Returns
        -------
        QueryMetrics
            Structured metrics payload for logging.
        """
        return QueryMetrics(
            endpoint="/v1/export/semantic",
            view_id=self.view_id,
            query=None,
            row_count=row_count,
            truncated=False,
            duration_ms=duration_ms,
            correlation_id=self.correlation_id,
            query_hash=self.query_hash,
            schema_hash=self.schema_hash,
        )


@dataclass(slots=True)
class _ExportStreamMetrics:
    metrics: ExportMetricsContext
    started: float = field(default_factory=time.perf_counter)
    row_count: int = 0

    def record_rows(self, count: int) -> None:
        self.row_count += count

    def finalize(self) -> None:
        duration_ms = (time.perf_counter() - self.started) * 1000
        log_query_metrics(
            self.metrics.to_metrics(row_count=self.row_count, duration_ms=duration_ms)
        )


@dataclass(frozen=True, slots=True)
class ExportDispatchOptions:
    """Options for dispatching export responses."""

    headers: dict[str, str]
    cancel_check: CancelCheck | None = None
    timeout_s: float | None = None


def _finalize_mode(schema_enforcement: str) -> Literal["strict", "tolerant"]:
    return "strict" if schema_enforcement == "strict" else "tolerant"


def _finalize_log_hook(table_key: str) -> Callable[[FinalizeResult], None]:
    logged_alignment = False
    logged_errors = False

    def _hook(result: FinalizeResult) -> None:
        nonlocal logged_alignment, logged_errors
        if result.stats.num_rows and not logged_errors:
            LOG.warning(
                "NDJSON finalize errors for %s: %s",
                table_key,
                list(iter_rows(result.stats)),
            )
            sample = next(iter_rows(result.errors), None)
            if sample is not None:
                context = {
                    name: sample.get(name)
                    for name in DEFAULT_ARROW_PROVENANCE_COLUMNS
                    if name in sample
                }
                details = {
                    key: sample.get(key)
                    for key in ("error_code", "column", "detail")
                    if key in sample
                }
                payload = {**details, **context}
                if payload:
                    LOG.warning("NDJSON finalize error sample for %s: %s", table_key, payload)
            logged_errors = True
        if result.alignment.num_rows and not logged_alignment:
            row = next(iter_rows(result.alignment), None)
            if row is None:
                return
            missing = row.get("missing_columns") or []
            extra = row.get("extra_columns") or []
            coerced = row.get("coerced_columns") or []
            if missing or extra or coerced:
                LOG.info(
                    "NDJSON finalize alignment for %s: missing=%s extra=%s coerced=%s",
                    table_key,
                    missing,
                    extra,
                    coerced,
                )
                logged_alignment = True

    return _hook


def _export_execution_context(ops: ServingOperations) -> ExecutionContext:
    use_threads = ops.settings.dataset_use_threads
    resolved_threads = True if use_threads is None else use_threads
    return ExecutionContext(
        use_threads=resolved_threads,
        determinism="canonical",
        combine_chunks=True,
    )


def _order_by_sort_keys(order_by: list[str]) -> tuple[SortKey, ...]:
    keys: list[SortKey] = []
    for item in order_by:
        if not item:
            continue
        descending = item.startswith("-")
        column = item[1:] if descending else item
        keys.append((column, "descending" if descending else "ascending"))
    return tuple(keys)


def _resolve_export_order_by(
    payload: SemanticExportRequest,
    view_desc: SemanticViewDescriptionResponse,
) -> tuple[SortKey, ...]:
    order_by = payload.order_by or view_desc.defaults.order_by
    if not order_by and view_desc.primary_key:
        order_by = list(view_desc.primary_key)
    if not order_by:
        msg = (
            "Export requires order_by for deterministic output when primary_key is missing: "
            f"{payload.view_id}"
        )
        raise ValueError(msg)
    return _order_by_sort_keys(order_by)


def export_hash_headers(
    *,
    query_hash: str,
    schema_hash: str | None,
    ast_fingerprint: str | None = None,
    sql_fingerprint: str | None = None,
) -> dict[str, str]:
    """Return stable hash headers used for export caching.

    Returns
    -------
    dict[str, str]
        Headers carrying query/schema hash identifiers.
    """
    headers: dict[str, str] = {"X-CodeIntel-Query-Hash": query_hash}
    if schema_hash is not None:
        headers["X-CodeIntel-Schema-Hash"] = schema_hash
    if ast_fingerprint is not None:
        headers["X-CodeIntel-AST-Fingerprint"] = ast_fingerprint
    if sql_fingerprint is not None:
        headers["X-CodeIntel-SQL-Fingerprint"] = sql_fingerprint
    return headers


async def dispatch_semantic_export(
    ops: ServingOperations,
    payload: SemanticExportRequest,
    metrics: ExportMetricsContext,
    *,
    options: ExportDispatchOptions,
) -> ExportDispatchResult:
    """Dispatch a semantic export request to an HTTP response builder.

    Returns
    -------
    ExportDispatchResult
        Response payload and optional metrics row count.
    """
    plan = build_export_plan(payload)
    if plan.delivery is ExportDelivery.ndjson_stream:
        tracker = _ExportStreamMetrics(metrics=metrics)
        view_desc = ops.describe(payload.view_id)
        execution_ctx = _export_execution_context(ops)
        order_by = _resolve_export_order_by(payload, view_desc)
        finalize_spec = finalize_spec_for_table(
            view_desc.table_key,
            mode=_finalize_mode(ops.settings.schema_enforcement),
            dedupe=FinalizeDedupe(
                enabled=False,
                keys=tuple(view_desc.primary_key),
                tie_breakers=order_by,
                tier="canonical",
                strategy="order_independent",
            ),
            context_fields=DEFAULT_ARROW_PROVENANCE_COLUMNS,
            order_by=order_by,
            emit_artifacts=True,
        )
        response = ndjson_response_from_batches(
            ops.export_record_batches(payload, cancel_check=options.cancel_check),
            options=NdjsonBatchResponseOptions(
                filename=f"{payload.view_id}{plan.suffix}",
                headers=options.headers,
                background=BackgroundTask(tracker.finalize),
                cancel_check=options.cancel_check,
                batch_hook=lambda batch: tracker.record_rows(batch.num_rows),
                finalize_spec=finalize_spec,
                finalize_hook=_finalize_log_hook(view_desc.table_key),
                execution_context=execution_ctx,
            ),
        )
        return ExportDispatchResult(response=response, metrics_row_count=None)
    if plan.delivery is ExportDelivery.binary_file:
        response, rows_written = await _binary_response(
            ops,
            payload,
            plan,
            options=options,
        )
        return ExportDispatchResult(response=response, metrics_row_count=rows_written)

    rows = await _run_blocking(
        lambda: list(ops.export_rows(payload, cancel_check=options.cancel_check)),
        timeout_s=options.timeout_s,
        cancel_check=options.cancel_check,
    )
    response = _json_dict_response(rows, plan=plan, metrics=metrics, headers=options.headers)
    return ExportDispatchResult(response=response, metrics_row_count=len(rows))


def _json_dict_response(
    rows: list[dict[str, object]],
    *,
    plan: ExportPlan,
    metrics: ExportMetricsContext,
    headers: dict[str, str],
) -> Response:
    payload: dict[str, object] = {
        "view_id": metrics.view_id,
        "rows": rows,
        "count": len(rows),
        "query_hash": metrics.query_hash,
    }
    if metrics.schema_hash is not None:
        payload["schema_hash"] = metrics.schema_hash
    if metrics.ast_fingerprint is not None:
        payload["ast_fingerprint"] = metrics.ast_fingerprint
    if metrics.sql_fingerprint is not None:
        payload["sql_fingerprint"] = metrics.sql_fingerprint
    return JSONResponse(
        content=payload,
        media_type=plan.mime_type,
        headers=headers,
    )


async def _binary_response(
    ops: ServingOperations,
    payload: SemanticExportRequest,
    plan: ExportPlan,
    *,
    options: ExportDispatchOptions,
) -> tuple[FileResponse, int]:
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"codeintel-export-{payload.view_id}-",
        suffix=plan.suffix,
    )
    os.close(fd)
    rows_written: int | None = None
    try:
        rows_written = await _run_blocking(
            lambda: write_export_file(
                ops,
                payload,
                output_path=Path(tmp_path),
                cancel_check=options.cancel_check,
            ),
            timeout_s=options.timeout_s,
            cancel_check=options.cancel_check,
        )
    finally:
        if rows_written is None:
            _unlink_best_effort(tmp_path)

    if rows_written is None:
        msg = "Export file writer returned no row count"
        raise RuntimeError(msg)

    response = FileResponse(
        path=tmp_path,
        media_type=plan.mime_type,
        filename=f"{payload.view_id}{plan.suffix}",
        headers=options.headers,
        background=BackgroundTask(lambda: _unlink_best_effort(tmp_path)),
    )
    return response, rows_written


def _unlink_best_effort(path: str) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return


async def _run_blocking[T](
    fn: Callable[[], T],
    *,
    timeout_s: float | None,
    cancel_check: CancelCheck | None,
) -> T:
    if cancel_check is not None:
        cancel_check()
    if timeout_s is None:
        return await to_thread.run_sync(fn, abandon_on_cancel=True)
    with anyio.fail_after(timeout_s):
        return await to_thread.run_sync(fn, abandon_on_cancel=True)


__all__ = [
    "ExportDispatchOptions",
    "ExportDispatchResult",
    "ExportMetricsContext",
    "dispatch_semantic_export",
    "export_hash_headers",
]
