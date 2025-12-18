"""HTTP export dispatch helpers.

This module centralizes the mapping from export format -> response builder so
HTTP route handlers stay thin and consistent.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from starlette.responses import FileResponse

from codeintel.serving.export.formats import ExportFormat, mime_type_for_export_format, suffix_for_export_format
from codeintel.serving.http.metrics import QueryMetrics, log_query_metrics
from codeintel.serving.http.streaming import ndjson_response
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.models import SemanticExportRequest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from starlette.responses import Response


@dataclass(frozen=True, slots=True)
class ExportDispatchResult:
    """Result of dispatching an export request for HTTP."""

    response: Response
    metrics_row_count: int | None


@dataclass(frozen=True, slots=True)
class ExportMetricsContext:
    """Context required to emit export metrics."""

    view_id: str
    correlation_id: str
    query_hash: str
    schema_hash: str | None

    def to_metrics(self, *, row_count: int, duration_ms: float) -> QueryMetrics:
        return QueryMetrics(
            endpoint="/export/semantic",
            view_id=self.view_id,
            query=None,
            row_count=row_count,
            truncated=False,
            duration_ms=duration_ms,
            correlation_id=self.correlation_id,
            query_hash=self.query_hash,
            schema_hash=self.schema_hash,
        )


def export_hash_headers(*, query_hash: str, schema_hash: str | None) -> dict[str, str]:
    headers: dict[str, str] = {"X-CodeIntel-Query-Hash": query_hash}
    if schema_hash is not None:
        headers["X-CodeIntel-Schema-Hash"] = schema_hash
    return headers


def _iter_rows_with_metrics(
    *,
    ops: ServingOperations,
    payload: SemanticExportRequest,
    metrics: ExportMetricsContext,
) -> Iterator[dict[str, object]]:
    row_count = 0
    started = time.perf_counter()
    try:
        for row in ops.export_rows(payload):
            row_count += 1
            yield row
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        log_query_metrics(metrics.to_metrics(row_count=row_count, duration_ms=duration_ms))


async def dispatch_semantic_export(
    ops: ServingOperations,
    payload: SemanticExportRequest,
    metrics: ExportMetricsContext,
    *,
    headers: dict[str, str],
) -> ExportDispatchResult:
    """Dispatch a semantic export request to an HTTP response builder."""
    if payload.format == "ndjson":
        response = ndjson_response(
            _iter_rows_with_metrics(ops=ops, payload=payload, metrics=metrics),
            filename=f"{payload.view_id}{suffix_for_export_format(payload.format)}",
            headers=headers,
        )
        return ExportDispatchResult(response=response, metrics_row_count=None)

    if payload.format == "parquet":
        response, rows_written = await _parquet_response(ops, payload, headers=headers)
        return ExportDispatchResult(response=response, metrics_row_count=rows_written)

    if payload.format == "arrow":
        response, rows_written = await _arrow_response(ops, payload, headers=headers)
        return ExportDispatchResult(response=response, metrics_row_count=rows_written)

    rows = await run_in_threadpool(lambda: list(ops.export_rows(payload)))
    response = _json_dict_response(
        payload.view_id,
        rows,
        query_hash=metrics.query_hash,
        schema_hash=metrics.schema_hash,
        headers=headers,
    )
    return ExportDispatchResult(response=response, metrics_row_count=len(rows))


def _json_dict_response(
    view_id: str,
    rows: list[dict[str, object]],
    *,
    query_hash: str,
    schema_hash: str | None,
    headers: dict[str, str],
) -> Response:
    payload: dict[str, object] = {
        "view_id": view_id,
        "rows": rows,
        "count": len(rows),
        "query_hash": query_hash,
    }
    if schema_hash is not None:
        payload["schema_hash"] = schema_hash
    return JSONResponse(
        content=payload,
        media_type=mime_type_for_export_format("json"),
        headers=headers,
    )


async def _parquet_response(
    ops: ServingOperations,
    payload: SemanticExportRequest,
    *,
    headers: dict[str, str],
) -> tuple[FileResponse, int]:
    suffix = suffix_for_export_format("parquet")
    fd, tmp_path = tempfile.mkstemp(prefix=f"codeintel-export-{payload.view_id}-", suffix=suffix)
    os.close(fd)
    try:
        rows_written = await run_in_threadpool(
            lambda: ops.export_to_parquet(payload, output_path=Path(tmp_path))
        )
    except Exception:
        _unlink_best_effort(tmp_path)
        raise

    response = FileResponse(
        path=tmp_path,
        media_type=mime_type_for_export_format("parquet"),
        filename=f"{payload.view_id}{suffix}",
        headers=headers,
        background=BackgroundTask(lambda: _unlink_best_effort(tmp_path)),
    )
    return response, rows_written


async def _arrow_response(
    ops: ServingOperations,
    payload: SemanticExportRequest,
    *,
    headers: dict[str, str],
) -> tuple[FileResponse, int]:
    suffix = suffix_for_export_format("arrow")
    fd, tmp_path = tempfile.mkstemp(prefix=f"codeintel-export-{payload.view_id}-", suffix=suffix)
    os.close(fd)
    try:
        rows_written = await run_in_threadpool(
            lambda: ops.export_to_arrow_ipc(payload, output_path=Path(tmp_path))
        )
    except Exception:
        _unlink_best_effort(tmp_path)
        raise

    response = FileResponse(
        path=tmp_path,
        media_type=mime_type_for_export_format("arrow"),
        filename=f"{payload.view_id}{suffix}",
        headers=headers,
        background=BackgroundTask(lambda: _unlink_best_effort(tmp_path)),
    )
    return response, rows_written


def _unlink_best_effort(path: str) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return


__all__ = ["ExportDispatchResult", "ExportMetricsContext", "dispatch_semantic_export", "export_hash_headers"]

