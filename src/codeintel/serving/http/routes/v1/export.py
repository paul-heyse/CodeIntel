"""V1 export/streaming HTTP routes.

Provides endpoints for exporting large datasets from semantic views in
multiple formats: JSON, NDJSON, Parquet, and Arrow.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from starlette.responses import FileResponse

from codeintel.serving.export.formats import mime_type_for_export_format
from codeintel.serving.http.dependencies import get_kernel, require_api_key
from codeintel.serving.http.errors import ProblemType, ServingError
from codeintel.serving.http.metrics import QueryMetrics, log_query_metrics
from codeintel.serving.http.middleware import get_correlation_id
from codeintel.serving.http.streaming import ndjson_response
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.semantic.models import SemanticExportRequest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from starlette.responses import Response

router = APIRouter(
    prefix="/export",
    tags=["export"],
    dependencies=[Depends(require_api_key)],
)

_KERNEL_DEPENDENCY = Depends(get_kernel)


@dataclass(frozen=True, slots=True)
class _ExportMetricsContext:
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


def _export_hash_headers(*, query_hash: str, schema_hash: str | None) -> dict[str, str]:
    headers: dict[str, str] = {"X-CodeIntel-Query-Hash": query_hash}
    if schema_hash is not None:
        headers["X-CodeIntel-Schema-Hash"] = schema_hash
    return headers


def _dependency_type_error(
    payload: object,
    background: object,
    request: object,
    kernel: object,
) -> str | None:
    if not isinstance(payload, SemanticExportRequest):
        return "FastAPI did not provide a SemanticExportRequest model"
    if not isinstance(background, BackgroundTasks):
        return "FastAPI did not provide a BackgroundTasks instance"
    if not isinstance(request, Request):
        return "FastAPI did not provide a Request instance"
    if not isinstance(kernel, SemanticQueryKernel):
        return "FastAPI did not provide a SemanticQueryKernel instance"
    return None


def _iter_rows_with_metrics(
    *,
    kernel: SemanticQueryKernel,
    payload: SemanticExportRequest,
    metrics: _ExportMetricsContext,
) -> Iterator[dict[str, object]]:
    row_count = 0
    started = time.perf_counter()
    try:
        for row in kernel.export_rows(payload):
            row_count += 1
            yield row
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        log_query_metrics(metrics.to_metrics(row_count=row_count, duration_ms=duration_ms))


def _log_export_metrics(
    *,
    background: BackgroundTasks,
    metrics: _ExportMetricsContext,
    row_count: int,
    duration_ms: float,
) -> None:
    background.add_task(log_query_metrics, metrics.to_metrics(row_count=row_count, duration_ms=duration_ms))


@router.post("/semantic/{view_id}")
async def export_view(
    view_id: str,
    payload: SemanticExportRequest,
    background: BackgroundTasks,
    request: Request,
    kernel: SemanticQueryKernel = _KERNEL_DEPENDENCY,
) -> Response:
    """Export semantic view data as JSON, NDJSON, Parquet, or Arrow.

    Supports larger result sets than the standard /semantic/query endpoint.
    Use NDJSON for streaming downloads or Parquet/Arrow for binary columnar
    formats suitable for data analysis tools.

    Parameters
    ----------
    view_id
        Semantic view identifier.
    payload
        Export request with format, filters, and pagination.
    background
        Background task queue.
    request
        Current HTTP request.
    kernel
        Semantic query kernel.

    Returns
    -------
    Response
        Streaming response in the requested format.

    Raises
    ------
    ServingError
        When the view is not found or export format is unavailable.
    TypeError
        When FastAPI fails to inject required dependencies.
    """
    if msg := _dependency_type_error(payload, background, request, kernel):
        raise TypeError(msg)
    if payload.view_id != view_id:
        payload = payload.model_copy(update={"view_id": view_id})

    correlation_id = get_correlation_id(request)
    start = time.perf_counter()
    try:
        query_hash, schema_hash = kernel.export_fingerprint(payload)
        headers = _export_hash_headers(query_hash=query_hash, schema_hash=schema_hash)
        metrics = _ExportMetricsContext(
            view_id=view_id,
            correlation_id=correlation_id,
            query_hash=query_hash,
            schema_hash=schema_hash,
        )

        if payload.format == "ndjson":
            return ndjson_response(
                _iter_rows_with_metrics(kernel=kernel, payload=payload, metrics=metrics),
                filename=f"{view_id}.ndjson",
                headers=headers,
            )

        if payload.format == "parquet":
            response = await _parquet_response(kernel, payload, view_id, headers=headers)
            duration_ms = (time.perf_counter() - start) * 1000
            _log_export_metrics(background=background, metrics=metrics, row_count=0, duration_ms=duration_ms)
            return response

        if payload.format == "arrow":
            response, rows_written = await _arrow_response(kernel, payload, view_id, headers=headers)
            duration_ms = (time.perf_counter() - start) * 1000
            _log_export_metrics(
                background=background,
                metrics=metrics,
                row_count=rows_written,
                duration_ms=duration_ms,
            )
            return response

        # Default: JSON format (same as /query but with higher limit)
        rows = await run_in_threadpool(lambda: list(kernel.export_rows(payload)))
        duration_ms = (time.perf_counter() - start) * 1000
        _log_export_metrics(
            background=background,
            metrics=metrics,
            row_count=len(rows),
            duration_ms=duration_ms,
        )
        return _json_dict_response(
            view_id,
            rows,
            query_hash=query_hash,
            schema_hash=schema_hash,
            headers=headers,
        )

    except KeyError as exc:
        raise ServingError(
            problem_type=ProblemType.VIEW_NOT_FOUND,
            title="View Not Found",
            status=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise ServingError(
            problem_type=ProblemType.INVALID_QUERY,
            title="Invalid Query",
            status=400,
            detail=str(exc),
        ) from exc


def _json_dict_response(
    view_id: str,
    rows: list[dict[str, object]],
    *,
    query_hash: str,
    schema_hash: str | None,
    headers: dict[str, str],
) -> Response:
    """Create a JSON response with rows and metadata.

    Parameters
    ----------
    view_id
        View identifier for metadata.
    rows
        Export rows.
    query_hash
        Stable fingerprint of query inputs.
    schema_hash
        Stable fingerprint of resolved schema when available.
    headers
        Extra response headers.

    Returns
    -------
    Response
        JSON response.
    """
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
    kernel: SemanticQueryKernel,
    payload: SemanticExportRequest,
    view_id: str,
    *,
    headers: dict[str, str],
) -> FileResponse:
    """Generate Parquet export (requires pyarrow).

    Parameters
    ----------
    kernel
        Semantic query kernel.
    payload
        Export request parameters.
    view_id
        View identifier for filename.
    headers
        Extra response headers.

    Returns
    -------
    FileResponse
        Parquet file response.

    Raises
    ------
    KeyError
        If the view cannot be resolved by the kernel.
    OSError
        If writing the Parquet file fails.
    RuntimeError
        If the kernel cannot acquire an export connection.
    ValueError
        If the export request is invalid.
    """
    fd, tmp_path = tempfile.mkstemp(prefix=f"codeintel-export-{view_id}-", suffix=".parquet")
    os.close(fd)
    try:
        await run_in_threadpool(
            lambda: kernel.export_to_parquet(payload, output_path=Path(tmp_path))
        )
    except (KeyError, OSError, RuntimeError, ValueError):
        _unlink_best_effort(tmp_path)
        raise

    def _cleanup() -> None:
        _unlink_best_effort(tmp_path)

    return FileResponse(
        path=tmp_path,
        media_type=mime_type_for_export_format("parquet"),
        filename=f"{view_id}.parquet",
        headers=headers,
        background=BackgroundTask(_cleanup),
    )


async def _arrow_response(
    kernel: SemanticQueryKernel,
    payload: SemanticExportRequest,
    view_id: str,
    *,
    headers: dict[str, str],
) -> tuple[FileResponse, int]:
    """Generate Arrow IPC export (requires pyarrow).

    Parameters
    ----------
    kernel
        Semantic query kernel.
    payload
        Export request parameters.
    view_id
        View identifier for filename.
    headers
        Extra response headers.

    Returns
    -------
    FileResponse
        Arrow IPC file response.
    int
        Number of rows written.

    Raises
    ------
    KeyError
        If the view cannot be resolved by the kernel.
    OSError
        If writing the Arrow file fails.
    RuntimeError
        If the kernel cannot acquire an export connection.
    ValueError
        If the export request is invalid.
    """
    fd, tmp_path = tempfile.mkstemp(prefix=f"codeintel-export-{view_id}-", suffix=".arrow")
    os.close(fd)
    try:
        rows_written = await run_in_threadpool(
            lambda: kernel.export_to_arrow_ipc(payload, output_path=Path(tmp_path))
        )
    except (KeyError, OSError, RuntimeError, ValueError):
        _unlink_best_effort(tmp_path)
        raise

    def _cleanup() -> None:
        _unlink_best_effort(tmp_path)

    response = FileResponse(
        path=tmp_path,
        media_type=mime_type_for_export_format("arrow"),
        filename=f"{view_id}.arrow",
        headers=headers,
        background=BackgroundTask(_cleanup),
    )
    return response, rows_written


def _unlink_best_effort(path: str) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return


__all__ = ["router"]
