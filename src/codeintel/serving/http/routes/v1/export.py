"""V1 export/streaming HTTP routes.

Provides endpoints for exporting large datasets from semantic views in
multiple formats: JSON, NDJSON, Parquet, and Arrow.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from starlette.responses import FileResponse

from codeintel.serving.http.dependencies import get_kernel, require_api_key
from codeintel.serving.http.errors import ProblemType, ServingError
from codeintel.serving.http.streaming import ndjson_response
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.semantic.models import SemanticExportRequest

if TYPE_CHECKING:
    from starlette.responses import Response

router = APIRouter(
    prefix="/export",
    tags=["export"],
    dependencies=[Depends(require_api_key)],
)

_KERNEL_DEPENDENCY = Depends(get_kernel)


@router.post("/semantic/{view_id}")
async def export_view(
    view_id: str,
    payload: SemanticExportRequest,
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
    if not isinstance(payload, SemanticExportRequest):
        msg = "FastAPI did not provide a SemanticExportRequest model"
        raise TypeError(msg)
    if not isinstance(kernel, SemanticQueryKernel):
        msg = "FastAPI did not provide a SemanticQueryKernel instance"
        raise TypeError(msg)
    if payload.view_id != view_id:
        payload = payload.model_copy(update={"view_id": view_id})

    try:
        if payload.format == "ndjson":
            return ndjson_response(kernel.export_rows(payload), filename=f"{view_id}.ndjson")

        if payload.format == "parquet":
            return await _parquet_response(kernel, payload, view_id)

        if payload.format == "arrow":
            return await _arrow_response(kernel, payload, view_id)

        # Default: JSON format (same as /query but with higher limit)
        rows = await run_in_threadpool(lambda: list(kernel.export_rows(payload)))
        return _json_dict_response(view_id, rows)

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


def _json_dict_response(view_id: str, rows: list[dict[str, object]]) -> Response:
    """Create a JSON response with rows and metadata.

    Parameters
    ----------
    view_id
        View identifier for metadata.
    rows
        Export rows.

    Returns
    -------
    Response
        JSON response.
    """
    return JSONResponse(
        content={"view_id": view_id, "rows": rows, "count": len(rows)},
        media_type="application/json",
    )


async def _parquet_response(
    kernel: SemanticQueryKernel,
    payload: SemanticExportRequest,
    view_id: str,
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
        media_type="application/vnd.apache.parquet",
        filename=f"{view_id}.parquet",
        background=BackgroundTask(_cleanup),
    )


async def _arrow_response(
    kernel: SemanticQueryKernel,
    payload: SemanticExportRequest,
    view_id: str,
) -> FileResponse:
    """Generate Arrow IPC export (requires pyarrow).

    Parameters
    ----------
    kernel
        Semantic query kernel.
    payload
        Export request parameters.
    view_id
        View identifier for filename.

    Returns
    -------
    FileResponse
        Arrow IPC file response.

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
        await run_in_threadpool(
            lambda: kernel.export_to_arrow_ipc(payload, output_path=Path(tmp_path))
        )
    except (KeyError, OSError, RuntimeError, ValueError):
        _unlink_best_effort(tmp_path)
        raise

    def _cleanup() -> None:
        _unlink_best_effort(tmp_path)

    return FileResponse(
        path=tmp_path,
        media_type="application/vnd.apache.arrow.file",
        filename=f"{view_id}.arrow",
        background=BackgroundTask(_cleanup),
    )


def _unlink_best_effort(path: str) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return


__all__ = ["router"]
