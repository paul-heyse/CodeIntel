"""V1 export/streaming HTTP routes.

Provides endpoints for exporting large datasets from semantic views in
multiple formats: JSON, NDJSON, Parquet, and Arrow.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from codeintel.serving.http.dependencies import Kernel, require_api_key
from codeintel.serving.http.errors import ProblemType, ServingError
from codeintel.serving.http.streaming import ndjson_response
from codeintel.serving.semantic.models import SemanticExportRequest

if TYPE_CHECKING:
    from starlette.responses import Response

router = APIRouter(
    prefix="/export",
    tags=["export"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/semantic/{view_id}")
async def export_view(
    view_id: str,
    payload: SemanticExportRequest,
    kernel: Kernel,
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
    """
    if payload.view_id != view_id:
        payload = payload.model_copy(update={"view_id": view_id})

    try:
        if payload.format == "ndjson":
            rows = await run_in_threadpool(
                lambda: list(kernel.export_rows(payload))
            )
            return ndjson_response(rows, filename=f"{view_id}.ndjson")

        if payload.format == "parquet":
            return await _parquet_response(kernel, payload, view_id)

        if payload.format == "arrow":
            return await _arrow_response(kernel, payload, view_id)

        # Default: JSON format (same as /query but with higher limit)
        rows = await run_in_threadpool(
            lambda: list(kernel.export_rows(payload))
        )
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
    from fastapi.responses import JSONResponse  # noqa: PLC0415

    return JSONResponse(
        content={"view_id": view_id, "rows": rows, "count": len(rows)},
        media_type="application/json",
    )


async def _parquet_response(
    kernel: Kernel,
    payload: SemanticExportRequest,
    view_id: str,
) -> StreamingResponse:
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
    StreamingResponse
        Parquet file response.

    Raises
    ------
    ServingError
        When pyarrow is not available.
    """
    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:
        raise ServingError(
            problem_type=ProblemType.INTERNAL_ERROR,
            title="Export Unavailable",
            status=501,
            detail="Parquet export requires pyarrow to be installed",
        ) from exc

    rows = await run_in_threadpool(lambda: list(kernel.export_rows(payload)))
    table = pa.Table.from_pylist(rows)

    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/vnd.apache.parquet",
        headers={
            "Content-Disposition": f'attachment; filename="{view_id}.parquet"'
        },
    )


async def _arrow_response(
    kernel: Kernel,
    payload: SemanticExportRequest,
    view_id: str,
) -> StreamingResponse:
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
    StreamingResponse
        Arrow IPC file response.

    Raises
    ------
    ServingError
        When pyarrow is not available.
    """
    try:
        import pyarrow as pa  # noqa: PLC0415
    except ImportError as exc:
        raise ServingError(
            problem_type=ProblemType.INTERNAL_ERROR,
            title="Export Unavailable",
            status=501,
            detail="Arrow export requires pyarrow to be installed",
        ) from exc

    rows = await run_in_threadpool(lambda: list(kernel.export_rows(payload)))
    table = pa.Table.from_pylist(rows)

    buffer = io.BytesIO()
    with pa.ipc.new_file(buffer, table.schema) as writer:
        writer.write_table(table)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/vnd.apache.arrow.file",
        headers={
            "Content-Disposition": f'attachment; filename="{view_id}.arrow"'
        },
    )


__all__ = ["router"]
