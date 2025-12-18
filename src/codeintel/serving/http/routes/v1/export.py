"""V1 export/streaming HTTP routes.

Provides endpoints for exporting large datasets from semantic views in multiple formats:
JSON, NDJSON, Parquet, and Arrow.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from codeintel.serving.http.dependencies import Ops, require_api_key
from codeintel.serving.http.export_dispatch import (
    ExportMetricsContext,
    dispatch_semantic_export,
    export_hash_headers,
)
from codeintel.serving.http.route_utils import schedule_query_metrics
from codeintel.serving.http.middleware import get_correlation_id
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
    background: BackgroundTasks,
    request: Request,
    ops: Ops,
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
    ops
        Serving operations facade.

    Returns
    -------
    Response
        Streaming response in the requested format.
    """
    if payload.view_id != view_id:
        payload = payload.model_copy(update={"view_id": view_id})

    correlation_id = get_correlation_id(request)
    start = time.perf_counter()

    query_hash, schema_hash = ops.export_fingerprint(payload)
    headers = export_hash_headers(query_hash=query_hash, schema_hash=schema_hash)
    metrics = ExportMetricsContext(
        view_id=view_id,
        correlation_id=correlation_id,
        query_hash=query_hash,
        schema_hash=schema_hash,
    )

    dispatched = await dispatch_semantic_export(ops, payload, metrics, headers=headers)
    if dispatched.metrics_row_count is not None:
        duration_ms = (time.perf_counter() - start) * 1000
        schedule_query_metrics(
            background,
            metrics.to_metrics(row_count=dispatched.metrics_row_count, duration_ms=duration_ms),
        )
    return dispatched.response


__all__ = ["router"]
