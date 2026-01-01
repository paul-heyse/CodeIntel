"""V1 semantic HTTP routes."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from codeintel.serving.http.dependencies import Ops, require_api_key
from codeintel.serving.http.middleware import get_correlation_id
from codeintel.serving.http.route_utils import (
    ThreadpoolMetricsContext,
    run_in_threadpool_with_metrics,
)
from codeintel.serving.http.streaming import ArrowIpcResponseOptions, arrow_ipc_response
from codeintel.serving.metrics import QueryMetrics, log_query_metrics
from codeintel.serving.operations.cancellation import CancelToken
from codeintel.serving.semantic.models import (
    SemanticCatalogResponse,
    SemanticExplainResponse,
    SemanticQueryRequest,
    SemanticViewDescriptionResponse,
)

router = APIRouter(
    prefix="/semantic",
    tags=["semantic"],
    dependencies=[Depends(require_api_key, scope="request")],
)


@dataclass(slots=True)
class _QueryStreamMetrics:
    stream: Iterable[bytes]
    success_metrics: Callable[[Iterable[bytes], float, str], QueryMetrics]
    error_metrics: Callable[[float, str], QueryMetrics]
    correlation_id: str
    started: float = field(default_factory=time.perf_counter)
    logged: bool = False

    def log_success(self) -> None:
        if self.logged:
            return
        self.logged = True
        duration_ms = (time.perf_counter() - self.started) * 1000
        log_query_metrics(
            self.success_metrics(self.stream, duration_ms, self.correlation_id),
        )

    def log_error(self) -> None:
        if self.logged:
            return
        self.logged = True
        duration_ms = (time.perf_counter() - self.started) * 1000
        log_query_metrics(
            self.error_metrics(duration_ms, self.correlation_id),
        )


@router.get("/views", response_model=SemanticCatalogResponse)
async def list_views(
    background: BackgroundTasks,
    request: Request,
    ops: Ops,
) -> SemanticCatalogResponse:
    """List available semantic views.

    Parameters
    ----------
    background
        Background task queue.
    request
        Current HTTP request.
    ops
        Serving operations facade.

    Returns
    -------
    SemanticCatalogResponse
        Catalog response payload.
    """

    def _success(
        payload: SemanticCatalogResponse, duration_ms: float, correlation_id: str
    ) -> QueryMetrics:
        views = payload.views
        return QueryMetrics(
            endpoint="/v1/semantic/views",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=None,
            query=None,
            row_count=len(views),
            truncated=False,
        )

    def _error(duration_ms: float, correlation_id: str) -> QueryMetrics:
        return QueryMetrics(
            endpoint="/v1/semantic/views",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=None,
            query=None,
            row_count=0,
            truncated=False,
        )

    context = ThreadpoolMetricsContext(
        background=background,
        request=request,
        success_metrics=_success,
        error_metrics=_error,
    )
    return await run_in_threadpool_with_metrics(context, ops.catalog)


@router.get("/views/{view_id}", response_model=SemanticViewDescriptionResponse)
async def describe_view(
    view_id: str,
    background: BackgroundTasks,
    request: Request,
    ops: Ops,
) -> SemanticViewDescriptionResponse:
    """Describe a semantic view.

    Parameters
    ----------
    view_id
        Semantic view identifier.
    background
        Background task queue.
    request
        Current HTTP request.
    ops
        Serving operations facade.

    Returns
    -------
    SemanticViewDescriptionResponse
        View description payload.
    """

    def _success(
        _payload: SemanticViewDescriptionResponse, duration_ms: float, correlation_id: str
    ) -> QueryMetrics:
        return QueryMetrics(
            endpoint=f"/v1/semantic/views/{view_id}",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=view_id,
            query=None,
            row_count=1,
            truncated=False,
        )

    def _error(duration_ms: float, correlation_id: str) -> QueryMetrics:
        return QueryMetrics(
            endpoint=f"/v1/semantic/views/{view_id}",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=view_id,
            query=None,
            row_count=0,
            truncated=False,
        )

    context = ThreadpoolMetricsContext(
        background=background,
        request=request,
        success_metrics=_success,
        error_metrics=_error,
    )
    return await run_in_threadpool_with_metrics(context, ops.describe, view_id)


@router.post("/query", response_class=StreamingResponse)
async def query_view(
    payload: SemanticQueryRequest,
    background: BackgroundTasks,
    request: Request,
    ops: Ops,
) -> StreamingResponse:
    """Execute a semantic query against a view.

    Parameters
    ----------
    payload
        Semantic query request.
    background
        Background task queue.
    request
        Current HTTP request.
    ops
        Serving operations facade.

    Returns
    -------
    StreamingResponse
        Arrow IPC stream response.
    """

    def _success(result: Iterable[bytes], duration_ms: float, correlation_id: str) -> QueryMetrics:
        scan_metrics = getattr(result, "scan_metrics", None)
        return QueryMetrics(
            endpoint="/v1/semantic/query",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=payload.view_id,
            query=None,
            row_count=0,
            truncated=False,
            engine=getattr(result, "engine", None),
            engine_preference=ops.settings.query_engine,
            query_hash=getattr(result, "query_hash", None),
            schema_hash=getattr(result, "schema_hash", None),
            batch_size=getattr(result, "batch_size", None),
            scan_rows=scan_metrics.row_count if scan_metrics else None,
            scan_files=scan_metrics.file_count if scan_metrics else None,
            scan_bytes=scan_metrics.total_bytes if scan_metrics else None,
        )

    def _error(duration_ms: float, correlation_id: str) -> QueryMetrics:
        return QueryMetrics(
            endpoint="/v1/semantic/query",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=payload.view_id,
            query=None,
            row_count=0,
            truncated=False,
        )

    correlation_id = get_correlation_id(request)
    start = time.perf_counter()
    cancel_token = CancelToken.from_timeout(ops.settings.query_timeout_s)
    context = ThreadpoolMetricsContext(
        background=background,
        request=request,
        success_metrics=_success,
        error_metrics=_error,
        timeout_s=ops.settings.query_timeout_s,
        cancel_token=cancel_token,
    )
    try:
        stream: Iterable[bytes] = await run_in_threadpool_with_metrics(
            context,
            ops.query_ipc_stream,
            payload,
            cancel_check=cancel_token.raise_if_cancelled,
            emit_metrics=False,
        )
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        log_query_metrics(_error(duration_ms, correlation_id))
        raise

    tracker = _QueryStreamMetrics(
        stream=stream,
        success_metrics=_success,
        error_metrics=_error,
        correlation_id=correlation_id,
        started=start,
    )

    def _stream_with_metrics() -> Iterator[bytes]:
        try:
            yield from stream
        except Exception:
            tracker.log_error()
            raise

    return arrow_ipc_response(
        _stream_with_metrics(),
        options=ArrowIpcResponseOptions(
            filename=f"{payload.view_id}.arrow",
            cancel_check=cancel_token.raise_if_cancelled,
            background=BackgroundTask(tracker.log_success),
        ),
    )


@router.post("/explain", response_model=SemanticExplainResponse)
async def explain_view(
    payload: SemanticQueryRequest,
    background: BackgroundTasks,
    request: Request,
    ops: Ops,
) -> SemanticExplainResponse:
    """Compile a semantic query and return SQL + plan text.

    Parameters
    ----------
    payload
        Semantic query request.
    background
        Background task queue.
    request
        Current HTTP request.
    ops
        Serving operations facade.

    Returns
    -------
    SemanticExplainResponse
        Compiled SQL plus plan text for the query.
    """

    def _success(
        _response: SemanticExplainResponse, duration_ms: float, correlation_id: str
    ) -> QueryMetrics:
        return QueryMetrics(
            endpoint="/v1/semantic/explain",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=payload.view_id,
            query=None,
            row_count=0,
            truncated=False,
        )

    def _error(duration_ms: float, correlation_id: str) -> QueryMetrics:
        return QueryMetrics(
            endpoint="/v1/semantic/explain",
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            view_id=payload.view_id,
            query=None,
            row_count=0,
            truncated=False,
        )

    context = ThreadpoolMetricsContext(
        background=background,
        request=request,
        success_metrics=_success,
        error_metrics=_error,
    )
    return await run_in_threadpool_with_metrics(context, ops.explain, payload)


__all__ = ["router"]
