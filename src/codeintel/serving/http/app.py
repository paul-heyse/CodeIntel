"""FastAPI application factory for semantic serving."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from codeintel.serving.http.errors import (
    CodeIntelDomainError,
    ProblemDetail,
    problem_from_domain_error,
    problem_from_exception,
    problem_response,
)
from codeintel.serving.http.middleware import (
    correlation_id_and_timing_middleware,
    get_correlation_id,
)
from codeintel.serving.http.routes import router as api_router
from codeintel.serving.http.state import ServingState
from codeintel.serving.mcp._compat import EventStore
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.meta.service import build_kernel_meta_payload
from codeintel.serving.runtime import build_runtime
from codeintel.serving.settings import ServingSettings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from contextlib import AbstractAsyncContextManager

    from fastapi import Request
    from fastapi.responses import JSONResponse

    from codeintel.serving.db.manager import ServingDBManager
    from codeintel.serving.operations.protocols import ServingKernelProtocol

LOG = logging.getLogger(__name__)


def create_serving_app(
    settings: ServingSettings | None = None,
    *,
    mount_mcp: bool = True,
) -> FastAPI:
    """Create FastAPI serving application.

    Parameters
    ----------
    settings
        Serving settings (defaults to environment).
    mount_mcp
        Whether to mount an MCP server under `/mcp`.

    Returns
    -------
    FastAPI
        Configured application.

    Notes
    -----
    Calls ``cfg.validate_auth_for_host()`` which raises ``ValueError`` if
    binding to a public interface (0.0.0.0, ::) without authentication.
    """
    cfg = settings or ServingSettings.from_env()

    # Fail-fast: require auth for public interfaces
    cfg.validate_auth_for_host()
    cfg.validate_mcp_single_worker(mount_mcp=mount_mcp)

    runtime = build_runtime(cfg)
    state = ServingState(
        settings=cfg,
        db=runtime.db_manager,
        kernel=runtime.kernel,
        ops=runtime.ops,
    )

    app = FastAPI(
        title="CodeIntel Serving",
        description="Semantic layer API for CodeIntel",
        version="1.0.0",
        lifespan=_build_lifespan(runtime.db_manager),
    )
    app.state.serving = state

    _install_exception_handlers(app)
    _install_middlewares(app, cfg)
    app.include_router(api_router)
    _install_observability_routes(app, db_manager=runtime.db_manager)
    _maybe_mount_mcp(app, kernel=runtime.kernel, settings=cfg, enabled=mount_mcp)

    app.openapi = lambda: _custom_openapi(app)
    return app


__all__ = ["create_serving_app"]


def _build_lifespan(
    db_manager: ServingDBManager,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await db_manager.start()
        try:
            yield
        finally:
            await db_manager.stop()

    return lifespan


def _handle_serving_error(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, CodeIntelDomainError):
        return _handle_unexpected(request, exc)
    problem = problem_from_domain_error(request, exc)
    return problem_response(problem)


def _handle_request_validation(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        return _handle_unexpected(request, exc)
    problem = ProblemDetail(
        type="/problems/validation-error",
        title="Validation Error",
        status=422,
        detail="Request validation failed.",
        instance=str(request.url.path),
        correlation_id=get_correlation_id(request),
        errors=[_normalize_validation_error(err) for err in exc.errors()],
    )
    return problem_response(problem)


def _handle_unexpected(request: Request, _exc: Exception) -> JSONResponse:
    return problem_response(problem_from_exception(request, _exc))


def _install_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(CodeIntelDomainError, _handle_serving_error)
    app.add_exception_handler(RequestValidationError, _handle_request_validation)
    app.add_exception_handler(Exception, _handle_unexpected)


def _install_middlewares(app: FastAPI, cfg: ServingSettings) -> None:
    if cfg.enable_gzip:
        app.add_middleware(GZipMiddleware, minimum_size=cfg.gzip_minimum_size)

    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if cfg.trusted_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(cfg.trusted_hosts))

    app.middleware("http")(correlation_id_and_timing_middleware)


def _install_observability_routes(app: FastAPI, *, db_manager: ServingDBManager) -> None:
    @app.get("/health")
    async def health() -> dict[str, str]:
        pointer = db_manager.current_pointer()
        return {
            "status": "ok",
            "repo": pointer.repo,
            "commit": pointer.commit,
            "run_id": pointer.run_id,
        }

    @app.get("/meta")
    async def meta() -> dict[str, object]:
        return await run_in_threadpool(lambda: build_kernel_meta_payload(db_manager))


def _maybe_mount_mcp(
    app: FastAPI,
    *,
    kernel: ServingKernelProtocol,
    settings: ServingSettings,
    enabled: bool,
) -> None:
    """Mount MCP server under /mcp with EventStore for resumability.

    Mount Contract
    --------------
    - FastAPI mounts at: /mcp
    - MCP ASGI app path: /
    - Effective MCP endpoint: /mcp (NOT /mcp/mcp)

    Parameters
    ----------
    app
        FastAPI application to mount MCP on.
    kernel
        Semantic query kernel for MCP tools.
    settings
        Serving settings for MCP configuration.
    enabled
        Whether MCP mounting is enabled.
    """
    if not enabled:
        return

    mcp = build_mcp_app(kernel=kernel, settings=settings)

    # Configure EventStore for SSE polling/resumability
    event_store = None
    retry_interval = None
    if settings.mcp_enable_event_store:
        if EventStore is None:
            LOG.warning("EventStore not available - SSE resumability disabled")
        else:
            event_store = EventStore()
            retry_interval = settings.mcp_retry_interval_ms

    # gofastmcp 2.x uses http_app() with path="/" to avoid double-prefix
    app.mount(
        "/mcp",
        mcp.http_app(
            path="/",
            transport="streamable-http",
            event_store=event_store,
            retry_interval=retry_interval,
            json_response=True,
            stateless_http=False,
        ),
    )


def _custom_openapi(app: FastAPI) -> dict[str, object]:
    if app.openapi_schema is not None:
        return cast("dict[str, object]", app.openapi_schema)

    openapi_schema = cast(
        "dict[str, object]",
        get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        ),
    )
    components = _ensure_dict(openapi_schema.get("components"), ctx="openapi.components")
    openapi_schema["components"] = components

    schemas = _ensure_dict(components.get("schemas"), ctx="openapi.components.schemas")
    components["schemas"] = schemas
    schemas["ProblemDetail"] = ProblemDetail.model_json_schema()

    app.openapi_schema = openapi_schema
    return openapi_schema


def _ensure_dict(value: object, *, ctx: str) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return cast("dict[str, object]", value)
    msg = f"Expected dict for {ctx}"
    raise TypeError(msg)


def _normalize_validation_error(err: object) -> dict[str, Any]:
    if isinstance(err, dict):
        normalized: dict[str, Any] = {}
        for key, value in err.items():
            normalized[str(key)] = value
        return normalized
    return {"msg": str(err)}
