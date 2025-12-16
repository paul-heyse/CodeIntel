"""FastAPI application factory for semantic serving."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.http.errors import (
    ProblemDetail,
    ProblemType,
    ServingError,
    internal_error_problem,
    problem_from_error,
    problem_response,
)
from codeintel.serving.http.middleware import (
    correlation_id_and_timing_middleware,
    get_correlation_id,
)
from codeintel.serving.http.routes import router as api_router
from codeintel.serving.http.state import ServingState
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from fastapi import Request
    from fastapi.responses import JSONResponse


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
    """
    cfg = settings or ServingSettings.from_env()
    db_manager = ServingDBManager(
        pointer_path=cfg.serve_dir / "current.json",
        pool_cfg=PoolConfig(size=cfg.pool_size),
        poll_interval_s=cfg.poll_interval_s,
        hot_swap=cfg.hot_swap,
    )
    kernel = SemanticQueryKernel(db=db_manager, settings=cfg)
    state = ServingState(settings=cfg, db=db_manager, kernel=kernel)

    app = FastAPI(
        title="CodeIntel Serving",
        description="Semantic layer API for CodeIntel",
        version="1.0.0",
        lifespan=_build_lifespan(db_manager),
    )
    app.state.serving = state

    _install_exception_handlers(app)
    _install_middlewares(app, cfg)
    app.include_router(api_router)
    _install_observability_routes(app, db_manager=db_manager, kernel=kernel)
    _maybe_mount_mcp(app, kernel=kernel, enabled=mount_mcp)

    app.openapi = lambda: _custom_openapi(app)
    return app


__all__ = ["create_serving_app"]


def _build_lifespan(db_manager: ServingDBManager) -> Callable[[FastAPI], AsyncGenerator[None]]:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        await db_manager.start()
        try:
            yield
        finally:
            await db_manager.stop()

    return lifespan


def _handle_serving_error(request: Request, exc: ServingError) -> JSONResponse:
    problem = problem_from_error(request, exc)
    return problem_response(problem, headers=exc.headers)


def _handle_request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
    problem = ProblemDetail(
        type=str(ProblemType.VALIDATION_ERROR),
        title="Validation Error",
        status=422,
        detail="Request validation failed.",
        instance=str(request.url.path),
        correlation_id=get_correlation_id(request),
        errors=[_normalize_validation_error(err) for err in exc.errors()],
    )
    return problem_response(problem)


def _handle_unexpected(request: Request, _exc: Exception) -> JSONResponse:
    return problem_response(internal_error_problem(request))


def _install_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ServingError, _handle_serving_error)
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


def _install_observability_routes(
    app: FastAPI, *, db_manager: ServingDBManager, kernel: SemanticQueryKernel
) -> None:
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
        return await run_in_threadpool(kernel.meta)


def _maybe_mount_mcp(app: FastAPI, *, kernel: SemanticQueryKernel, enabled: bool) -> None:
    if not enabled:
        return
    mcp = build_mcp_app(kernel=kernel, streamable_http_path="/")
    app.mount("/mcp", mcp.streamable_http_app())


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
