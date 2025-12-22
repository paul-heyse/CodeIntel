"""FastAPI application factory for semantic serving."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastmcp.server.event_store import EventStore
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from codeintel.core.runtime.loader import load_runtime_settings
from codeintel.observability.otel import (
    bootstrap_observability,
    get_observability,
    observability_config_from_settings,
)
from codeintel.serving.auth.policy import require_http_auth
from codeintel.serving.features import ServingFeatureSet
from codeintel.serving.http.errors import (
    CodeIntelDomainError,
    ProblemDetail,
    ProblemDetailSchema,
    problem_from_domain_error,
    problem_from_exception,
    problem_response,
)
from codeintel.serving.http.middleware import (
    correlation_id_and_timing_middleware,
    get_correlation_id,
)
from codeintel.serving.http.routes import build_http_router
from codeintel.serving.http.state import ServingState
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.meta.service import build_kernel_meta_payload
from codeintel.serving.runtime import build_runtime
from codeintel.serving.settings import ServingSettings

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    _FASTAPI_INSTRUMENTOR_AVAILABLE = True
except ImportError:
    _FASTAPI_INSTRUMENTOR_AVAILABLE = False
    FastAPIInstrumentor = None

try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"
    generate_latest = None

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from contextlib import AbstractAsyncContextManager

    from fastapi import Request
    from fastapi.responses import JSONResponse

    from codeintel.serving.db.manager import ServingDBManager
    from codeintel.serving.operations.protocols import ServingKernelProtocol


def create_serving_app(
    settings: ServingSettings,
    *,
    mount_mcp: bool = True,
) -> FastAPI:
    """Create FastAPI serving application.

    Parameters
    ----------
    settings
        Serving settings for runtime configuration.
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
    cfg = settings
    features = ServingFeatureSet.from_settings(cfg)

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

    observability_settings = load_runtime_settings().observability
    bootstrap_observability(
        observability_config_from_settings(
            observability_settings,
            default_service_name="codeintel-serving",
        )
    )

    _install_exception_handlers(app)
    _install_middlewares(app, cfg)
    app.include_router(build_http_router(features))
    _install_observability_routes(app, db_manager=runtime.db_manager, settings=cfg)
    _maybe_mount_mcp(app, kernel=runtime.kernel, settings=cfg, features=features, enabled=mount_mcp)

    _instrument_fastapi(app)

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
        extensions={
            "correlation_id": get_correlation_id(request),
            "errors": [_normalize_validation_error(err) for err in exc.errors()],
        },
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


def _instrument_fastapi(app: FastAPI) -> None:
    if not _FASTAPI_INSTRUMENTOR_AVAILABLE or FastAPIInstrumentor is None:
        return
    FastAPIInstrumentor.instrument_app(app)


def _install_observability_routes(
    app: FastAPI,
    *,
    db_manager: ServingDBManager,
    settings: ServingSettings,
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
        payload = await run_in_threadpool(lambda: build_kernel_meta_payload(db_manager))
        return payload.model_dump(mode="json")

    obs = get_observability()
    metrics_auth_required = settings.metrics_auth_required
    if not obs.prometheus_enabled or not _PROMETHEUS_AVAILABLE or generate_latest is None:
        return

    generate_latest_fn = generate_latest

    @app.get("/metrics", include_in_schema=False)
    async def metrics(request: Request) -> Response:
        if metrics_auth_required:
            require_http_auth(headers=request.headers, settings=settings)
        payload = generate_latest_fn()
        return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


def _maybe_mount_mcp(
    app: FastAPI,
    *,
    kernel: ServingKernelProtocol,
    settings: ServingSettings,
    features: ServingFeatureSet,
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
    features
        Derived feature toggles for MCP configuration.
    enabled
        Whether MCP mounting is enabled.
    """
    if not enabled:
        return

    mcp = build_mcp_app(kernel=kernel, settings=settings)

    # Configure EventStore for SSE polling/resumability
    event_store = None
    retry_interval = None
    if features.enable_mcp_event_store:
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
    schemas["ProblemDetail"] = ProblemDetailSchema.model_json_schema()

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
