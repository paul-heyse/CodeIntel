"""FastAPI server exposing MCP-aligned queries over DuckDB."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.responses import Response

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.context import (
    RequestContext,
    reset_current_request_context,
    set_current_request_context,
)
from codeintel.serving.http.routes.architecture import build_architecture_router
from codeintel.serving.http.routes.datasets import build_datasets_router
from codeintel.serving.http.routes.functions import build_functions_router
from codeintel.serving.http.routes.health import build_health_router
from codeintel.serving.http.routes.ide import build_ide_router
from codeintel.serving.http.routes.meta import build_meta_router
from codeintel.serving.http.routes.profiles import build_profiles_router
from codeintel.serving.http.routes.subsystems import build_subsystem_router
from codeintel.serving.mcp import errors as mcp_errors
from codeintel.serving.mcp.models import ProblemDetail as ProblemDetailModel
from codeintel.serving.services.errors import ProblemDetail as DomainProblemDetail
from codeintel.serving.services.errors import ProblemError, generate_correlation_id
from codeintel.serving.services.factory import BackendResource, build_backend_resource
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway

LOG = logging.getLogger("codeintel.serving.http.fastapi")


def _ensure_readable_db(path: Path) -> None:
    """
    Validate that the DuckDB path exists and is readable.

    Parameters
    ----------
    path
        Path to the DuckDB database file.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the path is not a file.
    PermissionError
        If the file cannot be opened for reading.
    """
    if not path.exists():
        message = f"DuckDB database not found at {path}"
        raise FileNotFoundError(message)
    if not path.is_file():
        message = f"DuckDB path {path} is not a file"
        raise ValueError(message)
    try:
        with path.open("rb"):
            return
    except PermissionError as exc:
        message = f"DuckDB path {path} is not readable"
        raise PermissionError(message) from exc


def load_api_config() -> ServingConfig:
    """
    Load and validate server configuration from environment variables.

    Raises
    ------
    ValueError
        If required configuration is missing or incompatible with local DB mode.

    Returns
    -------
    ServingConfig
        Validated configuration for the FastAPI surface.
    """
    config = ServingConfig.from_env()
    if not config.repo:
        message = "CODEINTEL_REPO must be set for the FastAPI server"
        raise ValueError(message)
    if not config.commit:
        message = "CODEINTEL_COMMIT must be set for the FastAPI server"
        raise ValueError(message)
    if config.mode == "local_db":
        db_path = config.db_path
        if db_path is None:
            message = "CODEINTEL_DB_PATH is required when CODEINTEL_MCP_MODE='local_db'"
            raise ValueError(message)
        _ensure_readable_db(db_path)
    return config


def create_backend_resource(
    cfg: ServingConfig, *, gateway: StorageGateway | None = None
) -> BackendResource:
    """
    Instantiate the DuckDB backend for the API.

    Parameters
    ----------
    cfg
        Application configuration containing repo metadata and paths.
    gateway
        StorageGateway supplying the connection and dataset registry (required for local_db).

    Returns
    -------
    BackendResource
        Backend instance plus shutdown hook.

    Raises
    ------
    McpError
        If the query service cannot be constructed for the selected mode.
    """
    try:
        return build_backend_resource(cfg, gateway=gateway)
    except Exception as exc:
        problem_detail = mcp_errors.backend_failure(str(exc)).detail
        raise mcp_errors.McpError(problem_detail) from exc


def problem_response(detail: DomainProblemDetail) -> JSONResponse:
    """
    Convert a domain ProblemDetail payload into a JSON HTTP response.

    Parameters
    ----------
    detail
        Problem detail instance to serialize.

    Returns
    -------
    JSONResponse
        Response with RFC 7807 payload.
    """
    status_code = detail.status or status.HTTP_500_INTERNAL_SERVER_ERROR
    model = ProblemDetailModel.from_domain(detail)
    payload = model.model_dump()
    payload.setdefault("status", status_code)
    return JSONResponse(status_code=status_code, content=payload)


def install_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers for consistent Problem Details."""

    @app.exception_handler(mcp_errors.McpError)
    def _handle_mcp_error(
        _request: Request,
        exc: mcp_errors.McpError,
    ) -> JSONResponse:
        return problem_response(exc.detail)

    @app.exception_handler(RequestValidationError)
    def _handle_validation_error(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        problem = DomainProblemDetail(
            type="https://codeintel/problems/invalid-request",
            title="Invalid request",
            detail=str(exc),
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="invalid-request",
            extras={"errors": exc.errors()},
        )
        return problem_response(problem)

    @app.exception_handler(ProblemError)
    def _handle_problem_error(
        _request: Request,
        exc: ProblemError,
    ) -> JSONResponse:
        return problem_response(exc.detail)

    @app.exception_handler(Exception)
    def _handle_unexpected(
        _request: Request,
        exc: Exception,
    ) -> JSONResponse:
        problem = DomainProblemDetail(
            type="https://codeintel/problems/backend-failure",
            title="Backend failure",
            detail=str(exc),
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="backend-failure",
        )
        return problem_response(problem)


def install_logging_middleware(app: FastAPI) -> None:
    """Add structured logging for each request."""

    @app.middleware("http")
    async def _log_request(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        config: ServingConfig | None = getattr(request.app.state, "config", None)
        repo = config.repo if config is not None else "unknown"
        commit = config.commit if config is not None else "unknown"
        LOG.info(
            "Handled %s %s status=%s repo=%s commit=%s duration_ms=%.2f params=%s",
            request.method,
            request.url.path,
            response.status_code,
            repo,
            commit,
            duration_ms,
            dict(request.query_params),
        )
        return response


def register_routes(app: FastAPI) -> None:
    """Wire all API routes onto the provided FastAPI application."""
    app.include_router(build_functions_router())
    app.include_router(build_profiles_router())
    app.include_router(build_architecture_router())
    app.include_router(build_subsystem_router())
    app.include_router(build_ide_router())
    app.include_router(build_datasets_router())
    app.include_router(build_meta_router())
    app.include_router(build_health_router())


def create_app(
    *,
    config_loader: Callable[[], ServingConfig] = load_api_config,
    backend_factory: Callable[..., BackendResource] = create_backend_resource,
    gateway: StorageGateway | None = None,
) -> FastAPI:
    """
    Build the FastAPI application with configured lifecycle and routes.

    Parameters
    ----------
    config_loader
        Factory for loading application configuration.
    backend_factory
        Factory that yields a backend resource for the given configuration.
    gateway
        Optional StorageGateway to supply the connection/registry to the backend factory.

    Returns
    -------
    FastAPI
        Configured FastAPI instance.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config = config_loader()
        gw = gateway
        if gw is None and config.mode == "local_db":
            db_path = config.db_path or Path(":memory:")
            base_cfg = (
                StorageConfig.for_readonly(db_path)
                if config.read_only
                else StorageConfig.for_ingest(db_path)
            )
            gw_cfg = replace(
                base_cfg,
                repo=config.repo,
                commit=config.commit,
            )
            gw = open_gateway(gw_cfg)
        if gw is None and config.mode == "local_db":
            message = "StorageGateway is required for local_db FastAPI app"
            raise mcp_errors.backend_failure(message)
        backend_kwargs: dict[str, object] = {}
        params = inspect.signature(backend_factory).parameters
        if "gateway" in params:
            backend_kwargs["gateway"] = gw
        elif "_gateway" in params:
            backend_kwargs["_gateway"] = gw
        backend_resource = backend_factory(config, **backend_kwargs)
        app.state.config = config
        app.state.backend = backend_resource.backend
        app.state.service = backend_resource.service
        try:
            await asyncio.sleep(0)
            yield
        finally:
            backend_resource.close()

    app = FastAPI(
        title="CodeIntel Metadata API",
        description="Thin API over DuckDB views for AI agents and MCP clients.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def _inject_request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """
        Attach a RequestContext for each incoming HTTP request.

        Correlation ID is taken from X-Request-ID / X-Correlation-ID if provided,
        otherwise generated via generate_correlation_id().

        Returns
        -------
        Response
            Downstream response with correlation id echoed in headers.
        """
        correlation_id = (
            request.headers.get("X-Request-ID")
            or request.headers.get("X-Correlation-ID")
            or generate_correlation_id()
        )
        cfg: ServingConfig | None = getattr(request.app.state, "config", None)
        repo = getattr(cfg, "repo", None) if cfg is not None else None
        commit = getattr(cfg, "commit", None) if cfg is not None else None
        ctx = RequestContext(
            correlation_id=correlation_id,
            transport="http",
            operation=None,
            dataset=None,
            repo=repo,
            commit=commit,
            snapshot=None,
            graph_scope=None,
            client_id=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
        )
        token = set_current_request_context(ctx)
        try:
            response = await call_next(request)
        finally:
            reset_current_request_context(token)
        if hasattr(response, "headers"):
            response.headers.setdefault("X-Request-ID", correlation_id)
        return response

    install_exception_handlers(app)
    install_logging_middleware(app)
    register_routes(app)
    return app


app = create_app()
