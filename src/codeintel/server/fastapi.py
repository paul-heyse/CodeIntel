"""FastAPI server exposing MCP-aligned queries over DuckDB."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.responses import Response

from codeintel.mcp import errors
from codeintel.mcp.backend import DuckDBBackend, QueryBackend, create_backend
from codeintel.mcp.config import McpServerConfig
from codeintel.mcp.models import (
    CallGraphNeighborsResponse,
    DatasetDescriptor,
    DatasetRowsResponse,
    FileProfileResponse,
    FileSummaryResponse,
    FunctionProfileResponse,
    FunctionSummaryResponse,
    HighRiskFunctionsResponse,
    ModuleProfileResponse,
    ProblemDetail,
    TestsForFunctionResponse,
)
from codeintel.server.datasets import build_dataset_registry
from codeintel.storage.duckdb_client import DuckDBClient, DuckDBConfig
from codeintel.storage.views import create_all_views

LOG = logging.getLogger("codeintel.server.fastapi")


@dataclass
class ApiAppConfig:
    """Application configuration for the FastAPI server."""

    server: McpServerConfig
    read_only: bool


@dataclass
class BackendResource:
    """Backend instance plus cleanup hook."""

    backend: QueryBackend
    close: Callable[[], None]


def _env_flag(name: str, *, default: bool) -> bool:
    """
    Interpret an environment variable as a boolean.

    Parameters
    ----------
    name:
        Environment variable name.
    default:
        Default value when the variable is not set.

    Returns
    -------
    bool
        True when the variable is set to a truthy value.
    """
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.lower() not in {"0", "false", "no", "off"}


def _ensure_readable_db(path: Path) -> None:
    """
    Validate that the DuckDB path exists and is readable.

    Parameters
    ----------
    path:
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


def load_api_config() -> ApiAppConfig:
    """
    Load and validate server configuration from environment variables.

    Raises
    ------
    ValueError
        If required configuration is missing or incompatible with local DB mode.

    Returns
    -------
    ApiAppConfig
        Validated configuration including read-only mode.
    """
    server_cfg = McpServerConfig.from_env()
    if not server_cfg.repo:
        message = "CODEINTEL_REPO must be set for the FastAPI server"
        raise ValueError(message)
    if not server_cfg.commit:
        message = "CODEINTEL_COMMIT must be set for the FastAPI server"
        raise ValueError(message)
    if server_cfg.mode == "local_db":
        db_path = server_cfg.db_path
        if db_path is None:
            message = "CODEINTEL_DB_PATH is required when CODEINTEL_MCP_MODE='local_db'"
            raise ValueError(message)
        _ensure_readable_db(db_path)
    elif server_cfg.mode == "remote_api":
        if not server_cfg.api_base_url:
            message = "CODEINTEL_API_BASE_URL is required when CODEINTEL_MCP_MODE='remote_api'"
            raise ValueError(message)
    else:
        message = f"Unsupported CODEINTEL_MCP_MODE: {server_cfg.mode}"
        raise ValueError(message)

    read_only = _env_flag("CODEINTEL_API_READ_ONLY", default=True)
    return ApiAppConfig(server=server_cfg, read_only=read_only)


def create_backend_resource(cfg: ApiAppConfig) -> BackendResource:
    """
    Instantiate the DuckDB backend for the API.

    Parameters
    ----------
    cfg:
        Application configuration containing repo metadata and paths.

    Returns
    -------
    BackendResource
        Backend instance plus shutdown hook.

    Raises
    ------
    ValueError
        If the database path is missing after configuration validation.
    """
    dataset_registry = build_dataset_registry()
    if cfg.server.mode == "local_db":
        db_path = cfg.server.db_path
        if db_path is None:
            message = "db_path cannot be None after validation"
            raise ValueError(message)

        client = DuckDBClient(DuckDBConfig(db_path=db_path, read_only=cfg.read_only))
        connection = client.con
        if not cfg.read_only:
            create_all_views(connection)

        backend = create_backend(
            cfg.server,
            con=connection,
            dataset_tables=dataset_registry,
        )
        return BackendResource(backend=backend, close=client.close)

    backend = create_backend(cfg.server, dataset_tables=dataset_registry)
    close = getattr(backend, "close", lambda: None)
    return BackendResource(backend=backend, close=close)


def problem_response(detail: ProblemDetail) -> JSONResponse:
    """
    Convert a ProblemDetail payload into a JSON HTTP response.

    Parameters
    ----------
    detail:
        Problem detail instance to serialize.

    Returns
    -------
    JSONResponse
        Response with RFC 7807 payload.
    """
    status_code = detail.status or status.HTTP_500_INTERNAL_SERVER_ERROR
    payload = detail.model_dump()
    payload.setdefault("status", status_code)
    return JSONResponse(status_code=status_code, content=payload)


def install_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers for consistent Problem Details."""

    @app.exception_handler(errors.McpError)
    def _handle_mcp_error(
        _request: Request,
        exc: errors.McpError,
    ) -> JSONResponse:
        return problem_response(exc.detail)

    @app.exception_handler(RequestValidationError)
    def _handle_validation_error(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        problem = ProblemDetail(
            type="https://example.com/problems/validation-error",
            title="Invalid request",
            detail="Request validation failed",
            status=status.HTTP_400_BAD_REQUEST,
            data={"errors": exc.errors()},
        )
        return problem_response(problem)

    @app.exception_handler(Exception)
    def _handle_unexpected(
        _request: Request,
        exc: Exception,
    ) -> JSONResponse:
        problem = ProblemDetail(
            type="https://example.com/problems/backend-failure",
            title="Backend failure",
            detail=str(exc),
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
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

        config = getattr(request.app.state, "config", None)
        repo = config.server.repo if isinstance(config, ApiAppConfig) else "unknown"
        commit = config.server.commit if isinstance(config, ApiAppConfig) else "unknown"

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


def get_app_config(request: Request) -> ApiAppConfig:
    """
    Retrieve the validated application configuration from state.

    Parameters
    ----------
    request:
        Incoming request providing access to application state.

    Returns
    -------
    ApiAppConfig
        Loaded application configuration.

    Raises
    ------
    errors.backend_failure
        If the configuration is missing.
    """
    config = getattr(request.app.state, "config", None)
    if not isinstance(config, ApiAppConfig):
        message = "Server configuration is not initialized"
        raise errors.backend_failure(message)
    return config


def get_backend(request: Request) -> QueryBackend:
    """
    Retrieve the shared backend from state.

    Parameters
    ----------
    request:
        Incoming request providing access to application state.

    Returns
    -------
    DuckDBBackend
        Backend connected to the configured DuckDB instance.

    Raises
    ------
    errors.backend_failure
        If the backend is missing.
    """
    backend: QueryBackend | None = getattr(request.app.state, "backend", None)
    if backend is None:
        message = "Backend is not initialized"
        raise errors.backend_failure(message)
    return backend


ConfigDep = Annotated[ApiAppConfig, Depends(get_app_config)]
BackendDep = Annotated[QueryBackend, Depends(get_backend)]


def build_functions_router() -> APIRouter:
    """
    Construct the router for function-centric endpoints.

    Returns
    -------
    APIRouter
        Router exposing function metadata endpoints.
    """
    router = APIRouter()

    @router.get(
        "/function/summary",
        response_model=FunctionSummaryResponse,
        summary="Get function summary",
    )
    def function_summary(
        *,
        backend: BackendDep,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> FunctionSummaryResponse:
        """
        Return a function summary identified by GOID, URN, or path.

        Returns
        -------
        FunctionSummaryResponse
            Summary payload describing the requested function.

        Raises
        ------
        errors.not_found
            If the function cannot be located.
        """
        summary = backend.get_function_summary(
            urn=urn,
            goid_h128=goid_h128,
            rel_path=rel_path,
            qualname=qualname,
        )
        if not summary.found or summary.summary is None:
            message = "Function not found"
            raise errors.not_found(message)
        return summary

    @router.get(
        "/functions/high-risk",
        response_model=HighRiskFunctionsResponse,
        summary="List high-risk functions",
    )
    def list_high_risk_functions(
        *,
        backend: BackendDep,
        config: ConfigDep,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
    ) -> HighRiskFunctionsResponse:
        """
        List high-risk functions with optional tested-only filtering.

        Returns
        -------
        HighRiskFunctionsResponse
            High-risk functions and truncation flag.
        """
        result = backend.list_high_risk_functions(
            min_risk=min_risk,
            limit=config.server.default_limit if limit is None else limit,
            tested_only=tested_only,
        )
        return HighRiskFunctionsResponse(functions=result.functions, truncated=result.truncated)

    @router.get(
        "/function/callgraph",
        response_model=CallGraphNeighborsResponse,
        summary="Get call graph neighbors for a function",
    )
    def function_callgraph(
        *,
        backend: BackendDep,
        config: ConfigDep,
        goid_h128: int,
        direction: Literal["in", "out", "both"] = "both",
        limit: int | None = None,
    ) -> CallGraphNeighborsResponse:
        """
        Return incoming and outgoing neighbors for a function.

        Returns
        -------
        CallGraphNeighborsResponse
            Incoming and outgoing edges adjacent to the function.
        """
        return backend.get_callgraph_neighbors(
            goid_h128=goid_h128,
            direction=direction,
            limit=config.server.default_limit if limit is None else limit,
        )

    @router.get(
        "/function/tests",
        response_model=TestsForFunctionResponse,
        summary="List tests that exercise a function",
    )
    def tests_for_function(
        *,
        backend: BackendDep,
        config: ConfigDep,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
    ) -> TestsForFunctionResponse:
        """
        List tests that exercised the requested function.

        Returns
        -------
        TestsForFunctionResponse
            Tests linked to the requested function.
        """
        return backend.get_tests_for_function(
            goid_h128=goid_h128,
            urn=urn,
            limit=config.server.default_limit if limit is None else limit,
        )

    @router.get(
        "/file/summary",
        response_model=FileSummaryResponse,
        summary="Get file summary with function details",
    )
    def file_summary(
        *,
        backend: BackendDep,
        rel_path: str,
    ) -> FileSummaryResponse:
        """
        Return file-level metrics plus function summaries.

        Returns
        -------
        FileSummaryResponse
            File summary and nested function details.

        Raises
        ------
        errors.not_found
            If the file cannot be located in metadata tables.
        """
        summary = backend.get_file_summary(rel_path=rel_path)
        if not summary.found or summary.file is None:
            message = "File not found"
            raise errors.not_found(message)
        return summary

    return router


def build_profiles_router() -> APIRouter:
    """
    Construct the router for profile endpoints.

    Returns
    -------
    APIRouter
        Router exposing function, file, and module profiles.
    """
    router = APIRouter()

    @router.get(
        "/profiles/function",
        response_model=FunctionProfileResponse,
        summary="Get a function profile",
    )
    def function_profile(
        *,
        backend: BackendDep,
        goid_h128: int,
    ) -> FunctionProfileResponse:
        """
        Return a denormalized function profile for the given GOID.

        Returns
        -------
        FunctionProfileResponse
            Profile payload for the requested GOID.

        Raises
        ------
        errors.not_found
            If the profile cannot be located.
        """
        profile = backend.get_function_profile(goid_h128=goid_h128)
        if not profile.found or profile.profile is None:
            message = "Function profile not found"
            raise errors.not_found(message)
        return profile

    @router.get(
        "/profiles/file",
        response_model=FileProfileResponse,
        summary="Get a file profile",
    )
    def file_profile(
        *,
        backend: BackendDep,
        rel_path: str,
    ) -> FileProfileResponse:
        """
        Return a denormalized profile for a file path.

        Returns
        -------
        FileProfileResponse
            Profile payload for the requested file.

        Raises
        ------
        errors.not_found
            If the profile cannot be located.
        """
        profile = backend.get_file_profile(rel_path=rel_path)
        if not profile.found or profile.profile is None:
            message = "File profile not found"
            raise errors.not_found(message)
        return profile

    @router.get(
        "/profiles/module",
        response_model=ModuleProfileResponse,
        summary="Get a module profile",
    )
    def module_profile(
        *,
        backend: BackendDep,
        module: str,
    ) -> ModuleProfileResponse:
        """
        Return a module-level profile including coverage and import metrics.

        Returns
        -------
        ModuleProfileResponse
            Profile payload for the requested module.

        Raises
        ------
        errors.not_found
            If the profile cannot be located.
        """
        profile = backend.get_module_profile(module=module)
        if not profile.found or profile.profile is None:
            message = "Module profile not found"
            raise errors.not_found(message)
        return profile

    return router


def build_datasets_router() -> APIRouter:
    """
    Construct the router for dataset browsing endpoints.

    Returns
    -------
    APIRouter
        Router exposing dataset discovery and access endpoints.
    """
    router = APIRouter()

    @router.get("/datasets", response_model=list[DatasetDescriptor], summary="List datasets")
    def list_datasets(*, backend: BackendDep) -> list[DatasetDescriptor]:
        """
        Return dataset descriptors available through the backend.

        Returns
        -------
        list[DatasetDescriptor]
            Dataset descriptors sorted by name.
        """
        return backend.list_datasets()

    @router.get(
        "/datasets/{dataset_name}",
        response_model=DatasetRowsResponse,
        summary="Read rows from a dataset",
    )
    def read_dataset_rows(
        *,
        backend: BackendDep,
        config: ConfigDep,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """
        Read a window of rows from a configured dataset.

        Returns
        -------
        DatasetRowsResponse
            Dataset slice with pagination metadata.
        """
        return backend.read_dataset_rows(
            dataset_name=dataset_name,
            limit=config.server.default_limit if limit is None else limit,
            offset=offset,
        )

    return router


def build_health_router() -> APIRouter:
    """
    Construct the router for health and diagnostics endpoints.

    Returns
    -------
    APIRouter
        Router exposing health status endpoints.
    """
    router = APIRouter()

    @router.get(
        "/health",
        summary="Health check for CodeIntel API",
    )
    def health(
        *,
        backend: BackendDep,
        config: ConfigDep,
    ) -> dict[str, object]:
        """
        Report server health and connectivity.

        Returns
        -------
        dict[str, object]
            Health payload including repo/commit and read-only state.
        """
        if isinstance(backend, DuckDBBackend):
            backend.con.execute("SELECT 1;")
        return {
            "status": "ok",
            "repo": config.server.repo,
            "commit": config.server.commit,
            "read_only": config.read_only,
        }

    return router


def register_routes(app: FastAPI) -> None:
    """Wire all API routes onto the provided FastAPI application."""
    app.include_router(build_functions_router())
    app.include_router(build_profiles_router())
    app.include_router(build_datasets_router())
    app.include_router(build_health_router())


def create_app(
    *,
    config_loader: Callable[[], ApiAppConfig] = load_api_config,
    backend_factory: Callable[[ApiAppConfig], BackendResource] = create_backend_resource,
) -> FastAPI:
    """
    Build the FastAPI application with configured lifecycle and routes.

    Parameters
    ----------
    config_loader:
        Factory for loading application configuration.
    backend_factory:
        Factory that yields a backend resource for the given configuration.

    Returns
    -------
    FastAPI
        Configured FastAPI instance.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config = config_loader()
        backend_resource = backend_factory(config)
        app.state.config = config
        app.state.backend = backend_resource.backend
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

    install_exception_handlers(app)
    install_logging_middleware(app)
    register_routes(app)
    return app


app = create_app()
