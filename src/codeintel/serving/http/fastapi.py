"""FastAPI server exposing MCP-aligned queries over DuckDB."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field
from starlette.responses import Response

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.backend import DuckDBBackend, QueryBackend
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    DatasetDescriptor,
    DatasetRowsResponse,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    FileHintsResponse,
    FileProfileResponse,
    FileSummaryResponse,
    FunctionArchitectureResponse,
    FunctionProfileResponse,
    FunctionSummaryResponse,
    GraphNeighborhoodResponse,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    ModuleArchitectureResponse,
    ModuleProfileResponse,
    ModuleSubsystemResponse,
    ProblemDetail,
    SubsystemCoverageResponse,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemSummaryResponse,
    TestsForFunctionResponse,
)
from codeintel.serving.services.factory import BackendResource, build_backend_resource
from codeintel.serving.services.query_service import QueryService
from codeintel.storage.gateway import DuckDBError, StorageConfig, StorageGateway, open_gateway

LOG = logging.getLogger("codeintel.serving.http.fastapi")


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
    cfg:
        Application configuration containing repo metadata and paths.
    gateway:
        StorageGateway supplying the connection and dataset registry (required for local_db).

    Returns
    -------
    BackendResource
        Backend instance plus shutdown hook.

    Raises
    ------
    errors.backend_failure
        If the query service cannot be constructed for the selected mode.
    """
    try:
        return build_backend_resource(cfg, gateway=gateway)
    except Exception as exc:
        raise errors.backend_failure(str(exc)) from exc


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


def _compute_etag(payload: object) -> str:
    """
    Compute a weak ETag for a JSON-serializable payload.

    Returns
    -------
    str
        Weak ETag header value.
    """
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return f'W/"{hashlib.sha256(encoded).hexdigest()}"'


def get_app_config(request: Request) -> ServingConfig:
    """
    Retrieve the validated application configuration from state.

    Parameters
    ----------
    request:
        Incoming request providing access to application state.

    Returns
    -------
    ServingConfig
        Loaded application configuration.

    Raises
    ------
    errors.backend_failure
        If the configuration is missing.
    """
    config: ServingConfig | None = getattr(request.app.state, "config", None)
    if config is not None:
        return config
    message = "Server configuration is not initialized"
    raise errors.backend_failure(message)


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


def get_service(request: Request) -> QueryService:
    """
    Retrieve the shared query service from state.

    Parameters
    ----------
    request:
        Incoming request providing access to application state.

    Returns
    -------
    QueryService
        Service used to satisfy API queries.

    Raises
    ------
    errors.backend_failure
        If the service is missing.
    """
    service: QueryService | None = getattr(request.app.state, "service", None)
    if service is None:
        backend: QueryBackend | None = getattr(request.app.state, "backend", None)
        service = getattr(backend, "service", None) if backend is not None else None
    if service is None:
        message = "Query service is not initialized"
        raise errors.backend_failure(message)
    return service


ConfigDep = Annotated[ServingConfig, Depends(get_app_config)]
BackendDep = Annotated[QueryBackend, Depends(get_backend)]
ServiceDep = Annotated[QueryService, Depends(get_service)]


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
        service: ServiceDep,
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
        summary = service.get_function_summary(
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
        service: ServiceDep,
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
        result = service.list_high_risk_functions(
            min_risk=min_risk,
            limit=limit,
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
        service: ServiceDep,
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
        return service.get_callgraph_neighbors(
            goid_h128=goid_h128,
            direction=direction,
            limit=limit,
        )

    @router.get(
        "/function/tests",
        response_model=TestsForFunctionResponse,
        summary="List tests that exercise a function",
    )
    def tests_for_function(
        *,
        service: ServiceDep,
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
        return service.get_tests_for_function(
            goid_h128=goid_h128,
            urn=urn,
            limit=limit,
        )

    @router.get(
        "/graph/call/neighborhood",
        response_model=GraphNeighborhoodResponse,
        summary="Call graph ego neighborhood",
    )
    def callgraph_neighborhood(
        *,
        service: ServiceDep,
        goid_h128: int,
        radius: Annotated[int, Field(ge=1)] = 1,
        max_nodes: int | None = None,
    ) -> GraphNeighborhoodResponse:
        """
        Return a bounded ego neighborhood in the call graph.

        Parameters
        ----------
        service : ServiceDep
            Query service providing backend access.
        goid_h128 : int
            GOID of the function to center the neighborhood on.
        radius : int
            Hop radius (>=1).
        max_nodes : int, optional
            Optional node cap; defaults to service max_rows_per_call when omitted.

        Returns
        -------
        GraphNeighborhoodResponse
            Ego subgraph with truncation metadata.
        """
        response = service.get_callgraph_neighborhood(
            goid_h128=goid_h128, radius=radius, max_nodes=max_nodes
        )
        LOG.info(
            "callgraph_neighborhood repo=%s commit=%s goid=%s radius=%s applied_limit=%s "
            "truncated=%s",
            getattr(service, "repo", "unknown"),
            getattr(service, "commit", "unknown"),
            goid_h128,
            radius,
            response.meta.applied_limit,
            response.meta.truncated,
        )
        return response

    @router.get(
        "/graph/import/boundary",
        response_model=ImportBoundaryResponse,
        summary="Import graph edges crossing a subsystem",
    )
    def import_boundary(
        *,
        service: ServiceDep,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> ImportBoundaryResponse:
        """
        Return import edges crossing the given subsystem boundary.

        Parameters
        ----------
        service : ServiceDep
            Query service providing backend access.
        subsystem_id : str
            Subsystem identifier to inspect.
        max_edges : int, optional
            Optional edge cap; defaults to service max_rows_per_call.

        Returns
        -------
        ImportBoundaryResponse
            Boundary edges plus metadata describing truncation.
        """
        response = service.get_import_boundary(subsystem_id=subsystem_id, max_edges=max_edges)
        LOG.info(
            "import_boundary repo=%s commit=%s subsystem=%s applied_limit=%s truncated=%s",
            getattr(service, "repo", "unknown"),
            getattr(service, "commit", "unknown"),
            subsystem_id,
            response.meta.applied_limit,
            response.meta.truncated,
        )
        return response

    @router.get(
        "/file/summary",
        response_model=FileSummaryResponse,
        summary="Get file summary with function details",
    )
    def file_summary(
        *,
        service: ServiceDep,
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
        summary = service.get_file_summary(rel_path=rel_path)
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
        service: ServiceDep,
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
        profile = service.get_function_profile(goid_h128=goid_h128)
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
        service: ServiceDep,
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
        profile = service.get_file_profile(rel_path=rel_path)
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
        service: ServiceDep,
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
        profile = service.get_module_profile(module=module)
        if not profile.found or profile.profile is None:
            message = "Module profile not found"
            raise errors.not_found(message)
        return profile

    return router


def build_architecture_router() -> APIRouter:
    """
    Construct the router for architecture and subsystem endpoints.

    Returns
    -------
    APIRouter
        Router exposing architecture datasets without direct SQL.
    """
    router = APIRouter()

    @router.get(
        "/architecture/function",
        response_model=FunctionArchitectureResponse,
        summary="Get architecture metrics for a function",
    )
    def function_architecture(
        *,
        service: ServiceDep,
        goid_h128: int,
    ) -> FunctionArchitectureResponse:
        """
        Return call-graph architecture metrics for a function.

        Returns
        -------
        FunctionArchitectureResponse
            Architecture payload for the GOID.

        Raises
        ------
        errors.not_found
            If no architecture row exists for the GOID.
        """
        response = service.get_function_architecture(goid_h128=goid_h128)
        if not response.found or response.architecture is None:
            message = "Function architecture not found"
            raise errors.not_found(message)
        return response

    @router.get(
        "/architecture/module",
        response_model=ModuleArchitectureResponse,
        summary="Get architecture metrics for a module",
    )
    def module_architecture(
        *,
        service: ServiceDep,
        module: str,
    ) -> ModuleArchitectureResponse:
        """
        Return import-graph architecture metrics for a module.

        Returns
        -------
        ModuleArchitectureResponse
            Architecture payload for the module.

        Raises
        ------
        errors.not_found
            If no architecture row exists for the module.
        """
        response = service.get_module_architecture(module=module)
        if not response.found or response.architecture is None:
            message = "Module architecture not found"
            raise errors.not_found(message)
        return response

    return router


def build_subsystem_router() -> APIRouter:
    """
    Construct the router for subsystem endpoints.

    Returns
    -------
    APIRouter
        Router exposing subsystem docs views and membership helpers.
    """
    router = APIRouter()

    @router.get(
        "/architecture/subsystems",
        response_model=SubsystemSummaryResponse,
        summary="List inferred subsystems",
    )
    def list_subsystems(
        *,
        service: ServiceDep,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> SubsystemSummaryResponse:
        """
        List inferred subsystems derived from module coupling.

        Returns
        -------
        SubsystemSummaryResponse
            Subsystem rows and metadata.
        """
        return service.list_subsystems(limit=limit, role=role, q=q)

    @router.get(
        "/architecture/subsystem-profiles",
        response_model=SubsystemProfileResponse,
        summary="List subsystem profiles",
    )
    def list_subsystem_profiles(
        *,
        service: ServiceDep,
        limit: int | None = None,
    ) -> SubsystemProfileResponse:
        """
        List subsystem profiles backed by docs views.

        Returns
        -------
        SubsystemProfileResponse
            Profile rows with metadata.
        """
        return service.list_subsystem_profiles(limit=limit)

    @router.get(
        "/architecture/subsystem-coverage",
        response_model=SubsystemCoverageResponse,
        summary="List subsystem coverage rollups",
    )
    def list_subsystem_coverage(
        *,
        service: ServiceDep,
        limit: int | None = None,
    ) -> SubsystemCoverageResponse:
        """
        List subsystem coverage rollups derived from test profiles.

        Returns
        -------
        SubsystemCoverageResponse
            Coverage rows with metadata.
        """
        return service.list_subsystem_coverage(limit=limit)

    @router.get(
        "/architecture/module-subsystems",
        response_model=ModuleSubsystemResponse,
        summary="List subsystem memberships for a module",
    )
    def module_subsystems(
        *,
        service: ServiceDep,
        module: str,
    ) -> ModuleSubsystemResponse:
        """
        Return subsystem memberships for the requested module.

        Returns
        -------
        ModuleSubsystemResponse
            Membership rows and metadata.

        Raises
        ------
        errors.not_found
            If the module is not mapped to any subsystem.
        """
        response = service.get_module_subsystems(module=module)
        if not response.found or not response.memberships:
            message = "Module has no subsystem mappings"
            raise errors.not_found(message)
        return response

    @router.get(
        "/architecture/subsystem",
        response_model=SubsystemModulesResponse,
        summary="Get modules and detail for a subsystem",
    )
    def subsystem_modules(
        *,
        service: ServiceDep,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> SubsystemModulesResponse:
        """
        Return subsystem metadata and modules.

        Returns
        -------
        SubsystemModulesResponse
            Subsystem detail and member modules.

        Raises
        ------
        errors.not_found
            If the subsystem cannot be located.
        """
        response = service.summarize_subsystem(
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )
        if not response.found or response.subsystem is None:
            message = "Subsystem not found"
            raise errors.not_found(message)
        return response

    return router


def build_ide_router() -> APIRouter:
    """
    Construct the router for IDE-facing hint endpoints.

    Returns
    -------
    APIRouter
        Router exposing contextual hints for editor integrations.
    """
    router = APIRouter()

    @router.get(
        "/ide/hints",
        response_model=FileHintsResponse,
        summary="Get IDE hints for a file",
    )
    def ide_hints(
        *,
        service: ServiceDep,
        rel_path: str,
    ) -> FileHintsResponse:
        """
        Return subsystem and module context suitable for IDE tooltips.

        Returns
        -------
        FileHintsResponse
            Hint rows keyed by the provided relative path.

        Raises
        ------
        errors.not_found
            If no hints can be derived for the path.
        """
        response = service.get_file_hints(rel_path=rel_path)
        if not response.found or not response.hints:
            message = "IDE hints not found for path"
            raise errors.not_found(message)
        return response

    return router


def build_datasets_router() -> APIRouter:
    """
    Construct the router for dataset browsing endpoints.

    Returns
    -------
    APIRouter
        Router exposing dataset discovery and access endpoints.
    """

    def _filter_datasets(
        datasets: list[DatasetDescriptor],
        *,
        docs_view: Literal["include", "exclude", "only"],
        read_only: Literal["include", "exclude", "only"],
    ) -> list[DatasetDescriptor]:
        filtered: list[DatasetDescriptor] = []
        for ds in datasets:
            caps = ds.capabilities or {}
            is_docs = bool(caps.get("docs_view"))
            is_read_only = bool(caps.get("read_only"))
            docs_ok = (docs_view != "only" or is_docs) and (docs_view != "exclude" or not is_docs)
            read_only_ok = (read_only != "only" or is_read_only) and (
                read_only != "exclude" or not is_read_only
            )
            if docs_ok and read_only_ok:
                filtered.append(ds)
        return filtered

    router = APIRouter()

    @router.get("/datasets", response_model=list[DatasetDescriptor], summary="List datasets")
    def list_datasets(
        *,
        service: ServiceDep,
        request: Request,
        response: Response,
        docs_view: Literal["include", "exclude", "only"] = "include",
        read_only: Literal["include", "exclude", "only"] = "include",
    ) -> Response | list[DatasetDescriptor]:
        """
        Return dataset descriptors available through the backend.

        Returns
        -------
        list[DatasetDescriptor]
            Dataset descriptors sorted by name.
        """
        datasets = service.list_datasets()
        filtered = _filter_datasets(
            datasets,
            docs_view=docs_view,
            read_only=read_only,
        )
        payload = [ds.model_dump() for ds in filtered]
        etag = _compute_etag(payload)
        response.headers["Cache-Control"] = "public, max-age=60"
        response.headers["ETag"] = etag
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=response.headers)
        LOG.info(
            "Listed %d datasets (docs_view=%s read_only=%s)", len(filtered), docs_view, read_only
        )
        return filtered

    @router.get(
        "/datasets/specs",
        response_model=list[DatasetSpecDescriptor],
        summary="Describe dataset contract",
    )
    def list_dataset_specs(
        *,
        service: ServiceDep,
        request: Request,
        response: Response,
    ) -> Response | list[DatasetSpecDescriptor]:
        """
        Return canonical dataset specs including filenames and schema IDs.

        Returns
        -------
        list[DatasetSpecDescriptor]
            Dataset specs sorted by name.
        """
        specs = service.dataset_specs()
        payload = [spec.model_dump() for spec in specs]
        etag = _compute_etag(payload)
        response.headers["Cache-Control"] = "public, max-age=60"
        response.headers["ETag"] = etag
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=response.headers)
        LOG.info("Listed %d dataset specs", len(specs))
        return specs

    @router.get(
        "/datasets/{dataset_name}",
        response_model=DatasetRowsResponse,
        summary="Read rows from a dataset",
    )
    def read_dataset_rows(
        *,
        service: ServiceDep,
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
        resp = service.read_dataset_rows(dataset_name=dataset_name, limit=limit, offset=offset)
        LOG.info(
            "Read dataset=%s limit=%s offset=%s returned_rows=%d",
            dataset_name,
            limit,
            offset,
            len(resp.rows),
        )
        return resp

    @router.get(
        "/datasets/{dataset_name}/schema",
        response_model=DatasetSchemaResponse,
        summary="Describe dataset schema and samples",
    )
    def dataset_schema(
        *,
        service: ServiceDep,
        dataset_name: str,
        limit: int = 5,
    ) -> DatasetSchemaResponse:
        """
        Return schema metadata, JSON Schema (when present), and sample rows for a dataset.

        Returns
        -------
        DatasetSchemaResponse
            Schema detail payload.
        """
        detail = service.dataset_schema(dataset_name=dataset_name, sample_limit=limit)
        LOG.info("Returned schema detail for dataset=%s", dataset_name)
        return detail

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

        Raises
        ------
        errors.backend_failure
            If the backend connection is unavailable.
        """
        limits: dict[str, int] | None = None
        service = getattr(backend, "service", None)
        service_limits = getattr(service, "limits", None)
        if service_limits is None and hasattr(service, "query"):
            query_obj = getattr(service, "query", None)
            service_limits = getattr(query_obj, "limits", None)
        if service_limits is not None:
            limits = {
                "default_limit": service_limits.default_limit,
                "max_rows_per_call": service_limits.max_rows_per_call,
            }

        if isinstance(backend, DuckDBBackend):
            con = backend.gateway.con
            try:
                con.execute("SELECT 1;")
            except DuckDBError as exc:
                message = "Backend connection failed health probe."
                raise errors.backend_failure(message) from exc
        payload: dict[str, object] = {
            "status": "ok",
            "repo": config.repo,
            "commit": config.commit,
            "read_only": config.read_only,
        }
        if limits is not None:
            payload["limits"] = limits
        return payload

    return router


def register_routes(app: FastAPI) -> None:
    """Wire all API routes onto the provided FastAPI application."""
    app.include_router(build_functions_router())
    app.include_router(build_profiles_router())
    app.include_router(build_architecture_router())
    app.include_router(build_subsystem_router())
    app.include_router(build_ide_router())
    app.include_router(build_datasets_router())
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
    config_loader:
        Factory for loading application configuration.
    backend_factory:
        Factory that yields a backend resource for the given configuration.
    gateway:
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
            raise errors.backend_failure(message)
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

    install_exception_handlers(app)
    install_logging_middleware(app)
    register_routes(app)
    return app


app = create_app()
