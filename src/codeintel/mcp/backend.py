"""Backend implementations for MCP tools over DuckDB or HTTP."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import anyio
import duckdb
import httpx

from codeintel.mcp import errors
from codeintel.mcp.config import McpServerConfig
from codeintel.mcp.models import (
    CallGraphNeighborsResponse,
    DatasetDescriptor,
    DatasetRowsResponse,
    FileHintsResponse,
    FileProfileResponse,
    FileSummaryResponse,
    FunctionArchitectureResponse,
    FunctionProfileResponse,
    FunctionSummaryResponse,
    HighRiskFunctionsResponse,
    ModuleArchitectureResponse,
    ModuleProfileResponse,
    ModuleSubsystemResponse,
    ProblemDetail,
    SubsystemModulesResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    TestsForFunctionResponse,
)
from codeintel.mcp.query_service import BackendLimits, DuckDBQueryService
from codeintel.server.datasets import build_dataset_registry
from codeintel.services.factory import (
    DatasetRegistryOptions,
    build_service_from_config,
    get_observability_from_config,
)
from codeintel.services.query_service import (
    HttpQueryService,
    LocalQueryService,
    QueryService,
    ServiceObservability,
)
from codeintel.storage.gateway import StorageGateway

MAX_ROWS_LIMIT = BackendLimits().max_rows_per_call
HTTP_ERROR_STATUS = 400
LOG = logging.getLogger("codeintel.mcp.backend")


async def _aclose_client(client: httpx.AsyncClient) -> None:
    """Close an async HTTPX client."""
    await client.aclose()


async def _get_async(
    client: httpx.AsyncClient, path: str, params: dict[str, str]
) -> httpx.Response:
    """
    Perform an async GET request with parameters.

    Returns
    -------
    httpx.Response
        Response from the remote server.
    """
    return await client.get(path, params=params)


class QueryBackend(Protocol):
    """Abstract interface consumed by MCP tools."""

    service: QueryService

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> FunctionSummaryResponse:
        """Return a function summary from docs.v_function_summary."""
        ...

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
    ) -> HighRiskFunctionsResponse:
        """List high-risk functions from analytics.goid_risk_factors."""
        ...

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
    ) -> CallGraphNeighborsResponse:
        """Return incoming/outgoing call graph neighbors."""
        ...

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
    ) -> TestsForFunctionResponse:
        """List tests that exercised a function."""
        ...

    def get_file_summary(
        self,
        *,
        rel_path: str,
    ) -> FileSummaryResponse:
        """Return file summary plus function rows."""
        ...

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        """Return a denormalized function profile."""
        ...

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        """Return a denormalized file profile."""
        ...

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        """Return a module profile."""
        ...

    def get_function_architecture(self, *, goid_h128: int) -> FunctionArchitectureResponse:
        """Return call-graph architecture metrics for a function."""
        ...

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        """Return import-graph architecture metrics for a module."""
        ...

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSummaryResponse:
        """List inferred subsystems."""
        ...

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        """Return subsystem memberships for a module."""
        ...

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        """Return IDE-focused hints for a file (module + subsystem context)."""
        ...

    def get_subsystem_modules(self, *, subsystem_id: str) -> SubsystemModulesResponse:
        """Return subsystem details and module memberships."""
        ...

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSearchResponse:
        """Search-oriented subsystem listing."""
        ...

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """Summarize a subsystem with truncated module list."""
        ...

    def list_datasets(self) -> list[DatasetDescriptor]:
        """List datasets available to browse."""
        ...

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int = 100,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """Read rows from a dataset in small slices."""
        ...


def _require_identifier(
    *, urn: str | None = None, goid_h128: int | None = None, rel_path: str | None = None
) -> None:
    """
    Ensure at least one identifier is provided.

    Raises
    ------
    errors.invalid_argument
        When all identifiers are missing.
    """
    if urn is None and goid_h128 is None and rel_path is None:
        message = "At least one identifier (urn, goid_h128, rel_path) must be provided"
        raise errors.invalid_argument(message)


def _validate_direction(direction: str) -> str:
    """
    Validate direction argument for callgraph endpoints.

    Returns
    -------
    str
        Normalized direction value.

    Raises
    ------
    errors.invalid_argument
        When the direction is not supported.
    """
    if direction in {"incoming", "outgoing", "both"}:
        return direction
    message = "direction must be one of incoming, outgoing, both"
    raise errors.invalid_argument(message)


@dataclass
class DuckDBBackend(QueryBackend):
    """
    DuckDB-backed implementation of QueryBackend.

    Prefer injecting a StorageGateway to source the connection and dataset
    registry. Assumes a single repo/commit per DuckDB file, but repo/commit
    filters are still applied for future multi-repo support.
    """

    gateway: StorageGateway | None = None
    con: duckdb.DuckDBPyConnection | None = None
    repo: str | None = None
    commit: str | None = None
    limits: BackendLimits = field(default_factory=BackendLimits)
    dataset_tables: dict[str, str] | None = None
    observability: ServiceObservability | None = None
    service_override: LocalQueryService | None = None
    service: QueryService = field(init=False)
    query: DuckDBQueryService | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """
        Initialize the query service and dataset registry.

        Raises
        ------
        ValueError
            If the connection or repo/commit cannot be derived.
        """
        if self.service_override is not None:
            self.service = self.service_override
            if isinstance(self.service, LocalQueryService):
                self.query = self.service.query
            return

        gateway_con = self.gateway.con if self.gateway is not None else None
        con = self.con or gateway_con
        if con is None:
            message = "DuckDBBackend requires either a gateway or a DuckDB connection."
            raise ValueError(message)

        gateway_repo = self.gateway.config.repo if self.gateway is not None else None
        gateway_commit = self.gateway.config.commit if self.gateway is not None else None
        repo = self.repo or gateway_repo
        commit = self.commit or gateway_commit
        if repo is None or commit is None:
            message = "repo and commit must be provided either directly or via the gateway config."
            raise ValueError(message)

        registry_from_gateway = (
            dict(self.gateway.datasets.mapping) if self.gateway is not None else None
        )
        dataset_tables = self.dataset_tables or registry_from_gateway or build_dataset_registry()

        self.con = con
        self.repo = repo
        self.commit = commit
        self.dataset_tables = dataset_tables

        con_instance = self.con
        repo_value = self.repo
        commit_value = self.commit
        dataset_tables_value = self.dataset_tables
        if con_instance is None or repo_value is None or commit_value is None:
            message = "DuckDBBackend requires non-null con, repo, and commit after initialization."
            raise ValueError(message)
        if dataset_tables_value is None:
            message = "dataset_tables must be provided or derived from gateway/datasets."
            raise ValueError(message)

        self.query = DuckDBQueryService(
            con=con_instance,
            repo=repo_value,
            commit=commit_value,
            limits=self.limits,
            dataset_tables=dataset_tables_value,
        )
        self.service = LocalQueryService(
            query=self.query,
            dataset_tables=dataset_tables_value,
            observability=self.observability,
        )

    def __setattr__(self, name: str, value: object) -> None:
        """Propagate dataset registry changes to the underlying query service."""
        super().__setattr__(name, value)
        if name == "dataset_tables":
            tables = value if isinstance(value, dict) else {}
            query = getattr(self, "query", None)
            service = getattr(self, "service", None)
            if query is not None:
                query.dataset_tables = tables
            if isinstance(service, LocalQueryService):
                service.dataset_tables = tables

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> FunctionSummaryResponse:
        """
        Return a function summary from the DuckDB-backed query service.

        Returns
        -------
        FunctionSummaryResponse
            Summary payload with found flag and metadata.
        """
        _require_identifier(urn=urn, goid_h128=goid_h128, rel_path=rel_path)
        return self.service.get_function_summary(
            urn=urn,
            goid_h128=goid_h128,
            rel_path=rel_path,
            qualname=qualname,
        )

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
    ) -> HighRiskFunctionsResponse:
        """
        List high-risk functions for the configured repo/commit.

        Returns
        -------
        HighRiskFunctionsResponse
            High-risk functions plus truncation metadata.
        """
        return self.service.list_high_risk_functions(
            min_risk=min_risk,
            limit=limit,
            tested_only=tested_only,
        )

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
    ) -> CallGraphNeighborsResponse:
        """
        Return call graph neighbors for a function GOID.

        Returns
        -------
        CallGraphNeighborsResponse
            Incoming and outgoing edges with metadata.
        """
        direction = _validate_direction(direction)
        return self.service.get_callgraph_neighbors(
            goid_h128=goid_h128,
            direction=direction,
            limit=limit,
        )

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
    ) -> TestsForFunctionResponse:
        """
        Return tests linked to a function.

        Returns
        -------
        TestsForFunctionResponse
            Tests exercising the function plus messages.
        """
        _require_identifier(urn=urn, goid_h128=goid_h128)
        return self.service.get_tests_for_function(
            goid_h128=goid_h128,
            urn=urn,
            limit=limit,
        )

    def get_file_summary(
        self,
        *,
        rel_path: str,
    ) -> FileSummaryResponse:
        """
        Return file summary plus function rows.

        Returns
        -------
        FileSummaryResponse
            File-level summary and nested function entries.
        """
        return self.service.get_file_summary(rel_path=rel_path)

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        """
        Return a denormalized function profile.

        Returns
        -------
        FunctionProfileResponse
            Profile payload and found flag.
        """
        return self.service.get_function_profile(goid_h128=goid_h128)

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        """
        Return a denormalized file profile.

        Returns
        -------
        FileProfileResponse
            Profile payload and found flag.
        """
        return self.service.get_file_profile(rel_path=rel_path)

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        """
        Return a module profile for the configured repo/commit.

        Returns
        -------
        ModuleProfileResponse
            Profile payload and found flag.
        """
        return self.service.get_module_profile(module=module)

    def get_function_architecture(self, *, goid_h128: int) -> FunctionArchitectureResponse:
        """
        Return call-graph architecture metrics for a function.

        Returns
        -------
        FunctionArchitectureResponse
            Architecture payload and found flag.
        """
        return self.service.get_function_architecture(goid_h128=goid_h128)

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        """
        Return import-graph and symbol-coupling metrics for a module.

        Returns
        -------
        ModuleArchitectureResponse
            Architecture payload and found flag.
        """
        return self.service.get_module_architecture(module=module)

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSummaryResponse:
        """
        List inferred subsystems for the current repo/commit.

        Returns
        -------
        SubsystemSummaryResponse
            Subsystem rows and metadata.
        """
        return self.service.list_subsystems(limit=limit, role=role, q=q)

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        """
        Return subsystem memberships for a module.

        Returns
        -------
        ModuleSubsystemResponse
            Membership rows and metadata.
        """
        return self.service.get_module_subsystems(module=module)

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        """
        Return IDE-focused hints for a file path.

        Returns
        -------
        FileHintsResponse
            Hints including subsystem context and module metrics.
        """
        return self.service.get_file_hints(rel_path=rel_path)

    def get_subsystem_modules(self, *, subsystem_id: str) -> SubsystemModulesResponse:
        """
        Return subsystem details and module memberships.

        Returns
        -------
        SubsystemModulesResponse
            Subsystem detail payload.
        """
        return self.service.get_subsystem_modules(subsystem_id=subsystem_id)

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSearchResponse:
        """
        Search subsystems with optional role/name filters.

        Returns
        -------
        SubsystemSearchResponse
            Subsystem rows and metadata.
        """
        return self.service.search_subsystems(limit=limit, role=role, q=q)

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """
        Summarize a subsystem with optional module truncation.

        Returns
        -------
        SubsystemModulesResponse
            Subsystem detail payload.
        """
        return self.service.summarize_subsystem(
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )

    def list_datasets(self) -> list[DatasetDescriptor]:
        """
        List datasets exposed by the backend.

        Returns
        -------
        list[DatasetDescriptor]
            Dataset metadata entries.
        """
        return self.service.list_datasets()

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int = 100,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """
        Read a slice of rows from a dataset.

        Returns
        -------
        DatasetRowsResponse
            Dataset slice payload with metadata.
        """
        return self.service.read_dataset_rows(
            dataset_name=dataset_name,
            limit=limit,
            offset=offset,
        )


@dataclass  # noqa: PLR0904
class HttpBackend(QueryBackend):
    """HTTP-backed QueryBackend that talks to the FastAPI server."""

    base_url: str
    repo: str
    commit: str
    timeout: float
    limits: BackendLimits
    client: httpx.Client | httpx.AsyncClient | None = None
    _owns_client: bool = field(init=False, default=False)
    _async_client: bool = field(init=False, default=False)
    observability: ServiceObservability | None = None
    service_override: HttpQueryService | None = None
    service: QueryService = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the HTTP client and verify server health."""
        if self.client is None:
            self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
            self._owns_client = True
            self._async_client = False
        else:
            self._owns_client = False
            self._async_client = isinstance(self.client, httpx.AsyncClient)

        self._verify_health()
        if self.service_override is not None:
            self.service = self.service_override
            return

        self.service = HttpQueryService(
            self._request_json,
            self.limits,
            observability=self.observability,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        client = self.client
        if not self._owns_client or client is None:
            return
        if isinstance(client, httpx.AsyncClient):
            anyio.run(_aclose_client, client)
            return
        if isinstance(client, httpx.Client):
            client.close()

    def _request_json(self, path: str, params: dict[str, object]) -> object:
        if self.client is None:
            message = "HTTP client is not initialized"
            raise errors.backend_failure(message)
        filtered_params = {k: v for k, v in params.items() if v is not None}
        normalized_params = {k: str(v) for k, v in filtered_params.items()}
        client = self.client
        if isinstance(client, httpx.AsyncClient):
            response = anyio.run(_get_async, client, path, normalized_params)
        else:
            response = client.get(path, params=normalized_params)
        if response.status_code >= HTTP_ERROR_STATUS:
            payload = response.json()
            problem = ProblemDetail.model_validate(payload)
            raise errors.McpError(detail=problem)
        return response.json()

    def request_json(self, path: str, params: dict[str, object]) -> object:
        """
        Public wrapper around HTTP GET to avoid private attribute access.

        Parameters
        ----------
        path:
            API path to fetch.
        params:
            Query parameters to include in the request.

        Returns
        -------
        object
            Decoded JSON payload.
        """
        return self._request_json(path, params)

    def _verify_health(self) -> None:
        """
        Verify remote API health once at startup.

        Raises
        ------
        errors.backend_failure
            When the health endpoint cannot be reached or returns an error.
        """
        try:
            _ = self._request_json("/health", {})
        except Exception as exc:
            message = "Failed to verify remote API health"
            raise errors.backend_failure(message) from exc

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> FunctionSummaryResponse:
        """
        Return a function summary from the remote API.

        Returns
        -------
        FunctionSummaryResponse
            Summary payload with found flag.
        """
        data = self._request_json(
            "/function/summary",
            {"urn": urn, "goid_h128": goid_h128, "rel_path": rel_path, "qualname": qualname},
        )
        return FunctionSummaryResponse.model_validate(data)

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
    ) -> HighRiskFunctionsResponse:
        """
        List high-risk functions from the remote API.

        Returns
        -------
        HighRiskFunctionsResponse
            Functions ordered by risk with truncation metadata.
        """
        return self.service.list_high_risk_functions(
            min_risk=min_risk,
            limit=limit,
            tested_only=tested_only,
        )

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
    ) -> CallGraphNeighborsResponse:
        """
        Return call graph neighbors for a function GOID.

        Returns
        -------
        CallGraphNeighborsResponse
            Neighbor edges and metadata.
        """
        return self.service.get_callgraph_neighbors(
            goid_h128=goid_h128,
            direction=direction,
            limit=limit,
        )

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
    ) -> TestsForFunctionResponse:
        """
        Return tests linked to a function.

        Returns
        -------
        TestsForFunctionResponse
            Tests hitting the function plus messages.
        """
        return self.service.get_tests_for_function(
            goid_h128=goid_h128,
            urn=urn,
            limit=limit,
        )

    def get_file_summary(
        self,
        *,
        rel_path: str,
    ) -> FileSummaryResponse:
        """
        Return a file summary from the remote API.

        Returns
        -------
        FileSummaryResponse
            File-level summary payload with functions.
        """
        return self.service.get_file_summary(rel_path=rel_path)

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        """
        Return a function profile from the remote API.

        Returns
        -------
        FunctionProfileResponse
            Profile payload including found flag.
        """
        return self.service.get_function_profile(goid_h128=goid_h128)

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        """
        Return a file profile from the remote API.

        Returns
        -------
        FileProfileResponse
            Profile payload including found flag.
        """
        return self.service.get_file_profile(rel_path=rel_path)

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        """
        Return a module profile from the remote API.

        Returns
        -------
        ModuleProfileResponse
            Profile payload including found flag.
        """
        return self.service.get_module_profile(module=module)

    def get_function_architecture(self, *, goid_h128: int) -> FunctionArchitectureResponse:
        """
        Return function architecture metrics from the remote API.

        Returns
        -------
        FunctionArchitectureResponse
            Architecture payload including found flag.
        """
        return self.service.get_function_architecture(goid_h128=goid_h128)

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        """
        Return module architecture metrics from the remote API.

        Returns
        -------
        ModuleArchitectureResponse
            Architecture payload including found flag.
        """
        return self.service.get_module_architecture(module=module)

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSummaryResponse:
        """
        List subsystems from the remote API.

        Returns
        -------
        SubsystemSummaryResponse
            Subsystem rows and metadata.
        """
        return self.service.list_subsystems(limit=limit, role=role, q=q)

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        """
        Return subsystem memberships for a module from the remote API.

        Returns
        -------
        ModuleSubsystemResponse
            Membership rows and metadata.
        """
        return self.service.get_module_subsystems(module=module)

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        """
        Return IDE-focused hints for a file from the remote API.

        Returns
        -------
        FileHintsResponse
            Hint rows and metadata for the path.
        """
        return self.service.get_file_hints(rel_path=rel_path)

    def get_subsystem_modules(self, *, subsystem_id: str) -> SubsystemModulesResponse:
        """
        Return subsystem details and module memberships from the remote API.

        Returns
        -------
        SubsystemModulesResponse
            Subsystem detail payload.
        """
        return self.service.get_subsystem_modules(subsystem_id=subsystem_id)

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSearchResponse:
        """
        Search subsystems from the remote API.

        Returns
        -------
        SubsystemSearchResponse
            Subsystem rows and metadata.
        """
        return self.service.search_subsystems(limit=limit, role=role, q=q)

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """
        Summarize subsystem detail with optional module truncation.

        Returns
        -------
        SubsystemModulesResponse
            Subsystem detail payload.
        """
        return self.service.summarize_subsystem(
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )

    def list_datasets(self) -> list[DatasetDescriptor]:
        """
        List datasets available from the remote API.

        Returns
        -------
        list[DatasetDescriptor]
            Dataset descriptors provided by the API.
        """
        return self.service.list_datasets()

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int = 100,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """
        Read dataset rows from the remote API.

        Returns
        -------
        DatasetRowsResponse
            Dataset slice payload with metadata.
        """
        return self.service.read_dataset_rows(
            dataset_name=dataset_name,
            limit=limit,
            offset=offset,
        )


def create_backend(
    cfg: McpServerConfig,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    gateway: StorageGateway | None = None,
    registry: DatasetRegistryOptions | dict[str, str] | None = None,
    observability: ServiceObservability | None = None,
) -> QueryBackend:
    """
    Create a QueryBackend using local DuckDB or remote HTTP.

    Parameters
    ----------
    cfg:
        Server configuration loaded from env or caller.
    con:
        Optional DuckDB connection to reuse in local_db mode.
    gateway:
        Optional StorageGateway to supply connection and dataset registry in local_db mode.
    registry:
        Optional dataset registry override or options.
    observability:
        Optional observability configuration for structured logging.

    Returns
    -------
    QueryBackend
        Backend bound to the selected transport.

    Raises
    ------
    ValueError
        If required configuration is missing or the mode is unsupported.
    """
    limits = BackendLimits.from_config(cfg)
    registry_from_gateway = dict(gateway.datasets.mapping) if gateway is not None else None
    registry_tables = registry.tables if isinstance(registry, DatasetRegistryOptions) else registry
    tables = (
        registry_tables
        if isinstance(registry_tables, dict)
        else registry_from_gateway
        if registry_from_gateway is not None
        else build_dataset_registry()
    )
    registry_opts = registry if isinstance(registry, DatasetRegistryOptions) else None
    registry_opts = registry_opts or DatasetRegistryOptions(tables=tables)
    resolved_observability = observability or get_observability_from_config(cfg)
    if cfg.mode == "local_db":
        if cfg.db_path is None and con is None and gateway is None:
            message = "db_path is required for local_db backend"
            raise ValueError(message)
        connection = con or (gateway.con if gateway is not None else None)
        if connection is None and cfg.db_path is not None:
            connection = duckdb.connect(str(cfg.db_path), read_only=True)
        if connection is None:
            message = "Unable to open DuckDB connection for local_db backend"
            raise ValueError(message)
        service = build_service_from_config(
            cfg,
            con=connection,
            registry=registry_opts,
            observability=resolved_observability,
        )
        return DuckDBBackend(
            gateway=gateway,
            con=connection,
            repo=cfg.repo,
            commit=cfg.commit,
            limits=limits,
            dataset_tables=tables,
            observability=resolved_observability,
            service_override=service if isinstance(service, LocalQueryService) else None,
        )
    if cfg.mode == "remote_api":
        if not cfg.api_base_url:
            message = "api_base_url is required for remote_api backend"
            raise ValueError(message)
        return HttpBackend(
            base_url=cfg.api_base_url,
            repo=cfg.repo,
            commit=cfg.commit,
            timeout=cfg.timeout_seconds,
            limits=limits,
            observability=resolved_observability,
        )
    message = f"Unsupported MCP mode: {cfg.mode}"
    raise ValueError(message)
