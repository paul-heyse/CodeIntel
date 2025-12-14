"""Backend implementations for MCP tools over DuckDB or HTTP."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, cast

import anyio
import httpx

from codeintel.serving.backend import (
    BackendLimits,
)
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.backend_dispatch import BackendDispatchMixin
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    DatasetDescriptor,
    DatasetRowsResponse,
    DatasetSchemaResponse,
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
    SubsystemModulesResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    TestsForFunctionResponse,
)
from codeintel.serving.mcp.models import (
    ProblemDetail as ProblemDetailModel,
)
from codeintel.serving.services.errors import DatasetNotFoundError, ProblemError
from codeintel.serving.services.query_service import (
    HttpQueryService,
    LocalQueryService,
)
from codeintel.serving.types import QueryBackendProtocol

if TYPE_CHECKING:
    from codeintel.graphs.engine import GraphEngine
    from codeintel.serving.backend import (
        DuckDBQueryService,
    )
    from codeintel.serving.backend.query_api import DuckDBQueryApi
    from codeintel.serving.mcp.models import (
        DatasetSpecDescriptor,
    )
    from codeintel.serving.services.query_service import (
        QueryService,
        ServiceObservability,
    )
    from codeintel.storage.gateway import StorageGateway

MAX_ROWS_LIMIT = BackendLimits().max_rows_per_call
HTTP_ERROR_STATUS = 400
RETRYABLE_MIN_STATUS = 500
LOG = logging.getLogger("codeintel.serving.mcp.backend")


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


QueryBackend = QueryBackendProtocol


class DatasetBackendMixin:
    """Common dataset helpers shared by backend implementations.

    This mixin provides dataset-related methods for both DuckDBBackend and
    HttpBackend. It relies on concrete classes also inheriting from
    BackendDispatchMixin, which provides the ``is_local`` property.

    The ``_dispatch_dataset`` helper method is similar to ``BackendDispatchMixin._dispatch``
    but adds handling for ``DatasetNotFoundError`` in addition to ``ProblemError``.

    Note
    ----
    This mixin expects the concrete class to also inherit from
    ``BackendDispatchMixin`` which provides ``is_local`` and ``service``.
    """

    if TYPE_CHECKING:
        # These are provided by BackendDispatchMixin at runtime
        service: QueryService
        is_local: bool

    def _dispatch_dataset[R](
        self,
        method_name: str,
        response_type: type[R],
        **kwargs: object,
    ) -> R:
        """
        Dispatch a dataset method with error handling and response conversion.

        Similar to ``BackendDispatchMixin._dispatch`` but handles
        ``DatasetNotFoundError`` in addition to ``ProblemError``.

        Parameters
        ----------
        method_name
            Name of the method on ``self.service`` to call.
        response_type
            Pydantic response model type for conversion.
        **kwargs
            Keyword arguments to pass to the service method.

        Returns
        -------
        R
            Response model instance.

        Raises
        ------
        errors.McpError
            When the underlying service reports a problem detail or
            dataset is not found (local only).
        """
        method = getattr(self.service, method_name)

        if self.is_local:
            try:
                domain_result = method(**kwargs)
            except DatasetNotFoundError as exc:
                raise errors.McpError(exc.detail) from exc
            except ProblemError as exc:
                raise errors.McpError(exc.detail) from exc
            return response_type.from_domain(domain_result)  # type: ignore[attr-defined]

        result = method(**kwargs)
        if isinstance(result, response_type):
            return result
        return response_type.from_domain(result)  # type: ignore[attr-defined]

    def list_datasets(self) -> list[DatasetDescriptor]:
        """
        List datasets exposed by the backend.

        Returns
        -------
        list[DatasetDescriptor]
            Dataset metadata entries.
        """
        descriptors: list[DatasetDescriptor] = []
        for dataset in self.service.list_datasets():
            if isinstance(dataset, DatasetDescriptor):
                descriptors.append(dataset)
                continue
            if is_dataclass(dataset):
                payload = asdict(dataset)
            elif isinstance(dataset, Mapping):
                payload = dict(dataset)
            elif hasattr(dataset, "model_dump"):
                payload = cast("dict[str, object]", dataset.model_dump())
            else:
                payload = cast("dict[str, object]", getattr(dataset, "__dict__", {}))
            descriptors.append(DatasetDescriptor.model_validate(payload))
        return descriptors

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """
        Read a slice of rows from a dataset.

        Returns
        -------
        DatasetRowsResponse
            Dataset slice payload with metadata.
        The dispatch layer may propagate ``errors.McpError`` if the service reports a problem.
        """
        return self._dispatch_dataset(
            "read_dataset_rows",
            DatasetRowsResponse,
            dataset_name=dataset_name,
            limit=limit,
            offset=offset,
        )

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        """
        Return canonical dataset specs.

        Returns
        -------
        list[DatasetSpecDescriptor]
            Dataset spec descriptors sorted by name.
        """
        return self.service.dataset_specs()

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
        """
        Return schema details for a dataset.

        Returns
        -------
        DatasetSchemaResponse
            Composite schema response including sample rows.
        The dispatch layer may propagate ``errors.McpError`` if the service reports a problem.
        """
        return self._dispatch_dataset(
            "dataset_schema",
            DatasetSchemaResponse,
            dataset_name=dataset_name,
            sample_limit=sample_limit,
        )


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
    normalized = {
        "incoming": "in",
        "in": "in",
        "outgoing": "out",
        "out": "out",
        "both": "both",
    }
    if direction in normalized:
        return normalized[direction]
    message = "direction must be one of in, out, both"
    raise errors.invalid_argument(message)


@dataclass
class DuckDBBackend(BackendDispatchMixin, DatasetBackendMixin):
    """DuckDB-backed implementation of QueryBackend.

    This class implements the ``QueryBackend`` protocol via duck typing
    (not direct inheritance to avoid dataclass field ordering issues
    with Python 3.13's protocol annotation handling).

    The backend requires a pre-constructed ``LocalQueryService`` which provides
    query capabilities. Use ``build_backend_resource()`` from
    ``codeintel.serving.bootstrap`` to construct the service.

    This class uses ``BackendDispatchMixin`` to consolidate the repetitive
    error handling and response conversion pattern across all methods.

    Example
    -------
    >>> from codeintel.serving.bootstrap import build_backend_resource
    >>> resource = build_backend_resource(gateway=gateway, repo="my/repo", commit="abc123")
    >>> backend = resource.backend
    """

    service: QueryService
    gateway: StorageGateway
    repo: str | None = None
    commit: str | None = None
    limits: BackendLimits = field(default_factory=BackendLimits)
    observability: ServiceObservability | None = None
    query: DuckDBQueryApi | DuckDBQueryService | None = field(init=False, default=None)
    query_engine: GraphEngine | None = None

    @property
    def is_local(self) -> bool:
        """Return True since this is a local DuckDB backend."""
        return True

    def __post_init__(self) -> None:
        """Initialize internal state from the provided service."""
        if isinstance(self.service, LocalQueryService):
            self.query = self.service.query

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: object | None = None,
    ) -> FunctionSummaryResponse:
        """
        Return a function summary from the DuckDB-backed query service.

        Returns
        -------
        FunctionSummaryResponse
            Summary payload with found flag and metadata.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        _require_identifier(urn=urn, goid_h128=goid_h128, rel_path=rel_path)
        return self._dispatch(
            "get_function_summary",
            FunctionSummaryResponse,
            urn=urn,
            goid_h128=goid_h128,
            rel_path=rel_path,
            qualname=qualname,
            scope=scope,
        )

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: object | None = None,
    ) -> HighRiskFunctionsResponse:
        """
        List high-risk functions for the configured repo/commit.

        Returns
        -------
        HighRiskFunctionsResponse
            High-risk functions plus truncation metadata.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        return self._dispatch(
            "list_high_risk_functions",
            HighRiskFunctionsResponse,
            min_risk=min_risk,
            limit=limit,
            tested_only=tested_only,
            scope=scope,
        )

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: object | None = None,
    ) -> CallGraphNeighborsResponse:
        """
        Return call graph neighbors for a function GOID.

        Returns
        -------
        CallGraphNeighborsResponse
            Incoming and outgoing edges with metadata.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        direction = _validate_direction(direction)
        return self._dispatch(
            "get_callgraph_neighbors",
            CallGraphNeighborsResponse,
            goid_h128=goid_h128,
            direction=direction,
            limit=limit,
            scope=scope,
        )

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> GraphNeighborhoodResponse:
        """
        Return a bounded ego neighborhood for a function.

        Returns
        -------
        GraphNeighborhoodResponse
            Nodes and edges in the neighborhood plus metadata.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        return self._dispatch(
            "get_callgraph_neighborhood",
            GraphNeighborhoodResponse,
            goid_h128=goid_h128,
            radius=radius,
            max_nodes=max_nodes,
        )

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> ImportBoundaryResponse:
        """
        Return import graph edges crossing a subsystem boundary.

        Returns
        -------
        ImportBoundaryResponse
            Boundary edges and truncation metadata.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        return self._dispatch(
            "get_import_boundary",
            ImportBoundaryResponse,
            subsystem_id=subsystem_id,
            max_edges=max_edges,
        )

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: object | None = None,
    ) -> TestsForFunctionResponse:
        """
        Return tests linked to a function.

        Returns
        -------
        TestsForFunctionResponse
            Tests exercising the function plus messages.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        _require_identifier(urn=urn, goid_h128=goid_h128)
        return self._dispatch(
            "get_tests_for_function",
            TestsForFunctionResponse,
            goid_h128=goid_h128,
            urn=urn,
            limit=limit,
            scope=scope,
        )

    def get_file_summary(
        self,
        *,
        rel_path: str,
        scope: object | None = None,
    ) -> FileSummaryResponse:
        """
        Return file summary plus function rows.

        Returns
        -------
        FileSummaryResponse
            File-level summary and nested function entries.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        return self._dispatch(
            "get_file_summary",
            FileSummaryResponse,
            rel_path=rel_path,
            scope=scope,
        )

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        """
        Return a denormalized function profile.

        Returns
        -------
        FunctionProfileResponse
            Profile payload and found flag.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        return self._dispatch(
            "get_function_profile",
            FunctionProfileResponse,
            goid_h128=goid_h128,
        )

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        """
        Return a denormalized file profile.

        Returns
        -------
        FileProfileResponse
            Profile payload and found flag.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        return self._dispatch(
            "get_file_profile",
            FileProfileResponse,
            rel_path=rel_path,
        )

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        """
        Return a module profile for the configured repo/commit.

        Returns
        -------
        ModuleProfileResponse
            Profile payload and found flag.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        return self._dispatch(
            "get_module_profile",
            ModuleProfileResponse,
            module=module,
        )

    def get_function_architecture(self, *, goid_h128: int) -> FunctionArchitectureResponse:
        """
        Return call-graph architecture metrics for a function.

        Returns
        -------
        FunctionArchitectureResponse
            Architecture payload and found flag.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        return self._dispatch(
            "get_function_architecture",
            FunctionArchitectureResponse,
            goid_h128=goid_h128,
        )

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        """
        Return import-graph and symbol-coupling metrics for a module.

        Returns
        -------
        ModuleArchitectureResponse
            Architecture payload and found flag.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        return self._dispatch(
            "get_module_architecture",
            ModuleArchitectureResponse,
            module=module,
        )

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSummaryResponse:
        """
        List inferred subsystems for the current repo/commit.

        Returns
        -------
        SubsystemSummaryResponse
            Subsystem rows and metadata.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        return self._dispatch(
            "list_subsystems",
            SubsystemSummaryResponse,
            limit=limit,
            role=role,
            q=q,
        )

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        """
        Return subsystem memberships for a module.

        Returns
        -------
        ModuleSubsystemResponse
            Membership rows and metadata.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        return self._dispatch(
            "get_module_subsystems",
            ModuleSubsystemResponse,
            module=module,
        )

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        """
        Return IDE-focused hints for a file path.

        Returns
        -------
        FileHintsResponse
            Hints including subsystem context and module metrics.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        return self._dispatch(
            "get_file_hints",
            FileHintsResponse,
            rel_path=rel_path,
        )

    def get_subsystem_modules(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """
        Return subsystem details and module memberships.

        Returns
        -------
        SubsystemModulesResponse
            Subsystem detail payload.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        return self._dispatch(
            "get_subsystem_modules",
            SubsystemModulesResponse,
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSearchResponse:
        """
        Search subsystems with optional role/name filters.

        Returns
        -------
        SubsystemSearchResponse
            Subsystem rows and metadata.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        return self._dispatch(
            "search_subsystems",
            SubsystemSearchResponse,
            limit=limit,
            role=role,
            q=q,
        )

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """
        Summarize a subsystem with optional module truncation.

        Returns
        -------
        SubsystemModulesResponse
            Subsystem detail payload.
        The dispatch layer may propagate ``errors.McpError`` when the service reports a problem.
        """
        return self._dispatch(
            "summarize_subsystem",
            SubsystemModulesResponse,
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )


@dataclass
class HttpBackend(BackendDispatchMixin, DatasetBackendMixin):
    """HTTP-backed QueryBackend that talks to the FastAPI server.

    This class implements the ``QueryBackend`` protocol via duck typing
    (not direct inheritance to avoid dataclass field ordering issues
    with Python 3.13's protocol annotation handling).

    This class uses ``BackendDispatchMixin`` to consolidate the repetitive
    response conversion pattern across all methods.
    """

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
    retry_attempts: int = 3
    retry_backoff: float = 0.1
    circuit_threshold: int = 5
    circuit_cooldown_s: float = 30.0
    consecutive_failures: int = field(init=False, default=0)
    last_failure_ts: float | None = field(init=False, default=None)
    last_retry_attempts: int = field(init=False, default=1)

    @property
    def is_local(self) -> bool:
        """Return False since this is an HTTP backend."""
        return False

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
        now = time.monotonic()
        if (
            self.consecutive_failures >= self.circuit_threshold
            and self.last_failure_ts is not None
            and now - self.last_failure_ts < self.circuit_cooldown_s
        ):
            message = "HTTP circuit open; retry later"
            LOG.warning(message)
            raise errors.backend_failure(message)

        attempt_error: Exception | None = None
        attempts_used = 0
        client = self.client
        for attempt in range(1, self.retry_attempts + 1):
            attempts_used = attempt
            try:
                if isinstance(client, httpx.AsyncClient):
                    response = anyio.run(_get_async, client, path, normalized_params)
                else:
                    response = client.get(path, params=normalized_params)
            except httpx.RequestError as exc:
                attempt_error = exc
            else:
                if response.status_code >= HTTP_ERROR_STATUS:
                    payload = response.json()
                    problem = ProblemDetailModel.model_validate(payload).to_domain()
                    attempt_error = errors.McpError(detail=problem)
                    LOG.warning("HTTP backend error: %s", problem.detail or problem.title)
                    if response.status_code < RETRYABLE_MIN_STATUS:
                        raise attempt_error
                else:
                    self.consecutive_failures = 0
                    self.last_failure_ts = None
                    self.last_retry_attempts = attempts_used
                    return response.json()
            time.sleep(self.retry_backoff * attempt)

        self.consecutive_failures += 1
        self.last_failure_ts = time.monotonic()
        self.last_retry_attempts = attempts_used
        if attempt_error is not None:
            raise attempt_error
        message = "HTTP request failed after retries"
        raise errors.backend_failure(message)

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
        except ProblemError as exc:
            message = "Failed to verify remote API health"
            raise errors.backend_failure(message) from exc
        except OSError as exc:
            message = "Failed to reach remote API health endpoint"
            raise errors.backend_failure(message) from exc

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: object | None = None,
    ) -> FunctionSummaryResponse:
        """
        Return a function summary from the remote API.

        Returns
        -------
        FunctionSummaryResponse
            Summary payload with found flag.
        """
        return self._dispatch(
            "get_function_summary",
            FunctionSummaryResponse,
            urn=urn,
            goid_h128=goid_h128,
            rel_path=rel_path,
            qualname=qualname,
            scope=scope,
        )

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: object | None = None,
    ) -> HighRiskFunctionsResponse:
        """
        List high-risk functions from the remote API.

        Returns
        -------
        HighRiskFunctionsResponse
            Functions ordered by risk with truncation metadata.
        """
        return self._dispatch(
            "list_high_risk_functions",
            HighRiskFunctionsResponse,
            min_risk=min_risk,
            limit=limit,
            tested_only=tested_only,
            scope=scope,
        )

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: object | None = None,
    ) -> CallGraphNeighborsResponse:
        """
        Return call graph neighbors for a function GOID.

        Returns
        -------
        CallGraphNeighborsResponse
            Neighbor edges and metadata.
        """
        return self._dispatch(
            "get_callgraph_neighbors",
            CallGraphNeighborsResponse,
            goid_h128=goid_h128,
            direction=direction,
            limit=limit,
            scope=scope,
        )

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> GraphNeighborhoodResponse:
        """
        Return a bounded ego neighborhood in the call graph.

        Returns
        -------
        GraphNeighborhoodResponse
            Nodes and edges for the neighborhood plus metadata.
        """
        return self._dispatch(
            "get_callgraph_neighborhood",
            GraphNeighborhoodResponse,
            goid_h128=goid_h128,
            radius=radius,
            max_nodes=max_nodes,
        )

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> ImportBoundaryResponse:
        """
        Return import graph edges crossing a subsystem boundary.

        Returns
        -------
        ImportBoundaryResponse
            Boundary edges and truncation metadata.
        """
        return self._dispatch(
            "get_import_boundary",
            ImportBoundaryResponse,
            subsystem_id=subsystem_id,
            max_edges=max_edges,
        )

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: object | None = None,
    ) -> TestsForFunctionResponse:
        """
        Return tests linked to a function.

        Returns
        -------
        TestsForFunctionResponse
            Tests hitting the function plus messages.
        """
        return self._dispatch(
            "get_tests_for_function",
            TestsForFunctionResponse,
            goid_h128=goid_h128,
            urn=urn,
            limit=limit,
            scope=scope,
        )

    def get_file_summary(
        self,
        *,
        rel_path: str,
        scope: object | None = None,
    ) -> FileSummaryResponse:
        """
        Return a file summary from the remote API.

        Returns
        -------
        FileSummaryResponse
            File-level summary payload with functions.
        """
        return self._dispatch(
            "get_file_summary",
            FileSummaryResponse,
            rel_path=rel_path,
            scope=scope,
        )

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        """
        Return a function profile from the remote API.

        Returns
        -------
        FunctionProfileResponse
            Profile payload including found flag.
        """
        return self._dispatch(
            "get_function_profile",
            FunctionProfileResponse,
            goid_h128=goid_h128,
        )

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        """
        Return a file profile from the remote API.

        Returns
        -------
        FileProfileResponse
            Profile payload including found flag.
        """
        return self._dispatch(
            "get_file_profile",
            FileProfileResponse,
            rel_path=rel_path,
        )

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        """
        Return a module profile from the remote API.

        Returns
        -------
        ModuleProfileResponse
            Profile payload including found flag.
        """
        return self._dispatch(
            "get_module_profile",
            ModuleProfileResponse,
            module=module,
        )

    def get_function_architecture(self, *, goid_h128: int) -> FunctionArchitectureResponse:
        """
        Return function architecture metrics from the remote API.

        Returns
        -------
        FunctionArchitectureResponse
            Architecture payload including found flag.
        """
        return self._dispatch(
            "get_function_architecture",
            FunctionArchitectureResponse,
            goid_h128=goid_h128,
        )

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        """
        Return module architecture metrics from the remote API.

        Returns
        -------
        ModuleArchitectureResponse
            Architecture payload including found flag.
        """
        return self._dispatch(
            "get_module_architecture",
            ModuleArchitectureResponse,
            module=module,
        )

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
        return self._dispatch(
            "list_subsystems",
            SubsystemSummaryResponse,
            limit=limit,
            role=role,
            q=q,
        )

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        """
        Return subsystem memberships for a module from the remote API.

        Returns
        -------
        ModuleSubsystemResponse
            Membership rows and metadata.
        """
        return self._dispatch(
            "get_module_subsystems",
            ModuleSubsystemResponse,
            module=module,
        )

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        """
        Return IDE-focused hints for a file from the remote API.

        Returns
        -------
        FileHintsResponse
            Hint rows and metadata for the path.
        """
        return self._dispatch(
            "get_file_hints",
            FileHintsResponse,
            rel_path=rel_path,
        )

    def get_subsystem_modules(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """
        Return subsystem details and module memberships from the remote API.

        Returns
        -------
        SubsystemModulesResponse
            Subsystem detail payload.
        """
        return self._dispatch(
            "get_subsystem_modules",
            SubsystemModulesResponse,
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )

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
        return self._dispatch(
            "search_subsystems",
            SubsystemSearchResponse,
            limit=limit,
            role=role,
            q=q,
        )

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
        return self._dispatch(
            "summarize_subsystem",
            SubsystemModulesResponse,
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )
