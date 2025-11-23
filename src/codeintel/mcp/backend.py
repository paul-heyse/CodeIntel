"""Backend implementations for MCP tools over DuckDB or HTTP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, cast

import anyio
import duckdb
import httpx

from codeintel.mcp import errors
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
    Message,
    ModuleProfileResponse,
    ProblemDetail,
    ResponseMeta,
    TestsForFunctionResponse,
    ViewRow,
)
from codeintel.mcp.query_service import (
    BackendLimits,
    DuckDBQueryService,
    clamp_limit_value,
    clamp_offset_value,
)
from codeintel.server.datasets import build_dataset_registry, describe_dataset

VALID_DIRECTIONS = frozenset({"in", "out", "both"})
MAX_ROWS_LIMIT = BackendLimits().max_rows_per_call
HTTP_ERROR_STATUS = 400


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
        limit: int = 50,
        tested_only: bool = False,
    ) -> HighRiskFunctionsResponse:
        """List high-risk functions from analytics.goid_risk_factors."""
        ...

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int = 50,
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


@dataclass
class DuckDBBackend(QueryBackend):
    """
    DuckDB-backed implementation of QueryBackend.

    Assumes a single repo/commit per DuckDB file, but repo/commit filters
    are still applied for future multi-repo support.
    """

    con: duckdb.DuckDBPyConnection
    repo: str
    commit: str
    limits: BackendLimits = field(default_factory=BackendLimits)
    dataset_tables: dict[str, str] = field(default_factory=build_dataset_registry)
    query: DuckDBQueryService = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the query service and dataset registry."""
        self.query = DuckDBQueryService(
            con=self.con,
            repo=self.repo,
            commit=self.commit,
            limits=self.limits,
            dataset_tables=self.dataset_tables,
        )

    def __setattr__(self, name: str, value: object) -> None:
        """Propagate dataset registry changes to the underlying query service."""
        super().__setattr__(name, value)
        if name == "dataset_tables" and hasattr(self, "query"):
            self.query.dataset_tables = value if isinstance(value, dict) else {}

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
        return self.query.get_function_summary(
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
        return self.query.list_high_risk_functions(
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
        return self.query.get_callgraph_neighbors(
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
        return self.query.get_tests_for_function(
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
        return self.query.get_file_summary(rel_path=rel_path)

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        """
        Return a denormalized function profile.

        Returns
        -------
        FunctionProfileResponse
            Profile payload and found flag.
        """
        return self.query.get_function_profile(goid_h128=goid_h128)

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        """
        Return a denormalized file profile.

        Returns
        -------
        FileProfileResponse
            Profile payload and found flag.
        """
        return self.query.get_file_profile(rel_path=rel_path)

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        """
        Return a module profile for the configured repo/commit.

        Returns
        -------
        ModuleProfileResponse
            Profile payload and found flag.
        """
        return self.query.get_module_profile(module=module)

    def list_datasets(self) -> list[DatasetDescriptor]:
        """
        List datasets exposed by the backend.

        Returns
        -------
        list[DatasetDescriptor]
            Dataset metadata entries.
        """
        return [
            DatasetDescriptor(
                name=name,
                table=table,
                description=describe_dataset(name, table),
            )
            for name, table in sorted(self.dataset_tables.items())
        ]

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

        Raises
        ------
        errors.invalid_argument
            When the dataset name is unknown.
        """
        table = self.dataset_tables.get(dataset_name)
        if not table:
            message = f"Unknown dataset: {dataset_name}"
            raise errors.invalid_argument(message)

        limit_clamp = clamp_limit_value(
            limit,
            default=limit,
            max_limit=self.limits.max_rows_per_call,
        )
        offset_clamp = clamp_offset_value(offset)
        meta = ResponseMeta(
            requested_limit=limit,
            applied_limit=limit_clamp.applied,
            requested_offset=offset,
            applied_offset=offset_clamp.applied,
            messages=[*limit_clamp.messages, *offset_clamp.messages],
        )

        if limit_clamp.has_error or offset_clamp.has_error:
            return DatasetRowsResponse(
                dataset=dataset_name,
                limit=limit_clamp.applied,
                offset=offset_clamp.applied,
                rows=[],
                meta=meta,
            )

        relation = self.con.table(table).limit(limit_clamp.applied, offset_clamp.applied)
        rows = relation.fetchall()
        cols = [desc[0] for desc in relation.description]
        mapped = [{col: row[idx] for idx, col in enumerate(cols)} for row in rows]
        meta.truncated = limit_clamp.applied > 0 and len(mapped) == limit_clamp.applied
        if not mapped:
            meta.messages.append(
                Message(
                    code="dataset_empty",
                    severity="info",
                    detail="Dataset returned no rows for the requested slice.",
                    context={"dataset": dataset_name, "offset": offset_clamp.applied},
                )
            )
        return DatasetRowsResponse(
            dataset=dataset_name,
            limit=limit_clamp.applied,
            offset=offset_clamp.applied,
            rows=[ViewRow.model_validate(r) for r in mapped],
            meta=meta,
        )


@dataclass
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

    def __post_init__(self) -> None:
        """Initialize the HTTP client and verify server health."""
        if self.client is None:
            self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
            self._owns_client = True
            self._async_client = False
            self._verify_health()
            return

        self._owns_client = False
        self._async_client = isinstance(self.client, httpx.AsyncClient)
        self._verify_health()

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
        applied_limit = self.limits.default_limit if limit is None else limit
        clamp = clamp_limit_value(
            applied_limit,
            default=applied_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        if clamp.has_error:
            return HighRiskFunctionsResponse(functions=[], truncated=False, meta=ResponseMeta())
        data = self._request_json(
            "/functions/high-risk",
            {"min_risk": min_risk, "limit": clamp.applied, "tested_only": tested_only},
        )
        return HighRiskFunctionsResponse.model_validate(data)

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
        applied_limit = self.limits.default_limit if limit is None else limit
        clamp = clamp_limit_value(
            applied_limit,
            default=applied_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        if clamp.has_error:
            return CallGraphNeighborsResponse(outgoing=[], incoming=[], meta=ResponseMeta())
        data = self._request_json(
            "/function/callgraph",
            {"goid_h128": goid_h128, "direction": direction, "limit": clamp.applied},
        )
        return CallGraphNeighborsResponse.model_validate(data)

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
        applied_limit = self.limits.default_limit if limit is None else limit
        clamp = clamp_limit_value(
            applied_limit,
            default=applied_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        if clamp.has_error:
            return TestsForFunctionResponse(tests=[], meta=ResponseMeta())
        data = self._request_json(
            "/function/tests",
            {"goid_h128": goid_h128, "urn": urn, "limit": clamp.applied},
        )
        return TestsForFunctionResponse.model_validate(data)

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
        data = self._request_json("/file/summary", {"rel_path": rel_path})
        return FileSummaryResponse.model_validate(data)

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        """
        Return a function profile from the remote API.

        Returns
        -------
        FunctionProfileResponse
            Profile payload including found flag.
        """
        data = self._request_json(
            "/profiles/function",
            {"goid_h128": goid_h128},
        )
        return FunctionProfileResponse.model_validate(data)

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        """
        Return a file profile from the remote API.

        Returns
        -------
        FileProfileResponse
            Profile payload including found flag.
        """
        data = self._request_json(
            "/profiles/file",
            {"rel_path": rel_path},
        )
        return FileProfileResponse.model_validate(data)

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        """
        Return a module profile from the remote API.

        Returns
        -------
        ModuleProfileResponse
            Profile payload including found flag.
        """
        data = self._request_json(
            "/profiles/module",
            {"module": module},
        )
        return ModuleProfileResponse.model_validate(data)

    def list_datasets(self) -> list[DatasetDescriptor]:
        """
        List datasets available from the remote API.

        Returns
        -------
        list[DatasetDescriptor]
            Dataset descriptors provided by the API.
        """
        data = self._request_json("/datasets", {})
        datasets = cast("list[dict[str, object]]", data)
        return [DatasetDescriptor.model_validate(item) for item in datasets]

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
        applied_limit = self.limits.default_limit if limit is None else limit
        clamp = clamp_limit_value(
            applied_limit,
            default=applied_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        if clamp.has_error:
            return DatasetRowsResponse(
                dataset=dataset_name,
                limit=0,
                offset=offset,
                rows=[],
                meta=ResponseMeta(),
            )
        data = self._request_json(
            f"/datasets/{dataset_name}",
            {"limit": clamp.applied, "offset": offset},
        )
        response = cast("dict[str, object]", data)
        return DatasetRowsResponse.model_validate(response)


def create_backend(
    cfg: McpServerConfig,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    dataset_tables: dict[str, str] | None = None,
) -> QueryBackend:
    """
    Create a QueryBackend using local DuckDB or remote HTTP.

    Parameters
    ----------
    cfg:
        Server configuration loaded from env or caller.
    con:
        Optional DuckDB connection to reuse in local_db mode.
    dataset_tables:
        Optional dataset registry override.

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
    registry = dataset_tables if dataset_tables is not None else build_dataset_registry()
    if cfg.mode == "local_db":
        if cfg.db_path is None and con is None:
            message = "db_path is required for local_db backend"
            raise ValueError(message)
        connection = con or duckdb.connect(str(cfg.db_path), read_only=True)
        return DuckDBBackend(
            con=connection,
            repo=cfg.repo,
            commit=cfg.commit,
            limits=limits,
            dataset_tables=registry,
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
        )
    message = f"Unsupported MCP mode: {cfg.mode}"
    raise ValueError(message)
