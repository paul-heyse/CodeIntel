"""Transport-agnostic query application services."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

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
    GraphNeighborhoodResponse,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    ModuleArchitectureResponse,
    ModuleProfileResponse,
    ModuleSubsystemResponse,
    ResponseMeta,
    SubsystemModulesResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    TestsForFunctionResponse,
)
from codeintel.mcp.query_service import (
    BackendLimits,
    DuckDBQueryService,
    clamp_limit_value,
    clamp_offset_value,
)
from codeintel.server.datasets import describe_dataset

LOG = logging.getLogger("codeintel.services.query")


@dataclass
class ServiceCallMetrics:
    """Structured metrics describing a service invocation."""

    name: str
    transport: str
    duration_ms: float
    rows: int | None = None
    dataset: str | None = None
    messages: int | None = None
    error: str | None = None


@dataclass
class ServiceObservability:
    """Configuration for service-level observability."""

    enabled: bool = False
    logger: logging.Logger = field(default_factory=lambda: LOG)

    def record(self, metrics: ServiceCallMetrics) -> None:
        """
        Emit a structured log line for a service call.

        Parameters
        ----------
        metrics:
            Call metrics describing the invocation outcome.
        """
        if not self.enabled or not self.logger.isEnabledFor(logging.INFO):
            return
        payload: dict[str, object] = {
            "name": metrics.name,
            "transport": metrics.transport,
            "duration_ms": round(metrics.duration_ms, 2),
        }
        if metrics.rows is not None:
            payload["rows"] = metrics.rows
        if metrics.dataset is not None:
            payload["dataset"] = metrics.dataset
        if metrics.messages is not None:
            payload["messages"] = metrics.messages
        if metrics.error is not None:
            payload["error"] = metrics.error
        self.logger.info("service_call %s", payload)


def _extract_row_count(result: object) -> int | None:
    """
    Attempt to derive a row count from common response shapes.

    Returns
    -------
    int | None
        Row count when inferrable; otherwise ``None``.
    """
    count: int | None = None
    if isinstance(result, DatasetRowsResponse):
        count = len(result.rows)
    elif isinstance(result, HighRiskFunctionsResponse):
        count = len(result.functions)
    elif isinstance(result, CallGraphNeighborsResponse):
        count = len(result.outgoing) + len(result.incoming)
    elif isinstance(result, TestsForFunctionResponse):
        count = len(result.tests)
    elif isinstance(result, SubsystemSummaryResponse):
        count = len(result.subsystems)
    elif isinstance(result, ModuleSubsystemResponse):
        count = len(result.memberships)
    elif isinstance(result, FileHintsResponse):
        count = len(result.hints)
    elif isinstance(result, SubsystemModulesResponse):
        count = len(result.modules)
    elif isinstance(result, SubsystemSearchResponse):
        count = len(result.subsystems)
    return count


def _extract_message_count(result: object) -> int | None:
    """
    Return the number of response messages when available.

    Returns
    -------
    int | None
        Message count if present; otherwise ``None``.
    """
    meta = getattr(result, "meta", None)
    if meta is None or meta.messages is None:
        return None
    return len(meta.messages)


def _observe_call[T](
    observability: ServiceObservability | None,
    *,
    transport: str,
    name: str,
    dataset: str | None,
    func: Callable[[], T],
) -> T:
    """
    Execute a callable while capturing observability signals.

    Returns
    -------
    T
        Result returned by the wrapped callable.
    """
    start = time.perf_counter()
    try:
        result = func()
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        if observability is not None:
            observability.record(
                ServiceCallMetrics(
                    name=name,
                    transport=transport,
                    duration_ms=duration_ms,
                    dataset=dataset,
                    error=exc.__class__.__name__,
                )
            )
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    if observability is not None:
        observability.record(
            ServiceCallMetrics(
                name=name,
                transport=transport,
                duration_ms=duration_ms,
                rows=_extract_row_count(result),
                dataset=dataset,
                messages=_extract_message_count(result),
            )
        )
    return result


class FunctionQueryApi(Protocol):
    """Function-centric query surface."""

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> FunctionSummaryResponse:
        """Return a function summary for an identifier."""
        ...

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
    ) -> HighRiskFunctionsResponse:
        """List high-risk functions."""
        ...

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
    ) -> CallGraphNeighborsResponse:
        """Return call graph neighbors for a function."""
        ...

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
    ) -> TestsForFunctionResponse:
        """List tests that exercise a function."""
        ...

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> GraphNeighborhoodResponse:
        """Return an ego neighborhood in the call graph."""
        ...

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> ImportBoundaryResponse:
        """Return import edges crossing a subsystem boundary."""
        ...

    def get_file_summary(self, *, rel_path: str) -> FileSummaryResponse:
        """Return a file summary."""
        ...


class ProfileQueryApi(Protocol):
    """Profile and architecture surfaces."""

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        """Return a function profile."""
        ...

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        """Return a file profile."""
        ...

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        """Return a module profile."""
        ...

    def get_function_architecture(self, *, goid_h128: int) -> FunctionArchitectureResponse:
        """Return architecture metrics for a function."""
        ...

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        """Return architecture metrics for a module."""
        ...


class SubsystemQueryApi(Protocol):
    """Subsystem and hints surfaces."""

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSummaryResponse:
        """List subsystems with optional filters."""
        ...

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        """Return subsystem memberships for a module."""
        ...

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        """Return IDE hints for a file."""
        ...

    def get_subsystem_modules(self, *, subsystem_id: str) -> SubsystemModulesResponse:
        """Return a subsystem with member modules."""
        ...

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSearchResponse:
        """Search subsystems."""
        ...

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """Summarize a subsystem with optional module limit."""
        ...


class DatasetQueryApi(Protocol):
    """Dataset listing and retrieval surface."""

    def list_datasets(self) -> list[DatasetDescriptor]:
        """List available datasets."""
        ...

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """Read rows from a dataset."""
        ...


class QueryService(
    FunctionQueryApi,
    ProfileQueryApi,
    SubsystemQueryApi,
    DatasetQueryApi,
    Protocol,
):
    """Composite query service consumed by HTTP and MCP transports."""


class _FunctionQueryDelegates:
    query: DuckDBQueryService
    _call: Callable[..., Any]

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> FunctionSummaryResponse:
        return self._call(
            "get_function_summary",
            lambda: self.query.get_function_summary(
                urn=urn, goid_h128=goid_h128, rel_path=rel_path, qualname=qualname
            ),
        )

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
    ) -> HighRiskFunctionsResponse:
        return self._call(
            "list_high_risk_functions",
            lambda: self.query.list_high_risk_functions(
                min_risk=min_risk, limit=limit, tested_only=tested_only
            ),
        )

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
    ) -> CallGraphNeighborsResponse:
        return self._call(
            "get_callgraph_neighbors",
            lambda: self.query.get_callgraph_neighbors(
                goid_h128=goid_h128, direction=direction, limit=limit
            ),
        )

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
    ) -> TestsForFunctionResponse:
        return self._call(
            "get_tests_for_function",
            lambda: self.query.get_tests_for_function(goid_h128=goid_h128, urn=urn, limit=limit),
        )

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> GraphNeighborhoodResponse:
        return self._call(
            "get_callgraph_neighborhood",
            lambda: self.query.get_callgraph_neighborhood(
                goid_h128=goid_h128, radius=radius, max_nodes=max_nodes
            ),
            dataset="call_graph_nodes",
        )

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> ImportBoundaryResponse:
        return self._call(
            "get_import_boundary",
            lambda: self.query.get_import_boundary(subsystem_id=subsystem_id, max_edges=max_edges),
            dataset="import_graph_edges",
        )

    def get_file_summary(self, *, rel_path: str) -> FileSummaryResponse:
        return self._call(
            "get_file_summary", lambda: self.query.get_file_summary(rel_path=rel_path)
        )


class _ProfileQueryDelegates:
    query: DuckDBQueryService
    _call: Callable[..., Any]

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        return self._call(
            "get_function_profile",
            lambda: self.query.get_function_profile(goid_h128=goid_h128),
        )

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        return self._call(
            "get_file_profile", lambda: self.query.get_file_profile(rel_path=rel_path)
        )

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        return self._call(
            "get_module_profile", lambda: self.query.get_module_profile(module=module)
        )

    def get_function_architecture(self, *, goid_h128: int) -> FunctionArchitectureResponse:
        return self._call(
            "get_function_architecture",
            lambda: self.query.get_function_architecture(goid_h128=goid_h128),
        )

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        return self._call(
            "get_module_architecture",
            lambda: self.query.get_module_architecture(module=module),
        )


class _SubsystemQueryDelegates:
    query: DuckDBQueryService
    _call: Callable[..., Any]

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSummaryResponse:
        return self._call(
            "list_subsystems", lambda: self.query.list_subsystems(limit=limit, role=role, q=q)
        )

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        return self._call(
            "get_module_subsystems", lambda: self.query.get_module_subsystems(module=module)
        )

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        return self._call("get_file_hints", lambda: self.query.get_file_hints(rel_path=rel_path))

    def get_subsystem_modules(self, *, subsystem_id: str) -> SubsystemModulesResponse:
        return self._call(
            "get_subsystem_modules",
            lambda: self.query.get_subsystem_modules(subsystem_id=subsystem_id),
        )

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSearchResponse:
        return self._call(
            "search_subsystems", lambda: self.query.search_subsystems(limit=limit, role=role, q=q)
        )

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        return self._call(
            "summarize_subsystem",
            lambda: self.query.summarize_subsystem(
                subsystem_id=subsystem_id, module_limit=module_limit
            ),
        )


@dataclass
class LocalQueryService(_FunctionQueryDelegates, _ProfileQueryDelegates, _SubsystemQueryDelegates):
    """Application service backed by a local DuckDB query layer."""

    query: DuckDBQueryService
    dataset_tables: dict[str, str] | None = None
    describe_dataset_fn: Callable[[str, str], str] = describe_dataset
    observability: ServiceObservability | None = None
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Derive dataset registry from the query gateway when not provided."""
        if self.dataset_tables is None:
            gateway = getattr(self.query, "gateway", None)
            self.dataset_tables = dict(gateway.datasets.mapping) if gateway is not None else {}

    def _call[T](
        self,
        name: str,
        func: Callable[[], T],
        *,
        dataset: str | None = None,
    ) -> T:
        """
        Invoke a query with observability tracking.

        Returns
        -------
        T
            Result returned by the wrapped callable.
        """
        self.calls.append(name)
        return _observe_call(
            self.observability,
            transport="local",
            name=name,
            dataset=dataset,
            func=func,
        )

    def list_datasets(self) -> list[DatasetDescriptor]:
        """
        List datasets available through the dataset registry.

        Returns
        -------
        list[DatasetDescriptor]
            Dataset descriptors with names, tables, and descriptions.
        """

        def _list() -> list[DatasetDescriptor]:
            mapping: dict[str, str] = self.dataset_tables or {}
            if not mapping:
                query_gateway = getattr(self.query, "gateway", None)
                mapping = query_gateway.datasets.mapping if query_gateway is not None else {}
            return [
                DatasetDescriptor(
                    name=name,
                    table=table,
                    description=self.describe_dataset_fn(name, table),
                )
                for name, table in sorted(mapping.items())
            ]

        return self._call("list_datasets", _list)

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """
        Read dataset rows with clamping and messaging.

        Returns
        -------
        DatasetRowsResponse
            Dataset slice and metadata for truncation/messaging.
        """
        applied_limit = self.query.limits.default_limit if limit is None else limit
        return self._call(
            "read_dataset_rows",
            lambda: self.query.read_dataset_rows(
                dataset_name=dataset_name,
                limit=applied_limit,
                offset=offset,
            ),
            dataset=dataset_name,
        )


@dataclass
class HttpQueryService:
    """Application service that forwards queries to a remote HTTP API."""

    request_json: Callable[[str, dict[str, object]], object]
    limits: BackendLimits
    observability: ServiceObservability | None = None

    def _call[T](
        self,
        name: str,
        func: Callable[[], T],
        *,
        dataset: str | None = None,
    ) -> T:
        """
        Invoke a remote query with observability tracking.

        Returns
        -------
        T
            Result returned by the wrapped callable.
        """
        return _observe_call(
            self.observability,
            transport="http",
            name=name,
            dataset=dataset,
            func=func,
        )

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
    ) -> HighRiskFunctionsResponse:
        """
        List high-risk functions via the remote API with clamped limits.

        Returns
        -------
        HighRiskFunctionsResponse
            Functions ordered by risk with metadata.
        """
        applied_limit = self.limits.default_limit if limit is None else limit
        clamp = clamp_limit_value(
            applied_limit,
            default=applied_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        if clamp.has_error:
            return HighRiskFunctionsResponse(functions=[], truncated=False, meta=ResponseMeta())
        return self._call(
            "list_high_risk_functions",
            lambda: HighRiskFunctionsResponse.model_validate(
                self.request_json(
                    "/functions/high-risk",
                    {
                        "min_risk": min_risk,
                        "limit": clamp.applied,
                        "tested_only": tested_only,
                    },
                )
            ),
        )

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
            Summary payload with found flag and metadata.
        """
        return self._call(
            "get_function_summary",
            lambda: FunctionSummaryResponse.model_validate(
                self.request_json(
                    "/function/summary",
                    {
                        "urn": urn,
                        "goid_h128": goid_h128,
                        "rel_path": rel_path,
                        "qualname": qualname,
                    },
                )
            ),
        )

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
    ) -> CallGraphNeighborsResponse:
        """
        Return call graph neighbors via the remote API.

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
        return self._call(
            "get_callgraph_neighbors",
            lambda: CallGraphNeighborsResponse.model_validate(
                self.request_json(
                    "/function/callgraph",
                    {"goid_h128": goid_h128, "direction": direction, "limit": clamp.applied},
                )
            ),
        )

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> GraphNeighborhoodResponse:
        """
        Fetch a call graph ego neighborhood via HTTP.

        Parameters
        ----------
        goid_h128 : int
            Center node for the ego neighborhood.
        radius : int, optional
            Hop distance to traverse when collecting neighbors.
        max_nodes : int, optional
            Optional cap on returned nodes; defaults to backend limits.

        Returns
        -------
        GraphNeighborhoodResponse
            Ego nodes, edges, and metadata reflecting applied limits.
        """
        applied_limit = self.limits.default_limit if max_nodes is None else max_nodes
        clamp = clamp_limit_value(
            applied_limit,
            default=applied_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        if clamp.has_error:
            return GraphNeighborhoodResponse(nodes=[], edges=[], meta=ResponseMeta())
        return self._call(
            "get_callgraph_neighborhood",
            lambda: GraphNeighborhoodResponse.model_validate(
                self.request_json(
                    "/graph/call/neighborhood",
                    {"goid_h128": goid_h128, "radius": radius, "max_nodes": clamp.applied},
                )
            ),
        )

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> ImportBoundaryResponse:
        """
        Fetch import edges that cross a subsystem boundary via HTTP.

        Parameters
        ----------
        subsystem_id : str
            Subsystem identifier to inspect.
        max_edges : int, optional
            Optional cap on returned edges; defaults to backend limits.

        Returns
        -------
        ImportBoundaryResponse
            Boundary nodes and edges plus truncation metadata.
        """
        applied_limit = self.limits.default_limit if max_edges is None else max_edges
        clamp = clamp_limit_value(
            applied_limit,
            default=applied_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        if clamp.has_error:
            return ImportBoundaryResponse(nodes=[], edges=[], meta=ResponseMeta())
        return self._call(
            "get_import_boundary",
            lambda: ImportBoundaryResponse.model_validate(
                self.request_json(
                    "/graph/import/boundary",
                    {"subsystem_id": subsystem_id, "max_edges": clamp.applied},
                )
            ),
        )

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
    ) -> TestsForFunctionResponse:
        """
        List tests exercising a function via the remote API.

        Returns
        -------
        TestsForFunctionResponse
            Tests plus metadata.
        """
        applied_limit = self.limits.default_limit if limit is None else limit
        clamp = clamp_limit_value(
            applied_limit,
            default=applied_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        if clamp.has_error:
            return TestsForFunctionResponse(tests=[], meta=ResponseMeta())
        return self._call(
            "get_tests_for_function",
            lambda: TestsForFunctionResponse.model_validate(
                self.request_json(
                    "/function/tests",
                    {"goid_h128": goid_h128, "urn": urn, "limit": clamp.applied},
                )
            ),
        )

    def get_file_summary(self, *, rel_path: str) -> FileSummaryResponse:
        """
        Return a file summary from the remote API.

        Returns
        -------
        FileSummaryResponse
            Summary plus function rows.
        """
        return self._call(
            "get_file_summary",
            lambda: FileSummaryResponse.model_validate(
                self.request_json("/file/summary", {"rel_path": rel_path})
            ),
        )

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        """
        Return a function profile from the remote API.

        Returns
        -------
        FunctionProfileResponse
            Profile payload with found flag.
        """
        return self._call(
            "get_function_profile",
            lambda: FunctionProfileResponse.model_validate(
                self.request_json("/profiles/function", {"goid_h128": goid_h128})
            ),
        )

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        """
        Return a file profile from the remote API.

        Returns
        -------
        FileProfileResponse
            Profile payload with found flag.
        """
        return self._call(
            "get_file_profile",
            lambda: FileProfileResponse.model_validate(
                self.request_json("/profiles/file", {"rel_path": rel_path})
            ),
        )

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        """
        Return a module profile from the remote API.

        Returns
        -------
        ModuleProfileResponse
            Profile payload with found flag.
        """
        return self._call(
            "get_module_profile",
            lambda: ModuleProfileResponse.model_validate(
                self.request_json("/profiles/module", {"module": module})
            ),
        )

    def get_function_architecture(self, *, goid_h128: int) -> FunctionArchitectureResponse:
        """
        Return function architecture metrics from the remote API.

        Returns
        -------
        FunctionArchitectureResponse
            Architecture payload with found flag.
        """
        return self._call(
            "get_function_architecture",
            lambda: FunctionArchitectureResponse.model_validate(
                self.request_json("/architecture/function", {"goid_h128": goid_h128})
            ),
        )

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        """
        Return module architecture metrics from the remote API.

        Returns
        -------
        ModuleArchitectureResponse
            Architecture payload with found flag.
        """
        return self._call(
            "get_module_architecture",
            lambda: ModuleArchitectureResponse.model_validate(
                self.request_json("/architecture/module", {"module": module})
            ),
        )

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSummaryResponse:
        """
        List subsystems from the remote API with clamped limits.

        Returns
        -------
        SubsystemSummaryResponse
            Subsystem rows and metadata.
        """
        applied_limit = self.limits.default_limit if limit is None else limit
        clamp = clamp_limit_value(
            applied_limit,
            default=applied_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        if clamp.has_error:
            return SubsystemSummaryResponse(subsystems=[], meta=ResponseMeta())
        return self._call(
            "list_subsystems",
            lambda: SubsystemSummaryResponse.model_validate(
                self.request_json(
                    "/architecture/subsystems",
                    {"limit": clamp.applied, "role": role, "q": q},
                )
            ),
        )

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        """
        Return subsystem memberships for a module via remote API.

        Returns
        -------
        ModuleSubsystemResponse
            Membership rows and metadata.
        """
        return self._call(
            "get_module_subsystems",
            lambda: ModuleSubsystemResponse.model_validate(
                self.request_json("/architecture/module-subsystems", {"module": module})
            ),
        )

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        """
        Return IDE hints for a file via remote API.

        Returns
        -------
        FileHintsResponse
            Hint rows and metadata.
        """
        return self._call(
            "get_file_hints",
            lambda: FileHintsResponse.model_validate(
                self.request_json("/ide/hints", {"rel_path": rel_path})
            ),
        )

    def get_subsystem_modules(self, *, subsystem_id: str) -> SubsystemModulesResponse:
        """
        Return subsystem detail and modules via remote API.

        Returns
        -------
        SubsystemModulesResponse
            Subsystem detail payload.
        """
        return self._call(
            "get_subsystem_modules",
            lambda: SubsystemModulesResponse.model_validate(
                self.request_json("/architecture/subsystem", {"subsystem_id": subsystem_id})
            ),
        )

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSearchResponse:
        """
        Search subsystems via remote API.

        Returns
        -------
        SubsystemSearchResponse
            Subsystem rows and metadata.
        """
        applied_limit = self.limits.default_limit if limit is None else limit
        clamp = clamp_limit_value(
            applied_limit,
            default=applied_limit,
            max_limit=self.limits.max_rows_per_call,
        )
        if clamp.has_error:
            return SubsystemSearchResponse(subsystems=[], meta=ResponseMeta())
        return self._call(
            "search_subsystems",
            lambda: SubsystemSearchResponse.model_validate(
                self.request_json(
                    "/architecture/subsystems",
                    {"limit": clamp.applied, "role": role, "q": q},
                )
            ),
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
        detail = self.get_subsystem_modules(subsystem_id=subsystem_id)
        if module_limit is None or not detail.modules:
            return detail
        return SubsystemModulesResponse(
            found=detail.found,
            subsystem=detail.subsystem,
            modules=detail.modules[:module_limit],
            meta=detail.meta,
        )

    def list_datasets(self) -> list[DatasetDescriptor]:
        """
        List datasets available from the remote API.

        Returns
        -------
        list[DatasetDescriptor]
            Dataset descriptors provided by the API.
        """
        data = cast("list[dict[str, object]]", self.request_json("/datasets", {}))
        return [DatasetDescriptor.model_validate(item) for item in data]

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = 100,
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
        offset_clamp = clamp_offset_value(offset)
        meta = ResponseMeta(
            requested_limit=limit,
            applied_limit=clamp.applied,
            requested_offset=offset,
            applied_offset=offset_clamp.applied,
            messages=[*clamp.messages, *offset_clamp.messages],
        )
        if clamp.has_error or offset_clamp.has_error:
            return DatasetRowsResponse(
                dataset=dataset_name,
                limit=clamp.applied,
                offset=offset_clamp.applied,
                rows=[],
                meta=meta,
            )
        data = self.request_json(
            f"/datasets/{dataset_name}",
            {"limit": clamp.applied, "offset": offset_clamp.applied},
        )
        response = DatasetRowsResponse.model_validate(data)
        existing_meta = response.meta if response.meta is not None else ResponseMeta()
        merged_meta = ResponseMeta(
            requested_limit=meta.requested_limit,
            applied_limit=meta.applied_limit,
            requested_offset=meta.requested_offset,
            applied_offset=meta.applied_offset,
            truncated=existing_meta.truncated,
            messages=[*meta.messages, *existing_meta.messages],
        )
        return response.model_copy(update={"meta": merged_meta})
