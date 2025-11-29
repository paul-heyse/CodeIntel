"""Transport-agnostic query application services."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from codeintel.serving.backend import (
    BackendLimits,
    DuckDBQueryService,
    clamp_limit_value,
    clamp_offset_value,
)
from codeintel.serving.backend.datasets import describe_dataset
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
    GraphScopePayload,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    ModuleArchitectureResponse,
    ModuleProfileResponse,
    ModuleSubsystemResponse,
    ResponseMeta,
    SubsystemCoverageResponse,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    TestsForFunctionResponse,
    parse_graph_scope,
)
from codeintel.storage.datasets import Dataset, load_dataset_registry

LOG = logging.getLogger("codeintel.serving.services.query")


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
    truncated: bool | None = None
    schema_version: str | None = None
    retries: int | None = None


@dataclass
class ServiceCallContext:
    """Context propagated into observability signals."""

    dataset: str | None = None
    schema_version: str | None = None
    retries: int | None = None


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
        if metrics.truncated is not None:
            payload["truncated"] = metrics.truncated
        if metrics.schema_version is not None:
            payload["schema_version"] = metrics.schema_version
        if metrics.retries is not None:
            payload["retries"] = metrics.retries
        self.logger.info("service_call %s", payload)


def _extract_row_count(result: object) -> int | None:
    """
    Attempt to derive a row count from common response shapes.

    Returns
    -------
    int | None
        Row count when inferrable; otherwise ``None``.
    """
    attr_counts: list[tuple[type, str]] = [
        (DatasetRowsResponse, "rows"),
        (HighRiskFunctionsResponse, "functions"),
        (TestsForFunctionResponse, "tests"),
        (SubsystemSummaryResponse, "subsystems"),
        (ModuleSubsystemResponse, "memberships"),
        (FileHintsResponse, "hints"),
        (SubsystemModulesResponse, "modules"),
        (SubsystemSearchResponse, "subsystems"),
        (SubsystemProfileResponse, "profiles"),
        (SubsystemCoverageResponse, "coverage"),
    ]
    if isinstance(result, CallGraphNeighborsResponse):
        return len(result.outgoing) + len(result.incoming)
    for response_type, attr in attr_counts:
        if isinstance(result, response_type):
            typed_result = cast("Any", result)
            return len(getattr(typed_result, attr))
    return None


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


def _extract_truncated(result: object) -> bool | None:
    """
    Return truncation state when available on response metadata.

    Returns
    -------
    bool | None
        Truncation flag if present; otherwise ``None``.
    """
    meta = getattr(result, "meta", None)
    if meta is None:
        return None
    truncated = getattr(meta, "truncated", None)
    return bool(truncated) if truncated is not None else None


def _normalize_validation_profile(
    value: str | None,
) -> Literal["strict", "lenient"] | None:
    """
    Normalize validation profile strings to allowed literal values.

    Returns
    -------
    Literal["strict", "lenient"] | None
        Normalized validation profile when valid.
    """
    if value == "strict":
        return "strict"
    if value == "lenient":
        return "lenient"
    return None


def _observe_call[T](
    observability: ServiceObservability | None,
    *,
    transport: str,
    name: str,
    context: ServiceCallContext | None,
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
                    dataset=context.dataset if context is not None else None,
                    schema_version=context.schema_version if context is not None else None,
                    retries=context.retries if context is not None else None,
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
                dataset=context.dataset if context is not None else None,
                messages=_extract_message_count(result),
                truncated=_extract_truncated(result),
                schema_version=context.schema_version if context is not None else None,
                retries=context.retries if context is not None else None,
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
        scope: GraphScopePayload | None = None,
    ) -> FunctionSummaryResponse:
        """Return a function summary for an identifier."""
        ...

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> HighRiskFunctionsResponse:
        """List high-risk functions."""
        ...

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> CallGraphNeighborsResponse:
        """Return call graph neighbors for a function."""
        ...

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
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

    def get_file_summary(
        self, *, rel_path: str, scope: GraphScopePayload | None = None
    ) -> FileSummaryResponse:
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

    def list_subsystem_profiles(self, *, limit: int | None = None) -> SubsystemProfileResponse:
        """List subsystem profiles from docs views."""
        ...

    def list_subsystem_coverage(self, *, limit: int | None = None) -> SubsystemCoverageResponse:
        """List subsystem coverage rollups from docs views."""
        ...


class DatasetQueryApi(Protocol):
    """Dataset listing and retrieval surface."""

    def list_datasets(self) -> list[DatasetDescriptor]:
        """List available datasets."""
        ...

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        """Return canonical dataset contract entries."""
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

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
        """Return schema and samples for a dataset."""
        ...


class QueryService(
    FunctionQueryApi,
    ProfileQueryApi,
    SubsystemQueryApi,
    DatasetQueryApi,
    Protocol,
):
    """
    Composite query service consumed by HTTP, MCP, and future transports.

    All application surfaces (FastAPI, MCP, CLI) must depend on this interface
    instead of touching DuckDB or raw SQL directly.

    Implementations:
        - LocalQueryService: wraps DuckDBQueryService for local DB access.
        - HttpQueryService: forwards calls to a remote HTTP server.
    """


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
        scope: GraphScopePayload | None = None,
    ) -> FunctionSummaryResponse:
        return self._call(
            "get_function_summary",
            lambda: self.query.get_function_summary(
                urn=urn,
                goid_h128=goid_h128,
                rel_path=rel_path,
                qualname=qualname,
                scope=parse_graph_scope(scope),
            ),
        )

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> HighRiskFunctionsResponse:
        return self._call(
            "list_high_risk_functions",
            lambda: self.query.list_high_risk_functions(
                min_risk=min_risk,
                limit=limit,
                tested_only=tested_only,
                scope=parse_graph_scope(scope),
            ),
        )

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> CallGraphNeighborsResponse:
        return self._call(
            "get_callgraph_neighbors",
            lambda: self.query.get_callgraph_neighbors(
                goid_h128=goid_h128,
                direction=direction,
                limit=limit,
                scope=parse_graph_scope(scope),
            ),
        )

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> TestsForFunctionResponse:
        return self._call(
            "get_tests_for_function",
            lambda: self.query.get_tests_for_function(
                goid_h128=goid_h128,
                urn=urn,
                limit=limit,
                scope=parse_graph_scope(scope),
            ),
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

    def get_file_summary(
        self, *, rel_path: str, scope: GraphScopePayload | None = None
    ) -> FileSummaryResponse:
        return self._call(
            "get_file_summary",
            lambda: self.query.get_file_summary(
                rel_path=rel_path,
                scope=parse_graph_scope(scope),
            ),
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
            "list_subsystems",
            lambda: self.query.list_subsystems(limit=limit, role=role, q=q),
        )

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        return self._call(
            "get_module_subsystems",
            lambda: self.query.get_module_subsystems(module=module),
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
            "search_subsystems",
            lambda: self.query.search_subsystems(limit=limit, role=role, q=q),
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

    def list_subsystem_profiles(self, *, limit: int | None = None) -> SubsystemProfileResponse:
        return self._call(
            "list_subsystem_profiles",
            lambda: self.query.list_subsystem_profiles(limit=limit),
            dataset="docs.v_subsystem_profile",
        )

    def list_subsystem_coverage(self, *, limit: int | None = None) -> SubsystemCoverageResponse:
        return self._call(
            "list_subsystem_coverage",
            lambda: self.query.list_subsystem_coverage(limit=limit),
            dataset="docs.v_subsystem_coverage",
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
        schema_version: str | None = None,
        retries: int | None = None,
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
            context=ServiceCallContext(
                dataset=dataset,
                schema_version=schema_version,
                retries=retries,
            ),
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
            registry = None
            if not mapping:
                query_gateway = getattr(self.query, "gateway", None)
                if query_gateway is not None:
                    mapping = query_gateway.datasets.mapping
                    registry = load_dataset_registry(query_gateway.con)
            if registry is None:
                registry = load_dataset_registry(self.query.gateway.con)
            results: list[DatasetDescriptor] = []
            for name, table in sorted(mapping.items()):
                ds: Dataset | None = registry.by_name.get(name) if registry is not None else None
                description = (
                    ds.description
                    if ds is not None and ds.description is not None
                    else self.describe_dataset_fn(name, table)
                )
                results.append(
                    DatasetDescriptor(
                        name=name,
                        table=table,
                        family=ds.family if ds is not None else None,
                        description=description,
                        owner=ds.owner if ds is not None else None,
                        freshness_sla=ds.freshness_sla if ds is not None else None,
                        retention_policy=ds.retention_policy if ds is not None else None,
                        schema_version=ds.schema_version if ds is not None else None,
                        stable_id=ds.stable_id if ds is not None else None,
                        validation_profile=_normalize_validation_profile(
                            ds.validation_profile if ds is not None else None
                        ),
                        capabilities=ds.capabilities() if ds is not None else {},
                    )
                )
            return results

        return self._call("list_datasets", _list)

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        """
        Return canonical dataset specs with filenames and schema metadata.

        Returns
        -------
        list[DatasetSpecDescriptor]
            Dataset specs sorted by name.
        """

        def _list_specs() -> list[DatasetSpecDescriptor]:
            return self.query.dataset_specs()

        return self._call("dataset_specs", _list_specs)

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
        """
        Return DuckDB + JSON Schema details and sample rows for a dataset.

        Returns
        -------
        DatasetSchemaResponse
            Composite schema and sample payload.
        """

        def _schema() -> DatasetSchemaResponse:
            return self.query.dataset_schema(dataset_name=dataset_name, sample_limit=sample_limit)

        registry = load_dataset_registry(self.query.gateway.con)
        schema_version = None
        if dataset_name in registry.by_name:
            schema_version = registry.by_name[dataset_name].schema_version
        return self._call(
            "dataset_schema",
            _schema,
            dataset=dataset_name,
            schema_version=schema_version,
        )

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
        registry = load_dataset_registry(self.query.gateway.con)
        schema_version = None
        if dataset_name in registry.by_name:
            schema_version = registry.by_name[dataset_name].schema_version
        return self._call(
            "read_dataset_rows",
            lambda: self.query.read_dataset_rows(
                dataset_name=dataset_name,
                limit=applied_limit,
                offset=offset,
            ),
            dataset=dataset_name,
            schema_version=schema_version,
        )


class _HttpTransportMixin:
    request_json: Callable[[str, dict[str, object]], object]
    limits: BackendLimits
    observability: ServiceObservability | None

    def _http_call[T](
        self,
        name: str,
        func: Callable[[], T],
        *,
        dataset: str | None = None,
        schema_version: str | None = None,
    ) -> T:
        backend = getattr(self.request_json, "__self__", None)
        retries = getattr(backend, "last_retry_attempts", None)
        result = _observe_call(
            self.observability,
            transport="http",
            name=name,
            context=ServiceCallContext(
                dataset=dataset,
                schema_version=schema_version,
                retries=retries if isinstance(retries, int) else None,
            ),
            func=func,
        )
        if retries and self.observability is not None:
            self.observability.record(
                ServiceCallMetrics(
                    name=f"{name}_retries",
                    transport="http",
                    duration_ms=0.0,
                    dataset=dataset,
                    retries=retries,
                    schema_version=schema_version,
                )
            )
        return result


class _HttpFunctionQueryMixin(_HttpTransportMixin):
    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> HighRiskFunctionsResponse:
        def _run() -> HighRiskFunctionsResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return HighRiskFunctionsResponse(
                    functions=[],
                    truncated=False,
                    meta=ResponseMeta(),
                )
            return HighRiskFunctionsResponse.model_validate(
                self.request_json(
                    "/functions/high-risk",
                    {
                        "min_risk": min_risk,
                        "limit": clamp.applied,
                        "tested_only": tested_only,
                        "scope": scope.model_dump() if scope is not None else None,
                    },
                )
            )

        return self._http_call("list_high_risk_functions", _run)

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphScopePayload | None = None,
    ) -> FunctionSummaryResponse:
        def _run() -> FunctionSummaryResponse:
            return FunctionSummaryResponse.model_validate(
                self.request_json(
                    "/function/summary",
                    {
                        "urn": urn,
                        "goid_h128": goid_h128,
                        "rel_path": rel_path,
                        "qualname": qualname,
                        "scope": scope.model_dump() if scope is not None else None,
                    },
                )
            )

        return self._http_call("get_function_summary", _run)

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> CallGraphNeighborsResponse:
        def _run() -> CallGraphNeighborsResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return CallGraphNeighborsResponse(outgoing=[], incoming=[], meta=ResponseMeta())
            return CallGraphNeighborsResponse.model_validate(
                self.request_json(
                    "/function/callgraph",
                    {
                        "goid_h128": goid_h128,
                        "direction": direction,
                        "limit": clamp.applied,
                        "scope": scope.model_dump() if scope is not None else None,
                    },
                )
            )

        return self._http_call("get_callgraph_neighbors", _run)

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> GraphNeighborhoodResponse:
        def _run() -> GraphNeighborhoodResponse:
            applied_limit = self.limits.default_limit if max_nodes is None else max_nodes
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return GraphNeighborhoodResponse(nodes=[], edges=[], meta=ResponseMeta())
            return GraphNeighborhoodResponse.model_validate(
                self.request_json(
                    "/graph/call/neighborhood",
                    {
                        "goid_h128": goid_h128,
                        "radius": radius,
                        "max_nodes": clamp.applied,
                    },
                )
            )

        return self._http_call(
            "get_callgraph_neighborhood",
            _run,
            dataset="call_graph_nodes",
        )

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> ImportBoundaryResponse:
        def _run() -> ImportBoundaryResponse:
            applied_limit = self.limits.default_limit if max_edges is None else max_edges
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return ImportBoundaryResponse(nodes=[], edges=[], meta=ResponseMeta())
            return ImportBoundaryResponse.model_validate(
                self.request_json(
                    "/graph/import/boundary",
                    {"subsystem_id": subsystem_id, "max_edges": clamp.applied},
                )
            )

        return self._http_call("get_import_boundary", _run, dataset="import_graph_edges")

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> TestsForFunctionResponse:
        def _run() -> TestsForFunctionResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return TestsForFunctionResponse(tests=[], meta=ResponseMeta())
            return TestsForFunctionResponse.model_validate(
                self.request_json(
                    "/function/tests",
                    {
                        "goid_h128": goid_h128,
                        "urn": urn,
                        "limit": clamp.applied,
                        "scope": scope.model_dump() if scope is not None else None,
                    },
                )
            )

        return self._http_call("get_tests_for_function", _run)

    def get_file_summary(
        self, *, rel_path: str, scope: GraphScopePayload | None = None
    ) -> FileSummaryResponse:
        def _run() -> FileSummaryResponse:
            return FileSummaryResponse.model_validate(
                self.request_json(
                    "/file/summary",
                    {
                        "rel_path": rel_path,
                        "scope": scope.model_dump() if scope is not None else None,
                    },
                )
            )

        return self._http_call("get_file_summary", _run)


class _HttpProfileQueryMixin(_HttpTransportMixin):
    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        def _run() -> FunctionProfileResponse:
            return FunctionProfileResponse.model_validate(
                self.request_json("/profiles/function", {"goid_h128": goid_h128})
            )

        return self._http_call("get_function_profile", _run)

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        def _run() -> FileProfileResponse:
            return FileProfileResponse.model_validate(
                self.request_json("/profiles/file", {"rel_path": rel_path})
            )

        return self._http_call("get_file_profile", _run)

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        def _run() -> ModuleProfileResponse:
            return ModuleProfileResponse.model_validate(
                self.request_json("/profiles/module", {"module": module})
            )

        return self._http_call("get_module_profile", _run)

    def get_function_architecture(self, *, goid_h128: int) -> FunctionArchitectureResponse:
        def _run() -> FunctionArchitectureResponse:
            return FunctionArchitectureResponse.model_validate(
                self.request_json("/architecture/function", {"goid_h128": goid_h128})
            )

        return self._http_call("get_function_architecture", _run)

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        def _run() -> ModuleArchitectureResponse:
            return ModuleArchitectureResponse.model_validate(
                self.request_json("/architecture/module", {"module": module})
            )

        return self._http_call("get_module_architecture", _run)


class _HttpSubsystemQueryMixin(_HttpTransportMixin):
    def list_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> SubsystemSummaryResponse:
        def _run() -> SubsystemSummaryResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return SubsystemSummaryResponse(subsystems=[], meta=ResponseMeta())
            return SubsystemSummaryResponse.model_validate(
                self.request_json(
                    "/architecture/subsystems",
                    {"limit": clamp.applied, "role": role, "q": q},
                )
            )

        return self._http_call("list_subsystems", _run)

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        def _run() -> ModuleSubsystemResponse:
            return ModuleSubsystemResponse.model_validate(
                self.request_json("/architecture/module-subsystems", {"module": module})
            )

        return self._http_call("get_module_subsystems", _run)

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        def _run() -> FileHintsResponse:
            return FileHintsResponse.model_validate(
                self.request_json("/ide/hints", {"rel_path": rel_path})
            )

        return self._http_call("get_file_hints", _run)

    def get_subsystem_modules(self, *, subsystem_id: str) -> SubsystemModulesResponse:
        def _run() -> SubsystemModulesResponse:
            return SubsystemModulesResponse.model_validate(
                self.request_json("/architecture/subsystem", {"subsystem_id": subsystem_id})
            )

        return self._http_call("get_subsystem_modules", _run)

    def search_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> SubsystemSearchResponse:
        def _run() -> SubsystemSearchResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return SubsystemSearchResponse(subsystems=[], meta=ResponseMeta())
            return SubsystemSearchResponse.model_validate(
                self.request_json(
                    "/architecture/subsystems",
                    {"limit": clamp.applied, "role": role, "q": q},
                )
            )

        return self._http_call("search_subsystems", _run)

    def summarize_subsystem(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> SubsystemModulesResponse:
        detail = self.get_subsystem_modules(subsystem_id=subsystem_id)
        if module_limit is None or not detail.modules:
            return detail
        return SubsystemModulesResponse(
            found=detail.found,
            subsystem=detail.subsystem,
            modules=detail.modules[:module_limit],
            meta=detail.meta,
        )

    def list_subsystem_profiles(self, *, limit: int | None = None) -> SubsystemProfileResponse:
        def _run() -> SubsystemProfileResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit, default=applied_limit, max_limit=self.limits.max_rows_per_call
            )
            if clamp.has_error:
                return SubsystemProfileResponse(profiles=[], meta=ResponseMeta())
            return SubsystemProfileResponse.model_validate(
                self.request_json(
                    "/architecture/subsystem-profiles",
                    {"limit": clamp.applied},
                )
            )

        return self._http_call(
            "list_subsystem_profiles",
            _run,
            dataset="docs.v_subsystem_profile",
        )

    def list_subsystem_coverage(self, *, limit: int | None = None) -> SubsystemCoverageResponse:
        def _run() -> SubsystemCoverageResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit, default=applied_limit, max_limit=self.limits.max_rows_per_call
            )
            if clamp.has_error:
                return SubsystemCoverageResponse(coverage=[], meta=ResponseMeta())
            return SubsystemCoverageResponse.model_validate(
                self.request_json(
                    "/architecture/subsystem-coverage",
                    {"limit": clamp.applied},
                )
            )

        return self._http_call(
            "list_subsystem_coverage",
            _run,
            dataset="docs.v_subsystem_coverage",
        )


class _HttpDatasetQueryMixin(_HttpTransportMixin):
    def list_datasets(self) -> list[DatasetDescriptor]:
        def _run() -> list[DatasetDescriptor]:
            data = cast("list[dict[str, object]]", self.request_json("/datasets", {}))
            return [DatasetDescriptor.model_validate(item) for item in data]

        return self._http_call("list_datasets", _run)

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        def _run() -> list[DatasetSpecDescriptor]:
            payload = cast("list[dict[str, object]]", self.request_json("/datasets/specs", {}))
            return [DatasetSpecDescriptor.model_validate(entry) for entry in payload]

        return self._http_call("dataset_specs", _run)

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        def _run() -> DatasetRowsResponse:
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

        return self._http_call("read_dataset_rows", _run, dataset=dataset_name)

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
        def _run() -> DatasetSchemaResponse:
            data = self.request_json(
                f"/datasets/{dataset_name}/schema",
                {"limit": sample_limit},
            )
            return DatasetSchemaResponse.model_validate(data)

        return self._http_call("dataset_schema", _run, dataset=dataset_name)


@dataclass
class HttpQueryService(
    _HttpDatasetQueryMixin,
    _HttpFunctionQueryMixin,
    _HttpProfileQueryMixin,
    _HttpSubsystemQueryMixin,
):
    """Application service that forwards queries to a remote HTTP API."""

    request_json: Callable[[str, dict[str, object]], object]
    limits: BackendLimits
    observability: ServiceObservability | None = None
