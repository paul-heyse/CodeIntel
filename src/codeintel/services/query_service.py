"""Transport-agnostic query application services."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, cast

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


class QueryService(Protocol):
    """Shared query service surface consumed by HTTP and MCP transports."""

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> FunctionSummaryResponse:
        """Return a function summary."""
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

    def get_file_summary(self, *, rel_path: str) -> FileSummaryResponse:
        """Return a file summary."""
        ...

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
        """Return subsystem detail and member modules."""
        ...

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSearchResponse:
        """Search subsystems with optional filters."""
        ...

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """Summarize a subsystem with optional module truncation."""
        ...

    def list_datasets(self) -> list[DatasetDescriptor]:
        """List datasets available to the service."""
        ...

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """Read rows from a registered dataset."""
        ...


@dataclass
class LocalQueryService:
    """Application service backed by a local DuckDB query layer."""

    query: DuckDBQueryService
    dataset_tables: dict[str, str] | None = None
    describe_dataset_fn: Callable[[str, str], str] = describe_dataset
    observability: ServiceObservability | None = None

    def __getattr__(self, name: str) -> object:
        """
        Delegate attribute access to the underlying DuckDB query service.

        Returns
        -------
        object
            Attribute resolved from the wrapped query service.
        """
        return getattr(self.query, name)

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
            Dataset descriptors with descriptions.
        """
        def _list() -> list[DatasetDescriptor]:
            if not self.dataset_tables:
                return []
            return [
                DatasetDescriptor(
                    name=name,
                    table=table,
                    description=self.describe_dataset_fn(name, table),
                )
                for name, table in sorted(self.dataset_tables.items())
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

        Parameters
        ----------
        dataset_name:
            Registry name for the dataset.
        limit:
            Optional requested row limit; defaults to service limit when None.
        offset:
            Requested offset; must be non-negative.

        Returns
        -------
        DatasetRowsResponse
            Dataset slice with metadata.
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

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> FunctionSummaryResponse:
        """
        Return a function summary identified by GOID, URN, or path/qualname.

        Parameters
        ----------
        urn : str, optional
            Stable URN that uniquely identifies the function.
        goid_h128 : int, optional
            Stable 128-bit GOID for the function.
        rel_path : str, optional
            Repository-relative path to the file containing the function.
        qualname : str, optional
            Dotted qualified name within the file.

        Returns
        -------
        FunctionSummaryResponse
            Summary payload with coverage and metadata fields.
        """
        return self._call(
            "get_function_summary",
            lambda: self.query.get_function_summary(
                urn=urn,
                goid_h128=goid_h128,
                rel_path=rel_path,
                qualname=qualname,
            ),
        )

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
    ) -> HighRiskFunctionsResponse:
        """
        Delegate high-risk function listing to the DuckDB query service.

        Parameters
        ----------
        min_risk : float
            Minimum risk score threshold inclusive.
        limit : int, optional
            Optional maximum rows to return; uses service default when None.
        tested_only : bool
            When True, restrict results to functions with test coverage.

        Returns
        -------
        HighRiskFunctionsResponse
            Functions ordered by risk with metadata and truncation flags.
        """
        return self._call(
            "list_high_risk_functions",
            lambda: self.query.list_high_risk_functions(
                min_risk=min_risk,
                limit=limit,
                tested_only=tested_only,
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
        Return incoming/outgoing call graph neighbors for a function.

        Parameters
        ----------
        goid_h128 : int
            Stable 128-bit GOID for the function.
        direction : str
            Neighbor direction: "incoming", "outgoing", or "both".
        limit : int, optional
            Optional maximum edges to return; uses service default when None.

        Returns
        -------
        CallGraphNeighborsResponse
            Incoming and outgoing edges with metadata.
        """
        return self._call(
            "get_callgraph_neighbors",
            lambda: self.query.get_callgraph_neighbors(
                goid_h128=goid_h128,
                direction=direction,
                limit=limit,
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
        List tests that exercise a function.

        Parameters
        ----------
        goid_h128 : int, optional
            Stable 128-bit GOID for the function.
        urn : str, optional
            Stable URN that uniquely identifies the function.
        limit : int, optional
            Optional maximum rows to return; uses service default when None.

        Returns
        -------
        TestsForFunctionResponse
            Tests covering the function along with metadata.
        """
        return self._call(
            "get_tests_for_function",
            lambda: self.query.get_tests_for_function(
                goid_h128=goid_h128,
                urn=urn,
                limit=limit,
            ),
        )

    def get_file_summary(self, *, rel_path: str) -> FileSummaryResponse:
        """
        Return file summary from the local DuckDB-backed dataset.

        Parameters
        ----------
        rel_path : str
            Repository-relative path of the file to summarize.

        Returns
        -------
        FileSummaryResponse
            Summary payload including functions contained within the file.
        """
        return self._call(
            "get_file_summary",
            lambda: self.query.get_file_summary(rel_path=rel_path),
        )

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        """
        Return a denormalized function profile.

        Parameters
        ----------
        goid_h128 : int
            Stable 128-bit GOID for the function.

        Returns
        -------
        FunctionProfileResponse
            Profile payload with churn and coverage metrics.
        """
        return self._call(
            "get_function_profile",
            lambda: self.query.get_function_profile(goid_h128=goid_h128),
        )

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        """
        Return a profile for a repo-relative file path.

        Parameters
        ----------
        rel_path : str
            Repository-relative path for the file.

        Returns
        -------
        FileProfileResponse
            Profile payload with churn and coverage data.
        """
        return self._call(
            "get_file_profile",
            lambda: self.query.get_file_profile(rel_path=rel_path),
        )

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        """
        Return a module profile including coverage and import metrics.

        Parameters
        ----------
        module : str
            Dotted module path within the repository.

        Returns
        -------
        ModuleProfileResponse
            Profile payload with churn and coverage data.
        """
        return self._call(
            "get_module_profile",
            lambda: self.query.get_module_profile(module=module),
        )

    def get_function_architecture(self, *, goid_h128: int) -> FunctionArchitectureResponse:
        """
        Return call-graph architecture metrics for a function.

        Parameters
        ----------
        goid_h128 : int
            Stable 128-bit GOID for the function.

        Returns
        -------
        FunctionArchitectureResponse
            Fan-in/fan-out counts and dependency metadata.
        """
        return self._call(
            "get_function_architecture",
            lambda: self.query.get_function_architecture(goid_h128=goid_h128),
        )

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        """
        Return import-graph architecture metrics for a module.

        Parameters
        ----------
        module : str
            Dotted module path within the repository.

        Returns
        -------
        ModuleArchitectureResponse
            Fan-in/fan-out counts and subsystem metadata.
        """
        return self._call(
            "get_module_architecture",
            lambda: self.query.get_module_architecture(module=module),
        )

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSummaryResponse:
        """
        List inferred subsystems with optional filters.

        Parameters
        ----------
        limit : int, optional
            Optional maximum rows to return; uses service default when None.
        role : str, optional
            Optional subsystem role filter.
        q : str, optional
            Optional search query for subsystem name or description.

        Returns
        -------
        SubsystemSummaryResponse
            Subsystems with summary metadata and truncation flags.
        """
        return self._call(
            "list_subsystems",
            lambda: self.query.list_subsystems(limit=limit, role=role, q=q),
        )

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        """
        Return subsystem memberships for a module.

        Parameters
        ----------
        module : str
            Dotted module path within the repository.

        Returns
        -------
        ModuleSubsystemResponse
            Subsystem membership rows with metadata.
        """
        return self._call(
            "get_module_subsystems",
            lambda: self.query.get_module_subsystems(module=module),
        )

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        """
        Return IDE hints (subsystem/module context) for a file.

        Parameters
        ----------
        rel_path : str
            Repository-relative path for the file.

        Returns
        -------
        FileHintsResponse
            Hint rows and metadata for IDE integrations.
        """
        return self._call("get_file_hints", lambda: self.query.get_file_hints(rel_path=rel_path))

    def get_subsystem_modules(self, *, subsystem_id: str) -> SubsystemModulesResponse:
        """
        Return subsystem detail and member modules.

        Parameters
        ----------
        subsystem_id : str
            Stable identifier for the subsystem.

        Returns
        -------
        SubsystemModulesResponse
            Subsystem metadata plus member modules.
        """
        return self.query.get_subsystem_modules(subsystem_id=subsystem_id)

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSearchResponse:
        """
        Search subsystems with optional filters.

        Parameters
        ----------
        limit : int, optional
            Optional maximum rows to return; uses service default when None.
        role : str, optional
            Optional subsystem role filter.
        q : str, optional
            Optional search query for subsystem name or description.

        Returns
        -------
        SubsystemSearchResponse
            Subsystem matches and metadata.
        """
        return self._call(
            "search_subsystems",
            lambda: self.query.search_subsystems(limit=limit, role=role, q=q),
        )

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """
        Summarize a subsystem, optionally truncating modules.

        Parameters
        ----------
        subsystem_id : str
            Stable identifier for the subsystem.
        module_limit : int, optional
            Optional cap on the number of modules returned.

        Returns
        -------
        SubsystemModulesResponse
            Subsystem detail payload, possibly truncated.
        """
        detail = self._call(
            "get_subsystem_modules",
            lambda: self.query.get_subsystem_modules(subsystem_id=subsystem_id),
        )
        if not detail.found or detail.subsystem is None:
            return detail
        if module_limit is None:
            return detail
        limited_modules = detail.modules[:module_limit]
        return SubsystemModulesResponse(
            found=detail.found,
            subsystem=detail.subsystem,
            modules=limited_modules,
            meta=detail.meta,
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
        if response.meta is None:
            response.meta = meta
        else:
            response.meta.requested_limit = meta.requested_limit
            response.meta.applied_limit = meta.applied_limit
            response.meta.requested_offset = meta.requested_offset
            response.meta.applied_offset = meta.applied_offset
            response.meta.messages = [*meta.messages, *response.meta.messages]
        return response
