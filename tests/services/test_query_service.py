"""Unit tests for LocalQueryService and HttpQueryService."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import NamedTuple, Protocol, cast, runtime_checkable

import pytest
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.http.datasets import build_registry_and_limits, validate_dataset_registry
from codeintel.serving.http.fastapi import BackendResource, create_app
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.backend import QueryBackend
from codeintel.serving.mcp.models import (
    FunctionSummaryResponse,
    ResponseMeta,
    SubsystemCoverageResponse,
    SubsystemCoverageRow,
    SubsystemProfileResponse,
    SubsystemProfileRow,
)
from codeintel.serving.mcp.query_service import BackendLimits, DuckDBQueryService
from codeintel.serving.services.query_service import (
    HttpQueryService,
    LocalQueryService,
    ServiceCallMetrics,
    ServiceObservability,
)
from codeintel.storage.gateway import StorageGateway
from tests._helpers.architecture import open_seeded_architecture_gateway


@runtime_checkable
class _SupportsModelDump(Protocol):
    """Protocol for objects exposing a pydantic-style model_dump."""

    def model_dump(self) -> object:
        """Return a serializable representation of the model."""


class HandlerArgSpec(NamedTuple):
    """Argument extraction spec for HTTP adapter handlers."""

    arg_name: str
    key: str
    converter: Callable[[object | None], object | None]
    default: object | None = None


def _as_str(value: object | None) -> str | None:
    return cast("str | None", value)


def _as_int(value: object | None) -> int | None:
    return cast("int | None", value)


def _as_float(value: object | None) -> float:
    return cast("float", value)


def _as_bool(value: object | None) -> bool:
    return cast("bool", value)


def _invoke_method(
    method: Callable[..., object],
    specs: tuple[HandlerArgSpec, ...],
    params: dict[str, object],
) -> object:
    kwargs = {spec.arg_name: spec.converter(params.get(spec.key, spec.default)) for spec in specs}
    result = method(**kwargs)
    if isinstance(result, _SupportsModelDump):
        return result.model_dump()
    return result


def _invoke_list_datasets(
    local: LocalQueryService, _params: dict[str, object]
) -> list[dict[str, object]]:
    return [descriptor.model_dump() for descriptor in local.list_datasets()]


class RecordingObservability(ServiceObservability):
    """Observability sink that records metrics for assertions."""

    def __init__(self) -> None:
        logger = logging.getLogger("test_observability")
        logger.setLevel(logging.INFO)
        super().__init__(enabled=True, logger=logger)
        self.records: list[ServiceCallMetrics] = []

    def record(self, metrics: ServiceCallMetrics) -> None:
        """Capture a recorded metrics payload."""
        self.records.append(metrics)


class _StubSubsystemQuery:
    """Minimal query stub for subsystem docs observability tests."""

    def __init__(self) -> None:
        self.profile = SubsystemProfileRow(
            repo="demo/repo",
            commit="deadbeef",
            subsystem_id="subsysdemo",
            name="Subsystem Demo",
        )
        self.coverage = SubsystemCoverageRow(
            repo="demo/repo",
            commit="deadbeef",
            subsystem_id="subsysdemo",
            test_count=2,
        )

    def list_subsystem_profiles(self, *, limit: int | None = None) -> SubsystemProfileResponse:
        return SubsystemProfileResponse(
            profiles=[self.profile],
            meta=ResponseMeta(requested_limit=limit, applied_limit=limit),
        )

    def list_subsystem_coverage(self, *, limit: int | None = None) -> SubsystemCoverageResponse:
        return SubsystemCoverageResponse(
            coverage=[self.coverage],
            meta=ResponseMeta(requested_limit=limit, applied_limit=limit),
        )


def _build_http_adapter_from_local(
    local: LocalQueryService, limits: BackendLimits
) -> HttpQueryService:
    handlers: dict[str, Callable[[dict[str, object]], object]] = {
        "/function/summary": partial(
            _invoke_method,
            local.get_function_summary,
            (
                HandlerArgSpec("urn", "urn", _as_str),
                HandlerArgSpec("goid_h128", "goid_h128", _as_int),
                HandlerArgSpec("rel_path", "rel_path", _as_str),
                HandlerArgSpec("qualname", "qualname", _as_str),
            ),
        ),
        "/functions/high-risk": partial(
            _invoke_method,
            local.list_high_risk_functions,
            (
                HandlerArgSpec("min_risk", "min_risk", _as_float, default=0.7),
                HandlerArgSpec("limit", "limit", _as_int),
                HandlerArgSpec("tested_only", "tested_only", _as_bool, default=False),
            ),
        ),
        "/function/callgraph": partial(
            _invoke_method,
            local.get_callgraph_neighbors,
            (
                HandlerArgSpec("goid_h128", "goid_h128", _as_int),
                HandlerArgSpec("direction", "direction", _as_str, default="both"),
                HandlerArgSpec("limit", "limit", _as_int),
            ),
        ),
        "/function/tests": partial(
            _invoke_method,
            local.get_tests_for_function,
            (
                HandlerArgSpec("goid_h128", "goid_h128", _as_int),
                HandlerArgSpec("urn", "urn", _as_str),
                HandlerArgSpec("limit", "limit", _as_int),
            ),
        ),
        "/file/summary": partial(
            _invoke_method,
            local.get_file_summary,
            (HandlerArgSpec("rel_path", "rel_path", _as_str),),
        ),
        "/profiles/function": partial(
            _invoke_method,
            local.get_function_profile,
            (HandlerArgSpec("goid_h128", "goid_h128", _as_int),),
        ),
        "/profiles/file": partial(
            _invoke_method,
            local.get_file_profile,
            (HandlerArgSpec("rel_path", "rel_path", _as_str),),
        ),
        "/profiles/module": partial(
            _invoke_method,
            local.get_module_profile,
            (HandlerArgSpec("module", "module", _as_str),),
        ),
        "/architecture/function": partial(
            _invoke_method,
            local.get_function_architecture,
            (HandlerArgSpec("goid_h128", "goid_h128", _as_int),),
        ),
        "/architecture/module": partial(
            _invoke_method,
            local.get_module_architecture,
            (HandlerArgSpec("module", "module", _as_str),),
        ),
        "/architecture/subsystems": partial(
            _invoke_method,
            local.list_subsystems,
            (
                HandlerArgSpec("limit", "limit", _as_int),
                HandlerArgSpec("role", "role", _as_str),
                HandlerArgSpec("q", "q", _as_str),
            ),
        ),
        "/architecture/subsystem": partial(
            _invoke_method,
            local.get_subsystem_modules,
            (HandlerArgSpec("subsystem_id", "subsystem_id", _as_str),),
        ),
        "/ide/hints": partial(
            _invoke_method,
            local.get_file_hints,
            (HandlerArgSpec("rel_path", "rel_path", _as_str),),
        ),
        "/datasets": partial(_invoke_list_datasets, local),
    }

    def request_json(path: str, params: dict[str, object]) -> object:
        handler = handlers.get(path)
        if handler is not None:
            return handler(params)
        if path.startswith("/datasets/"):
            dataset_name = path.rsplit("/", maxsplit=1)[-1]
            offset = _as_int(params.get("offset", 0)) or 0
            return local.read_dataset_rows(
                dataset_name=dataset_name,
                limit=_as_int(params.get("limit")),
                offset=offset,
            ).model_dump()
        message = f"Unhandled HTTP path: {path} params={params}"
        raise AssertionError(message)

    return HttpQueryService(
        request_json=request_json,
        limits=limits,
        observability=local.observability,
    )


def test_observability_tracks_subsystem_profile_calls() -> None:
    """Subsystem profile listing should record dataset and row count."""
    obs = RecordingObservability()
    local = LocalQueryService(
        query=cast("DuckDBQueryService", _StubSubsystemQuery()),
        dataset_tables={},
        observability=obs,
    )

    resp = local.list_subsystem_profiles(limit=3)

    if not resp.profiles:
        pytest.fail("Expected subsystem profiles in response")
    metrics = obs.records[-1]
    if metrics.dataset != "docs.v_subsystem_profile":
        pytest.fail("Expected dataset to be set for subsystem profile call")
    if metrics.rows != 1:
        pytest.fail("Expected row count to be recorded for subsystem profile call")
    if metrics.transport != "local":
        pytest.fail("Expected transport to be local")


def test_observability_tracks_subsystem_coverage_calls() -> None:
    """Subsystem coverage listing should record dataset and row count."""
    obs = RecordingObservability()
    local = LocalQueryService(
        query=cast("DuckDBQueryService", _StubSubsystemQuery()),
        dataset_tables={},
        observability=obs,
    )

    resp = local.list_subsystem_coverage(limit=2)

    if not resp.coverage:
        pytest.fail("Expected subsystem coverage in response")
    metrics = obs.records[-1]
    if metrics.dataset != "docs.v_subsystem_coverage":
        pytest.fail("Expected dataset to be set for subsystem coverage call")
    if metrics.rows != 1:
        pytest.fail("Expected row count to be recorded for subsystem coverage call")
    if metrics.transport != "local":
        pytest.fail("Expected transport to be local")


def _build_local_service(
    gateway: StorageGateway,
    *,
    limits: BackendLimits | None = None,
    observability: ServiceObservability | None = None,
) -> LocalQueryService:
    effective_limits = limits or BackendLimits(default_limit=3, max_rows_per_call=5)
    registry = dict(gateway.datasets.mapping)
    validate_dataset_registry(gateway)
    repo = getattr(getattr(gateway, "config", None), "repo", "demo/repo") or "demo/repo"
    commit = getattr(getattr(gateway, "config", None), "commit", "deadbeef") or "deadbeef"
    query = DuckDBQueryService(
        gateway=gateway,
        repo=repo,
        commit=commit,
        limits=effective_limits,
    )
    return LocalQueryService(
        query=query,
        dataset_tables=registry,
        observability=observability,
    )


def _seed_call_graph_edges(gateway: StorageGateway) -> None:
    gateway.con.execute(
        """
        INSERT INTO graph.call_graph_edges (
            repo,
            commit,
            caller_goid_h128,
            callee_goid_h128,
            callsite_path,
            callsite_line,
            callsite_col,
            language,
            kind,
            resolved_via,
            confidence,
            evidence_json
        )
        VALUES ('demo/repo', 'deadbeef', 101, NULL, 'pkg/mod.py', 1, 0, 'python', 'unresolved', 'unresolved', 0.0, '{}')
        """
    )


def test_local_read_dataset_rows_defaults_limit(architecture_gateway: StorageGateway) -> None:
    """Local service uses backend default limit when none is provided."""
    limits = BackendLimits(default_limit=3, max_rows_per_call=5)
    _seed_call_graph_edges(architecture_gateway)
    service = _build_local_service(architecture_gateway, limits=limits)
    resp = service.read_dataset_rows(dataset_name="call_graph_edges", limit=None, offset=0)
    if resp.limit != limits.default_limit:
        message = f"Unexpected applied limit: {resp.limit}"
        pytest.fail(message)
    if resp.meta is None or resp.meta.applied_limit != limits.default_limit:
        pytest.fail(f"Meta did not capture applied limit: {resp.meta}")


def test_local_read_dataset_rows_propagates_errors(architecture_gateway: StorageGateway) -> None:
    """Local service does not swallow errors from the query layer."""
    service = _build_local_service(architecture_gateway)
    with pytest.raises(errors.McpError, match="Unknown dataset"):
        service.read_dataset_rows(dataset_name="missing", limit=1, offset=0)


def test_local_read_dataset_rows_records_observability(
    architecture_gateway: StorageGateway,
) -> None:
    """Dataset reads emit observability metrics with dataset, rows, and truncation."""
    observability = RecordingObservability()
    _seed_call_graph_edges(architecture_gateway)
    service = _build_local_service(
        architecture_gateway,
        limits=BackendLimits(default_limit=2, max_rows_per_call=5),
        observability=observability,
    )
    resp = service.read_dataset_rows(dataset_name="call_graph_edges", limit=1, offset=0)

    if not observability.records:
        pytest.fail("Expected observability record")
    metrics = observability.records[0]
    expected_rows = len(resp.rows)
    if metrics.dataset != "call_graph_edges":
        pytest.fail(f"Unexpected dataset in metrics: {metrics}")
    if metrics.rows != expected_rows:
        pytest.fail(f"Unexpected row count in metrics: {metrics}")
    if metrics.truncated != resp.meta.truncated:
        pytest.fail(f"Truncation flag missing in metrics: {metrics}")


def test_http_read_dataset_rows_clamps_limit() -> None:
    """HTTP service clamps large limits and forwards applied values."""
    calls: list[tuple[str, dict[str, object]]] = []

    def request_json(path: str, params: dict[str, object]) -> dict[str, object]:
        calls.append((path, params))
        return {
            "dataset": "functions",
            "limit": params["limit"],
            "offset": params["offset"],
            "rows": [{"issue": "ok"}],
        }

    limits = BackendLimits(default_limit=10, max_rows_per_call=20)
    service = HttpQueryService(request_json=request_json, limits=limits)
    resp = service.read_dataset_rows(dataset_name="functions", limit=1000, offset=0)
    expected_call = [("/datasets/functions", {"limit": 20, "offset": 0})]
    if calls != expected_call:
        message = f"Unexpected HTTP calls: {calls}"
        pytest.fail(message)
    codes = {msg.code for msg in resp.meta.messages}
    if "limit_clamped" not in codes:
        message = f"Missing clamp message: {codes}"
        pytest.fail(message)


def test_http_read_dataset_rows_invalid_offset() -> None:
    """HTTP service returns empty rows and messages when offset is invalid."""
    limits = BackendLimits(default_limit=5, max_rows_per_call=10)
    service = HttpQueryService(request_json=lambda _p, _q: {}, limits=limits)
    resp = service.read_dataset_rows(dataset_name="functions", limit=2, offset=-1)
    if resp.rows:
        pytest.fail("Expected no rows when offset is invalid")
    codes = {msg.code for msg in resp.meta.messages}
    if "offset_invalid" not in codes:
        message = f"Missing offset_invalid message: {codes}"
        pytest.fail(message)


def test_http_high_risk_clamps_using_limits() -> None:
    """High-risk listing uses clamped limits before issuing HTTP calls."""
    limits = BackendLimits(default_limit=5, max_rows_per_call=5)
    requested: dict[str, object] = {}

    def capture(path: str, params: dict[str, object]) -> dict[str, object]:
        requested["path"] = path
        requested["params"] = params
        return {"functions": [], "truncated": False, "meta": {}}

    service = HttpQueryService(request_json=capture, limits=limits)
    _resp = service.list_high_risk_functions(limit=20)
    expected_params = {"min_risk": 0.7, "limit": 5, "tested_only": False}
    if requested.get("path") != "/functions/high-risk":
        pytest.fail(f"Unexpected path: {requested}")
    if requested.get("params") != expected_params:
        pytest.fail(f"Unexpected params: {requested}")


def test_fastapi_delegates_to_query_service(tmp_path: Path) -> None:
    """FastAPI routes delegate to the injected QueryService instance."""
    calls: list[str] = []

    gateway = open_seeded_architecture_gateway(repo="demo/repo", commit="deadbeef")
    limits = BackendLimits(default_limit=3, max_rows_per_call=5)

    class RecordingService(LocalQueryService):
        """Service that records method calls."""

        def __init__(self) -> None:
            query = DuckDBQueryService(
                gateway=gateway,
                repo="demo/repo",
                commit="deadbeef",
                limits=limits,
            )
            super().__init__(query=query, dataset_tables=dict(gateway.datasets.mapping))

        def get_function_summary(
            self,
            *,
            urn: str | None = None,
            goid_h128: int | None = None,
            rel_path: str | None = None,
            qualname: str | None = None,
        ) -> FunctionSummaryResponse:
            _ = (goid_h128, rel_path)
            calls.append("get_function_summary")
            return super().get_function_summary(
                urn=urn,
                goid_h128=goid_h128,
                rel_path=rel_path,
                qualname=qualname,
            )

    service = RecordingService()
    backend_obj = type("StubBackend", (), {"service": service})()
    backend = cast("QueryBackend", backend_obj)

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="remote_api",
            repo_root=tmp_path,
            repo="demo/repo",
            commit="deadbeef",
            api_base_url="http://test",
            read_only=True,
        )

    def backend_factory(_: ServingConfig, *, _gateway: object | None = None) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)
    with TestClient(app) as client:
        response = client.get("/function/summary", params={"goid_h128": 1})
        if response.status_code != HTTPStatus.OK:
            message = f"Unexpected status: {response.status_code}"
            pytest.fail(message)
    if "get_function_summary" not in calls:
        pytest.fail("Service was not invoked by route")
    gateway.close()


def test_local_query_service_reads_architecture_seed(
    architecture_gateway: StorageGateway,
) -> None:
    """LocalQueryService accesses DuckDB queries via the shared registry."""
    cfg = ServingConfig(repo="demo/repo", commit="deadbeef")
    registry, limits = build_registry_and_limits(cfg)
    validate_dataset_registry(architecture_gateway)
    query = DuckDBQueryService(
        gateway=architecture_gateway,
        repo=cfg.repo,
        commit=cfg.commit,
        limits=limits,
    )
    service = LocalQueryService(query=query, dataset_tables=registry)

    profile = service.get_function_profile(goid_h128=1)
    if not profile.found:
        pytest.fail("Function profile not found in seeded gateway")
    dataset_names = {dataset.name for dataset in service.list_datasets()}
    if "function_profile" not in dataset_names:
        message = f"Registry missing function_profile: {sorted(dataset_names)}"
        pytest.fail(message)


def test_mcp_tool_delegation() -> None:
    """MCP tool registration delegates to the provided service."""
    fastmcp_mod = pytest.importorskip("mcp.server.fastmcp")
    registry = importlib.import_module("codeintel.serving.mcp.registry")

    calls: list[str] = []

    gateway = open_seeded_architecture_gateway(repo="demo/repo", commit="deadbeef")
    limits = BackendLimits(default_limit=3, max_rows_per_call=5)

    class DummyService(LocalQueryService):
        """Service that records method invocations."""

        def __init__(self) -> None:
            query = DuckDBQueryService(
                gateway=gateway,
                repo="demo/repo",
                commit="deadbeef",
                limits=limits,
            )
            super().__init__(query=query, dataset_tables=dict(gateway.datasets.mapping))

        def get_function_summary(
            self,
            *,
            urn: str | None = None,
            goid_h128: int | None = None,
            rel_path: str | None = None,
            qualname: str | None = None,
        ) -> FunctionSummaryResponse:
            _ = (goid_h128, rel_path)
            calls.append("get_function_summary")
            return super().get_function_summary(
                urn=urn,
                goid_h128=goid_h128,
                rel_path=rel_path,
                qualname=qualname,
            )

    mcp = fastmcp_mod.FastMCP("test")
    if not hasattr(mcp, "tools"):
        pytest.skip("FastMCP tools registry not available")
    service = DummyService()
    registry.register_tools(mcp, service)
    tool = mcp.tools["get_function_summary"]
    _result = tool({"urn": "urn:foo"})
    if "get_function_summary" not in calls:
        pytest.fail("Service was not invoked via MCP tool")
    gateway.close()


def test_query_service_contract_local_vs_http(
    architecture_gateway: StorageGateway,
) -> None:
    """Ensure QueryService surface is callable via both Local and HTTP implementations."""
    cfg = ServingConfig(repo="demo/repo", commit="deadbeef")
    registry, limits = build_registry_and_limits(cfg)
    validate_dataset_registry(architecture_gateway)
    query = DuckDBQueryService(
        gateway=architecture_gateway,
        repo=cfg.repo,
        commit=cfg.commit,
        limits=limits,
    )
    local = LocalQueryService(query=query, dataset_tables=registry)
    http = _build_http_adapter_from_local(local, limits)
    cases = [
        ("get_function_summary", lambda svc: svc.get_function_summary(goid_h128=1)),
        ("list_high_risk_functions", lambda svc: svc.list_high_risk_functions(limit=5)),
        (
            "get_callgraph_neighbors",
            lambda svc: svc.get_callgraph_neighbors(goid_h128=1, direction="both", limit=5),
        ),
        ("get_tests_for_function", lambda svc: svc.get_tests_for_function(goid_h128=1, limit=5)),
        ("get_file_summary", lambda svc: svc.get_file_summary(rel_path="pkg/mod.py")),
        ("get_function_profile", lambda svc: svc.get_function_profile(goid_h128=1)),
        ("get_file_profile", lambda svc: svc.get_file_profile(rel_path="pkg/mod.py")),
        ("get_module_profile", lambda svc: svc.get_module_profile(module="pkg.mod")),
        ("get_function_architecture", lambda svc: svc.get_function_architecture(goid_h128=1)),
        ("get_module_architecture", lambda svc: svc.get_module_architecture(module="pkg.mod")),
        ("list_subsystems", lambda svc: svc.list_subsystems(limit=5)),
        (
            "get_subsystem_modules",
            lambda svc: svc.get_subsystem_modules(subsystem_id="subsysdemo"),
        ),
        ("get_file_hints", lambda svc: svc.get_file_hints(rel_path="pkg/mod.py")),
        ("list_datasets", lambda svc: svc.list_datasets()),
        (
            "read_dataset_rows",
            lambda svc: svc.read_dataset_rows(dataset_name="function_profile", limit=5),
        ),
    ]

    def _normalize(result: object) -> object:
        if isinstance(result, list):
            normalized: list[object] = []
            for item in result:
                if isinstance(item, _SupportsModelDump):
                    normalized.append(item.model_dump())
                else:
                    normalized.append(item)
            return normalized
        if isinstance(result, _SupportsModelDump):
            return result.model_dump()
        return result

    for name, call in cases:
        local_resp = call(local)
        http_resp = call(http)
        if local_resp.__class__ is not http_resp.__class__:
            pytest.fail(f"{name} response types diverged: {type(local_resp)} vs {type(http_resp)}")
        if _normalize(local_resp) != _normalize(http_resp):
            pytest.fail(f"{name} response payloads diverged")
