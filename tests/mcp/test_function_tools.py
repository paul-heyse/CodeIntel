"""Tests for function MCP tool wrappers."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.context import get_current_request_context
from codeintel.serving.mcp.function_tools import FunctionToolOptions, register_function_tools
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from codeintel.serving.operations.catalog import DataSourceType, Operation
from tests._helpers.assertions import (
    assert_logged,
    expect_equal,
    expect_true,
)
from tests._helpers.gateway import GatewayFactory
from tests._helpers.mcp_registrar import RecordingMcpRegistrar as RecordingMcp

LOGGER = logging.getLogger(__name__)


@dataclass
class _DomainModel:
    payload: object

    def model_dump(self) -> dict[str, object]:
        if isinstance(self.payload, Mapping):
            return dict(self.payload)
        return {"value": str(self.payload)}

    @classmethod
    def from_domain(cls, payload: object) -> _DomainModel:
        return cls(payload=payload)


class _Backend:
    def __init__(self) -> None:
        self.gateway = GatewayFactory().with_macros().open()
        self.limits = BackendLimits()
        self.repo = self.gateway.config.repo or "demo/repo"
        self.commit = self.gateway.config.commit or "deadbeef"
        self.captured_scope: str | None = None
        self.calls: list[str] = []

    def get_fn(self, *, payload: object = "x") -> _DomainModel:
        self.calls.append("get_fn")
        enriched = {
            "id": "fn-alpha",
            "payload": payload,
            "summary": "helper fn",
            "notes": None,
        }
        return _DomainModel.from_domain(enriched)

    def get_fn_with_scope(self, *, scope: str | None = None) -> _DomainModel:
        ctx = get_current_request_context()
        if ctx is not None:
            self.captured_scope = ctx.graph_scope
        return _DomainModel.from_domain(scope or "none")

    def get_raw(self, **_: object) -> dict[str, object]:
        self.calls.append("get_raw")
        return {"id": "raw-fn", "name": "raw函数", "metadata": None}

    def get_domain_from_str(self, **_: object) -> str:
        self.calls.append("get_domain_from_str")
        return "payload"

    def raise_error(self, **_: object) -> _DomainModel:
        self.calls.append("raise_error")
        message = "fail"
        LOGGER.error("Function tool failure: %s", message)
        raise RuntimeError(message)

    def validate_fn(self, **_: object) -> str:
        self.calls.append("validate_fn")
        return "validated"


class _ValidatingModel:
    payload: object

    @classmethod
    def model_validate(cls, payload: object) -> _DomainModel:
        return _DomainModel.from_domain(payload)


def test_register_function_tools_registers_and_executes() -> None:
    """Function tool registration wraps backend calls with context and serialization."""
    spec = Operation(
        id="functions.summary",
        category="functions",
        summary="fn",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="functions_summary",
        output_model_name="_DomainModel",
        backend_method="get_fn",
        data_source=DataSourceType.VIEW,
        source_name=None,
        repository_method=None,
        required_datasets=(),
        required_graphs=(),
        exposed_datasets=(),
        supports_pagination=False,
        default_limit=None,
        max_limit=None,
    )
    backend = _Backend()
    typed_backend = cast("QueryBackendOrService", backend)
    mcp = RecordingMcp()

    def _resolve_model(name: str) -> type[_DomainModel] | None:
        if name == "_DomainModel":
            return _DomainModel
        return None

    register_function_tools(
        mcp,
        typed_backend,
        config=None,
        options=FunctionToolOptions(
            operations=(spec,),
            model_resolver=_resolve_model,
        ),
    )
    expect_equal([reg.name for reg in mcp.registrations.calls], ["functions_summary"])
    tool_func = mcp.registry["functions_summary"]
    result = tool_func(payload="hello")
    expect_equal(
        result,
        {"id": "fn-alpha", "payload": "hello", "summary": "helper fn", "notes": None},
    )


def test_serialize_payload_validator_path() -> None:
    """Validator-only model classes should serialize payloads."""
    spec = Operation(
        id="functions.validate",
        category="functions",
        summary="fn",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="functions_validate",
        output_model_name="_ValidatingModel",
        backend_method="validate_fn",
        data_source=DataSourceType.VIEW,
        source_name=None,
        repository_method=None,
        required_datasets=(),
        required_graphs=(),
        exposed_datasets=(),
        supports_pagination=False,
        default_limit=None,
        max_limit=None,
    )
    backend = _Backend()
    mcp = RecordingMcp()

    def _resolve_model(name: str) -> type[_ValidatingModel] | None:
        if name == "_ValidatingModel":
            return _ValidatingModel
        return None

    register_function_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        config=None,
        options=FunctionToolOptions(
            operations=(spec,),
            model_resolver=_resolve_model,
        ),
    )
    result = mcp.registry["functions_validate"]()
    expect_equal(result, {"value": "validated"})


def test_function_tool_sets_scope_and_resets_context() -> None:
    """Graph scope should flow into request context and reset after call."""
    spec = Operation(
        id="graph.scope",
        category="graph",
        summary="scope",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="graph_scope",
        output_model_name="_DomainModel",
        backend_method="get_fn_with_scope",
        data_source=DataSourceType.VIEW,
        source_name=None,
        repository_method=None,
        required_datasets=(),
        required_graphs=(),
        exposed_datasets=(),
        supports_pagination=False,
        default_limit=None,
        max_limit=None,
    )
    backend = _Backend()
    mcp = RecordingMcp()
    register_function_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        config=None,
        options=FunctionToolOptions(operations=(spec,)),
    )
    result = mcp.registry["graph_scope"](scope="abc")
    expect_equal(result, {"value": "abc"})
    expect_equal(backend.captured_scope, "abc")
    expect_true(get_current_request_context() is None)


def test_function_tool_resets_context_on_error(caplog: pytest.LogCaptureFixture) -> None:
    """Request context should reset even when backend raises and logs."""
    spec = Operation(
        id="graph.error",
        category="graph",
        summary="error",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="graph_error",
        output_model_name="_DomainModel",
        backend_method="raise_error",
        data_source=DataSourceType.VIEW,
        source_name=None,
        repository_method=None,
        required_datasets=(),
        required_graphs=(),
        exposed_datasets=(),
        supports_pagination=False,
        default_limit=None,
        max_limit=None,
    )
    backend = _Backend()
    mcp = RecordingMcp()
    register_function_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        config=None,
        options=FunctionToolOptions(operations=(spec,)),
    )
    with pytest.raises(RuntimeError):
        mcp.registry["graph_error"]()
    expect_true(get_current_request_context() is None)
    assert_logged(caplog.records, containing="Function tool failure: fail")


def test_function_tool_from_domain_path() -> None:
    """Payloads without model_dump should serialize via model_cls.from_domain."""
    spec = Operation(
        id="functions.domain",
        category="functions",
        summary="domain",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="functions_domain",
        output_model_name="_DomainModel",
        backend_method="get_domain_from_str",
        data_source=DataSourceType.VIEW,
        source_name=None,
        repository_method=None,
        required_datasets=(),
        required_graphs=(),
        exposed_datasets=(),
        supports_pagination=False,
        default_limit=None,
        max_limit=None,
    )
    backend = _Backend()
    mcp = RecordingMcp()
    register_function_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        config=None,
        options=FunctionToolOptions(
            operations=(spec,),
            model_resolver=lambda name: _DomainModel if name == "_DomainModel" else None,
        ),
    )
    expect_equal(mcp.registry["functions_domain"](), {"value": "payload"})


def test_function_tool_passes_through_raw_dict_when_no_model() -> None:
    """Raw dict payloads should pass through when no model class is found."""
    spec = Operation(
        id="functions.raw",
        category="functions",
        summary="raw",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="functions_raw",
        output_model_name="",
        backend_method="get_raw",
        data_source=DataSourceType.VIEW,
        source_name=None,
        repository_method=None,
        required_datasets=(),
        required_graphs=(),
        exposed_datasets=(),
        supports_pagination=False,
        default_limit=None,
        max_limit=None,
    )
    backend = _Backend()
    mcp = RecordingMcp()
    register_function_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        config=None,
        options=FunctionToolOptions(operations=(spec,)),
    )
    expect_equal(
        mcp.registry["functions_raw"](),
        {"id": "raw-fn", "name": "raw函数", "metadata": None},
    )


def test_function_tool_invokes_auto_pipeline() -> None:
    """Auto-pipeline hook should trigger when env enabled and gateway present."""
    spec = Operation(
        id="functions.summary",
        category="functions",
        summary="fn",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="functions_summary",
        output_model_name="_DomainModel",
        backend_method="get_fn",
        data_source=DataSourceType.VIEW,
        source_name=None,
        repository_method=None,
        required_datasets=(),
        required_graphs=(),
        exposed_datasets=(),
        supports_pagination=False,
        default_limit=None,
        max_limit=None,
    )
    calls: list[str] = []

    def _ensure_prereqs_for_mcp(op_id: str, config: object, backend: object) -> None:
        _ = config
        _ = backend
        calls.append(op_id)

    backend = _Backend()
    mcp = RecordingMcp()
    previous = os.environ.get("CODEINTEL_AUTO_PIPELINE")
    os.environ["CODEINTEL_AUTO_PIPELINE"] = "1"
    try:
        register_function_tools(
            mcp,
            cast("QueryBackendOrService", backend),
            config=ServingConfig(mode="local_db"),
            options=FunctionToolOptions(
                operations=(spec,),
                model_resolver=lambda name: _DomainModel if name == "_DomainModel" else None,
                prereq_runner=_ensure_prereqs_for_mcp,
            ),
        )
        mcp.registry["functions_summary"]()
    finally:
        if previous is None:
            os.environ.pop("CODEINTEL_AUTO_PIPELINE", None)
        else:
            os.environ["CODEINTEL_AUTO_PIPELINE"] = previous
    expect_equal(calls, ["functions.summary"])
