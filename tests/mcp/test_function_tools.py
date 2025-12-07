"""Tests for function MCP tool wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.serving.context import get_current_request_context
from codeintel.serving.mcp import function_tools, models
from codeintel.serving.mcp.function_tools import register_function_tools
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from codeintel.serving.operations.catalog import DataSourceType, Operation
from tests._helpers.mcp import RecordingMcp

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from codeintel.config.serving_models import ServingConfig


@dataclass
class _DomainModel:
    value: str

    def model_dump(self) -> dict[str, object]:
        return {"value": self.value}

    @classmethod
    def from_domain(cls, payload: object) -> _DomainModel:
        return cls(value=str(payload))


class _Backend:
    def __init__(self) -> None:
        self.gateway = SimpleNamespace()
        self.captured_scope: str | None = None

    def get_fn(self, *, payload: object = "x") -> _DomainModel:
        return _DomainModel.from_domain(payload)

    def get_fn_with_scope(self, *, scope: str | None = None) -> _DomainModel:
        ctx = get_current_request_context()
        if ctx is not None:
            self.captured_scope = ctx.graph_scope
        return _DomainModel.from_domain(scope or "none")

    def get_raw(self, **_: object) -> dict[str, object]:
        return {"id": "raw-fn"}

    def get_domain_from_str(self, **_: object) -> str:
        return "payload"

    def raise_error(self, **_: object) -> _DomainModel:
        message = "fail"
        raise RuntimeError(message)

    def validate_fn(self, **_: object) -> str:
        return "validated"


class _ValidatingModel:
    value: str

    @classmethod
    def model_validate(cls, payload: object) -> _DomainModel:
        return _DomainModel.from_domain(payload)


def test_register_function_tools_registers_and_executes(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(models, "_DomainModel", _DomainModel, raising=False)
    monkeypatch.setattr(
        "codeintel.serving.mcp.function_tools.iter_operations",
        lambda: (spec,),
    )
    backend = _Backend()
    typed_backend = cast("QueryBackendOrService", backend)
    mcp = RecordingMcp()
    register_function_tools(cast("FastMCP", mcp), typed_backend, config=None)
    assert [reg.name for reg in mcp.registrations.calls] == ["functions_summary"]
    tool_func = mcp.registry["functions_summary"]
    result = tool_func(payload="hello")
    assert result == {"value": "hello"}


def test_serialize_payload_validator_path(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(function_tools, "iter_operations", lambda: (spec,))
    monkeypatch.setattr(models, "_ValidatingModel", _ValidatingModel, raising=False)
    register_function_tools(
        cast("FastMCP", mcp), cast("QueryBackendOrService", backend), config=None
    )
    result = mcp.registry["functions_validate"]()
    assert result == {"value": "validated"}


def test_function_tool_sets_scope_and_resets_context(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(function_tools, "iter_operations", lambda: (spec,))
    register_function_tools(
        cast("FastMCP", mcp), cast("QueryBackendOrService", backend), config=None
    )
    result = mcp.registry["graph_scope"](scope="abc")
    assert result == {"value": "abc"}
    assert backend.captured_scope == "abc"
    assert get_current_request_context() is None


def test_function_tool_resets_context_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Request context should reset even when backend raises."""
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
    monkeypatch.setattr(function_tools, "iter_operations", lambda: (spec,))
    register_function_tools(
        cast("FastMCP", mcp), cast("QueryBackendOrService", backend), config=None
    )
    with pytest.raises(RuntimeError):
        mcp.registry["graph_error"]()
    assert get_current_request_context() is None


def test_function_tool_from_domain_path(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(function_tools, "iter_operations", lambda: (spec,))
    monkeypatch.setattr(models, "_DomainModel", _DomainModel, raising=False)
    register_function_tools(
        cast("FastMCP", mcp), cast("QueryBackendOrService", backend), config=None
    )
    assert mcp.registry["functions_domain"]() == {"value": "payload"}


def test_function_tool_passes_through_raw_dict_when_no_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(function_tools, "iter_operations", lambda: (spec,))
    register_function_tools(
        cast("FastMCP", mcp), cast("QueryBackendOrService", backend), config=None
    )
    assert mcp.registry["functions_raw"]() == {"id": "raw-fn"}


def test_function_tool_invokes_auto_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setenv("CODEINTEL_AUTO_PIPELINE", "1")
    calls: list[str] = []

    def _ensure_prereqs_for_mcp(*, op_id: str, config: object, backend: object) -> None:
        _ = config
        _ = backend
        calls.append(op_id)

    backend = _Backend()
    mcp = RecordingMcp()
    monkeypatch.setattr(function_tools, "iter_operations", lambda: (spec,))
    monkeypatch.setattr(models, "_DomainModel", _DomainModel, raising=False)
    monkeypatch.setattr(
        "codeintel.serving.mcp.function_tools.ensure_prereqs_for_mcp",
        _ensure_prereqs_for_mcp,
    )
    register_function_tools(
        cast("FastMCP", mcp),
        cast("QueryBackendOrService", backend),
        config=cast("ServingConfig", SimpleNamespace(mode="local_db")),
    )
    mcp.registry["functions_summary"]()
    assert calls == ["functions.summary"]
    monkeypatch.delenv("CODEINTEL_AUTO_PIPELINE", raising=False)
