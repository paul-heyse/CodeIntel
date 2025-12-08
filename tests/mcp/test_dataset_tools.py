"""Tests for dataset MCP tool wrappers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.serving.context import get_current_request_context
from codeintel.serving.mcp import dataset_tools, models
from codeintel.serving.mcp.dataset_tools import register_dataset_tools
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from codeintel.serving.operations.catalog import DataSourceType, Operation
from tests._helpers.assertions import (
    expect_equal,
    expect_is_instance,
    expect_true,
)
from tests._helpers.mcp import RecordingMcp

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from codeintel.config.serving_models import ServingConfig


@dataclass
class _Dumpable:
    value: str

    def model_dump(self) -> dict[str, object]:
        return {"value": self.value}


@dataclass
class _FromDomainModel:
    value: str

    @classmethod
    def from_domain(cls, payload: object) -> _FromDomainModel:
        return cls(value=str(payload))

    def model_dump(self) -> dict[str, object]:
        return {"value": self.value}


@dataclass
class _ValidatingModel:
    value: str

    @classmethod
    def model_validate(cls, payload: object) -> _ValidatingModel:
        return cls(value=str(payload))

    def model_dump(self) -> dict[str, object]:
        return {"value": self.value}


class _Backend:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.gateway = SimpleNamespace()

    def list_dataset(self, **_: object) -> list[str]:
        self.calls.append("list_dataset")
        return ["one", "two"]

    def list_models(self, **_: object) -> list[_Dumpable]:
        self.calls.append("list_models")
        return [_Dumpable("a"), _Dumpable("b")]

    def list_raw(self, **_: object) -> list[dict[str, object]]:
        self.calls.append("list_raw")
        return [{"id": "d1"}, {"id": "d2"}]

    def validate_one(self, **_: object) -> str:
        self.calls.append("validate_one")
        return "hello"

    def raw_payload(self, **_: object) -> dict[str, object]:
        self.calls.append("raw_payload")
        return {"id": "raw"}

    def raise_error(self, **_: object) -> list[str]:
        self.calls.append("raise_error")
        message = "boom"
        raise RuntimeError(message)


def test_register_dataset_tools_registers_and_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dataset tool registration should wrap backend methods and serialize lists."""
    spec = Operation(
        id="datasets.list",
        category="datasets",
        summary="list datasets",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="datasets_list",
        output_model_name="_FromDomainModel",
        backend_method="list_dataset",
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
    monkeypatch.setattr(models, "_FromDomainModel", _FromDomainModel, raising=False)
    monkeypatch.setattr(
        "codeintel.serving.mcp.dataset_tools.iter_operations",
        lambda: (spec,),
    )
    backend = _Backend()
    typed_backend = cast("QueryBackendOrService", backend)
    mcp = RecordingMcp()
    register_dataset_tools(cast("FastMCP", mcp), typed_backend, config=None)
    expect_equal([reg.name for reg in mcp.registrations.calls], ["datasets_list"])
    # Execute registered tool
    tool_func = cast("Callable[[], list[dict[str, object]]]", mcp.registry["datasets_list"])
    result = tool_func()
    expect_is_instance(result, list)
    expect_equal(result[0], {"value": "one"})


def test_serialize_list_payload_prefers_model_dump_when_no_model_cls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no model class is provided, model_dump objects should be serialized directly."""
    spec = Operation(
        id="datasets.models",
        category="datasets",
        summary="models",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="datasets_models",
        output_model_name="",
        backend_method="list_models",
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
    monkeypatch.setattr(dataset_tools, "iter_operations", lambda: (spec,))
    register_dataset_tools(
        cast("FastMCP", mcp), cast("QueryBackendOrService", backend), config=None
    )
    result = cast("Callable[[], list[dict[str, object]]]", mcp.registry["datasets_models"])()
    expect_equal(result, [{"value": "a"}, {"value": "b"}])


def test_serialize_payload_uses_validator_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validator-only response models should serialize via model_validate."""
    spec = Operation(
        id="datasets.validate",
        category="datasets",
        summary="validate",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="datasets_validate",
        output_model_name="_ValidatingModel",
        backend_method="validate_one",
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
    monkeypatch.setattr(dataset_tools, "iter_operations", lambda: (spec,))
    monkeypatch.setattr(models, "_ValidatingModel", _ValidatingModel, raising=False)
    register_dataset_tools(
        cast("FastMCP", mcp), cast("QueryBackendOrService", backend), config=None
    )
    result = cast("Callable[[], dict[str, object]]", mcp.registry["datasets_validate"])()
    expect_equal(result, {"value": "hello"})


def test_dataset_tool_resets_context_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Request context must reset even when backend raises."""
    spec = Operation(
        id="datasets.error",
        category="datasets",
        summary="error",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="datasets_error",
        output_model_name="_FromDomainModel",
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
    monkeypatch.setattr(models, "_FromDomainModel", _FromDomainModel, raising=False)
    monkeypatch.setattr(dataset_tools, "iter_operations", lambda: (spec,))
    register_dataset_tools(
        cast("FastMCP", mcp), cast("QueryBackendOrService", backend), config=None
    )
    with pytest.raises(RuntimeError):
        mcp.registry["datasets_error"](dataset_name="d1")
    expect_true(get_current_request_context() is None)


def test_dataset_tool_invokes_auto_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-pipeline hook should be invoked when enabled and gateway present."""
    spec = Operation(
        id="datasets.list",
        category="datasets",
        summary="list datasets",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="datasets_list",
        output_model_name="_FromDomainModel",
        backend_method="list_dataset",
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

    monkeypatch.setattr(
        "codeintel.serving.mcp.dataset_tools.ensure_prereqs_for_mcp",
        _ensure_prereqs_for_mcp,
    )
    backend = cast("QueryBackendOrService", _Backend())
    mcp = RecordingMcp()
    monkeypatch.setattr(dataset_tools, "iter_operations", lambda: (spec,))
    register_dataset_tools(
        cast("FastMCP", mcp),
        backend,
        config=cast("ServingConfig", SimpleNamespace(mode="local_db")),
    )
    cast("Callable[[], object]", mcp.registry["datasets_list"])()
    expect_equal(calls, ["datasets.list"])
    monkeypatch.delenv("CODEINTEL_AUTO_PIPELINE", raising=False)


def test_serialize_payload_passes_through_raw_dict_when_no_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no model class exists, raw dict payloads should return unchanged."""
    spec = Operation(
        id="datasets.raw",
        category="datasets",
        summary="raw",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="datasets_raw",
        output_model_name="",
        backend_method="raw_payload",
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
    monkeypatch.setattr(dataset_tools, "iter_operations", lambda: (spec,))
    register_dataset_tools(
        cast("FastMCP", mcp),
        cast("QueryBackendOrService", backend),
        config=None,
    )
    expect_equal(
        cast("Callable[[], dict[str, object]]", mcp.registry["datasets_raw"])(), {"id": "raw"}
    )


def test_serialize_list_payload_passes_through_raw_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    """List serialization should keep raw dict items when no model provided."""
    spec = Operation(
        id="datasets.raw_list",
        category="datasets",
        summary="raw list",
        description=None,
        http_method=None,
        http_path=None,
        tool_name="datasets_raw_list",
        output_model_name="",
        backend_method="list_raw",
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
    monkeypatch.setattr(dataset_tools, "iter_operations", lambda: (spec,))
    register_dataset_tools(
        cast("FastMCP", mcp),
        cast("QueryBackendOrService", backend),
        config=None,
    )
    expect_equal(
        cast("Callable[[], list[dict[str, object]]]", mcp.registry["datasets_raw_list"])(),
        [{"id": "d1"}, {"id": "d2"}],
    )
