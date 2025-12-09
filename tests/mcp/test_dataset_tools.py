"""Tests for dataset MCP tool wrappers."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.context import get_current_request_context
from codeintel.serving.mcp.dataset_tools import DatasetToolOptions, register_dataset_tools
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from codeintel.serving.operations.catalog import DataSourceType, Operation
from tests._helpers.assertions import (
    expect_equal,
    expect_is_instance,
    expect_true,
)
from tests._helpers.gateway import GatewayFactory
from tests._helpers.mcp_registrar import RecordingMcpRegistrar as RecordingMcp


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
        self.gateway = GatewayFactory().with_macros().open()
        self.limits = BackendLimits()
        self.repo = self.gateway.config.repo or "demo/repo"
        self.commit = self.gateway.config.commit or "deadbeef"

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


def test_register_dataset_tools_registers_and_executes() -> None:
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
    backend = _Backend()
    typed_backend = cast("QueryBackendOrService", backend)
    mcp = RecordingMcp()

    def _resolve_model(name: str) -> type[_FromDomainModel] | None:
        if name == "_FromDomainModel":
            return _FromDomainModel
        return None

    register_dataset_tools(
        mcp,
        typed_backend,
        config=None,
        options=DatasetToolOptions(
            operations=(spec,),
            model_resolver=_resolve_model,
        ),
    )
    expect_equal([reg.name for reg in mcp.registrations.calls], ["datasets_list"])
    # Execute registered tool
    tool_func = cast("Callable[[], list[dict[str, object]]]", mcp.registry["datasets_list"])
    result = tool_func()
    expect_is_instance(result, list)
    expect_equal(result[0], {"value": "one"})


def test_serialize_list_payload_prefers_model_dump_when_no_model_cls() -> None:
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
    register_dataset_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        config=None,
        options=DatasetToolOptions(operations=(spec,)),
    )
    result = cast("Callable[[], list[dict[str, object]]]", mcp.registry["datasets_models"])()
    expect_equal(result, [{"value": "a"}, {"value": "b"}])


def test_serialize_payload_uses_validator_fallback() -> None:
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

    def _resolve_model(name: str) -> type[_ValidatingModel] | None:
        if name == "_ValidatingModel":
            return _ValidatingModel
        return None

    register_dataset_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        config=None,
        options=DatasetToolOptions(
            operations=(spec,),
            model_resolver=_resolve_model,
        ),
    )
    result = cast("Callable[[], dict[str, object]]", mcp.registry["datasets_validate"])()
    expect_equal(result, {"value": "hello"})


def test_dataset_tool_resets_context_on_error() -> None:
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

    def _resolve_model(name: str) -> type[_FromDomainModel] | None:
        if name == "_FromDomainModel":
            return _FromDomainModel
        return None

    register_dataset_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        config=None,
        options=DatasetToolOptions(
            operations=(spec,),
            model_resolver=_resolve_model,
        ),
    )
    with pytest.raises(RuntimeError):
        mcp.registry["datasets_error"](dataset_name="d1")
    expect_true(get_current_request_context() is None)


def test_dataset_tool_invokes_auto_pipeline() -> None:
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
    calls: list[str] = []

    def _ensure_prereqs_for_mcp(op_id: str, config: object, backend: object) -> None:
        _ = config
        _ = backend
        calls.append(op_id)

    backend = cast("QueryBackendOrService", _Backend())
    mcp = RecordingMcp()
    previous = os.environ.get("CODEINTEL_AUTO_PIPELINE")
    os.environ["CODEINTEL_AUTO_PIPELINE"] = "1"
    try:
        register_dataset_tools(
            mcp,
            backend,
            config=ServingConfig(mode="local_db"),
            options=DatasetToolOptions(
                operations=(spec,),
                model_resolver=lambda name: _FromDomainModel
                if name == "_FromDomainModel"
                else None,
                prereq_runner=_ensure_prereqs_for_mcp,
            ),
        )
        cast("Callable[[], object]", mcp.registry["datasets_list"])()
    finally:
        if previous is None:
            os.environ.pop("CODEINTEL_AUTO_PIPELINE", None)
        else:
            os.environ["CODEINTEL_AUTO_PIPELINE"] = previous
    expect_equal(calls, ["datasets.list"])


def test_serialize_payload_passes_through_raw_dict_when_no_model() -> None:
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
    register_dataset_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        config=None,
        options=DatasetToolOptions(operations=(spec,)),
    )
    expect_equal(
        cast("Callable[[], dict[str, object]]", mcp.registry["datasets_raw"])(), {"id": "raw"}
    )


def test_serialize_list_payload_passes_through_raw_dicts() -> None:
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
    register_dataset_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        config=None,
        options=DatasetToolOptions(operations=(spec,)),
    )
    expect_equal(
        cast("Callable[[], list[dict[str, object]]]", mcp.registry["datasets_raw_list"])(),
        [{"id": "d1"}, {"id": "d2"}],
    )
