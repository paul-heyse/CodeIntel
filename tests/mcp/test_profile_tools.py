"""Tests for profile MCP tools."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from codeintel.serving.mcp import errors
from codeintel.serving.mcp.profile_tools import ProfileToolOptions, register_profile_tools
from codeintel.serving.operations.catalog import DataSourceType, Operation
from tests._helpers.assertions import expect_equal, expect_in
from tests._helpers.mcp_registrar import RecordingMcpRegistrar as RecordingMcp

if TYPE_CHECKING:
    from codeintel.config.serving_models import ServingConfig
    from codeintel.serving.mcp.tool_utils import QueryBackendOrService


class _BackendGateway:
    """Minimal gateway stub to satisfy profile backend usage."""

    @staticmethod
    def close() -> None:
        return None


@dataclass(frozen=True)
class _ServingConfigStub:
    """Minimal ServingConfig stub for tests."""

    mode: str


class _ProfileModel:
    def __init__(self, value: str) -> None:
        self.value = value

    @classmethod
    def from_domain(cls, payload: object) -> _ProfileModel:
        return cls(str(payload))

    @classmethod
    def model_validate(cls, payload: object) -> _ProfileModel:
        return cls(str(payload))

    def model_dump(self) -> dict[str, object]:
        return {"value": self.value}


class _Backend:
    def __init__(self) -> None:
        self.gateway = _BackendGateway()
        self.calls: list[str] = []

    def get_function_profile(self, *, goid_h128: int) -> str:
        self.calls.append(f"fn:{goid_h128}")
        return f"fn-{goid_h128}"

    def get_file_profile(self, *, rel_path: str) -> str:
        self.calls.append(f"file:{rel_path}")
        return f"file-{rel_path}"

    def get_module_profile(self, *, module: str) -> str:
        self.calls.append(f"mod:{module}")
        return f"mod-{module}"


class _ExplodingBackend(_Backend):
    """Backend variant that raises backend failure for function profile."""

    def get_function_profile(self, *, goid_h128: int) -> str:
        _ = (self, goid_h128)
        message = "fail"
        raise errors.backend_failure(message)


class _BackendWithoutGateway:
    """Backend variant without a gateway attribute to test auto-pipeline guard."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_function_profile(self, *, goid_h128: int) -> dict[str, object]:
        self.calls.append(f"fn:{goid_h128}")
        return {"value": f"fn-{goid_h128}"}

    def get_file_profile(self, *, rel_path: str) -> dict[str, object]:
        self.calls.append(f"file:{rel_path}")
        return {"value": f"file-{rel_path}"}

    def get_module_profile(self, *, module: str) -> dict[str, object]:
        self.calls.append(f"mod:{module}")
        return {"value": f"mod-{module}"}


def _profile_operations() -> tuple[Operation, Operation, Operation]:
    return (
        Operation(
            id="profiles.function",
            category="profiles",
            summary="function profile",
            description=None,
            http_method=None,
            http_path=None,
            tool_name="get_function_profile",
            output_model_name="_ProfileModel",
            backend_method="get_function_profile",
            data_source=DataSourceType.VIEW,
            source_name=None,
            repository_method=None,
            required_datasets=(),
            required_graphs=(),
            exposed_datasets=(),
            supports_pagination=False,
            default_limit=None,
            max_limit=None,
        ),
        Operation(
            id="profiles.file",
            category="profiles",
            summary="file profile",
            description=None,
            http_method=None,
            http_path=None,
            tool_name="get_file_profile",
            output_model_name="_ProfileModel",
            backend_method="get_file_profile",
            data_source=DataSourceType.VIEW,
            source_name=None,
            repository_method=None,
            required_datasets=(),
            required_graphs=(),
            exposed_datasets=(),
            supports_pagination=False,
            default_limit=None,
            max_limit=None,
        ),
        Operation(
            id="profiles.module",
            category="profiles",
            summary="module profile",
            description=None,
            http_method=None,
            http_path=None,
            tool_name="get_module_profile",
            output_model_name="_ProfileModel",
            backend_method="get_module_profile",
            data_source=DataSourceType.VIEW,
            source_name=None,
            repository_method=None,
            required_datasets=(),
            required_graphs=(),
            exposed_datasets=(),
            supports_pagination=False,
            default_limit=None,
            max_limit=None,
        ),
    )


def _resolve_profile_model(name: str) -> type[_ProfileModel] | None:
    if name == "_ProfileModel":
        return _ProfileModel
    return None


def test_register_profile_tools_registers_and_serializes() -> None:
    """Profile tools should serialize via from_domain/model_validate and support auto-pipeline."""
    backend = _Backend()
    mcp = RecordingMcp()
    calls: list[str] = []

    def _ensure_prereqs_for_mcp(op_id: str, config: object, backend: object) -> None:
        _ = config
        _ = backend
        calls.append(op_id)

    previous = os.environ.get("CODEINTEL_AUTO_PIPELINE")
    os.environ["CODEINTEL_AUTO_PIPELINE"] = "1"
    try:
        register_profile_tools(
            mcp,
            cast("QueryBackendOrService", backend),
            config=cast("ServingConfig", _ServingConfigStub(mode="local_db")),
            options=ProfileToolOptions(
                operations=_profile_operations(),
                model_resolver=_resolve_profile_model,
                prereq_runner=_ensure_prereqs_for_mcp,
            ),
        )
    finally:
        if previous is None:
            os.environ.pop("CODEINTEL_AUTO_PIPELINE", None)
        else:
            os.environ["CODEINTEL_AUTO_PIPELINE"] = previous

    result_fn = cast("Callable[..., dict[str, object]]", mcp.registry["get_function_profile"])(
        goid_h128=1
    )
    result_file = cast("Callable[..., dict[str, object]]", mcp.registry["get_file_profile"])(
        rel_path="a.py"
    )
    result_mod = cast("Callable[..., dict[str, object]]", mcp.registry["get_module_profile"])(
        module="pkg.mod"
    )

    expect_equal(result_fn, {"value": "fn-1"})
    expect_equal(result_file, {"value": "file-a.py"})
    expect_equal(result_mod, {"value": "mod-pkg.mod"})
    expect_equal(
        set(calls),
        {
            "profiles.function",
            "profiles.file",
            "profiles.module",
        },
    )


def test_profile_tools_wrap_mcp_error() -> None:
    """Backend McpError should serialize to error payload."""
    backend = _ExplodingBackend()
    mcp = RecordingMcp()
    register_profile_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        config=None,
        options=ProfileToolOptions(
            operations=_profile_operations(),
            model_resolver=_resolve_profile_model,
        ),
    )
    result = cast("Callable[..., dict[str, object]]", mcp.registry["get_function_profile"])(
        goid_h128=1
    )
    expect_in("error", result)


def test_profile_tools_skip_auto_pipeline_without_gateway() -> None:
    """Auto-pipeline should not run when backend lacks gateway attribute."""
    previous = os.environ.get("CODEINTEL_AUTO_PIPELINE")
    os.environ["CODEINTEL_AUTO_PIPELINE"] = "1"
    backend = _BackendWithoutGateway()
    calls: list[str] = []

    def _record(op_id: str, config: object, backend: object) -> None:
        _ = config
        _ = backend
        calls.append(op_id)

    mcp = RecordingMcp()
    try:
        register_profile_tools(
            mcp,
            cast("QueryBackendOrService", backend),
            config=cast("ServingConfig", _ServingConfigStub(mode="local_db")),
            options=ProfileToolOptions(
                operations=_profile_operations(),
                model_resolver=_resolve_profile_model,
                prereq_runner=_record,
            ),
        )
    finally:
        if previous is None:
            os.environ.pop("CODEINTEL_AUTO_PIPELINE", None)
        else:
            os.environ["CODEINTEL_AUTO_PIPELINE"] = previous
    cast("Callable[..., dict[str, object]]", mcp.registry["get_file_profile"])(rel_path="a.py")
    expect_equal(calls, [])
