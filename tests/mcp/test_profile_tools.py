"""Tests for profile MCP tools."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.serving.mcp import errors, profile_tools
from tests._helpers.assertions import expect_equal, expect_in
from tests._helpers.mcp import RecordingMcp

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from codeintel.config.serving_models import ServingConfig
    from codeintel.serving.mcp.tool_utils import QueryBackendOrService
else:
    FastMCP = QueryBackendOrService = ServingConfig = object


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
        self.gateway = SimpleNamespace()
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


def _patch_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_tools, "FunctionProfileResponse", _ProfileModel, raising=False)
    monkeypatch.setattr(profile_tools, "FileProfileResponse", _ProfileModel, raising=False)
    monkeypatch.setattr(profile_tools, "ModuleProfileResponse", _ProfileModel, raising=False)


def test_register_profile_tools_registers_and_serializes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Profile tools should serialize via from_domain/model_validate and support auto-pipeline."""
    _patch_models(monkeypatch)
    backend = _Backend()
    mcp = RecordingMcp()
    # Enable auto-pipeline path
    monkeypatch.setenv("CODEINTEL_AUTO_PIPELINE", "1")
    calls: list[str] = []

    def _ensure_prereqs_for_mcp(*, op_id: str, config: object, backend: object) -> None:
        _ = config
        _ = backend
        calls.append(op_id)

    monkeypatch.setattr(
        profile_tools, "ensure_prereqs_for_mcp", _ensure_prereqs_for_mcp, raising=False
    )

    profile_tools.register_profile_tools(
        cast("FastMCP", mcp),
        cast("QueryBackendOrService", backend),
        config=cast("ServingConfig", SimpleNamespace(mode="local_db")),
    )

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
    monkeypatch.delenv("CODEINTEL_AUTO_PIPELINE", raising=False)


def test_profile_tools_wrap_mcp_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backend McpError should serialize to error payload."""
    _patch_models(monkeypatch)
    backend = _ExplodingBackend()
    mcp = RecordingMcp()
    profile_tools.register_profile_tools(
        cast("FastMCP", mcp),
        cast("QueryBackendOrService", backend),
        config=None,
    )
    result = cast("Callable[..., dict[str, object]]", mcp.registry["get_function_profile"])(
        goid_h128=1
    )
    expect_in("error", result)


def test_profile_tools_skip_auto_pipeline_without_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-pipeline should not run when backend lacks gateway attribute."""
    _patch_models(monkeypatch)
    monkeypatch.setenv("CODEINTEL_AUTO_PIPELINE", "1")
    backend = SimpleNamespace(
        get_file_profile=lambda rel_path: {"value": rel_path},
        get_function_profile=lambda goid_h128: {"value": goid_h128},
        get_module_profile=lambda module: {"value": module},
    )
    calls: list[str] = []

    def _record(*, op_id: str, config: object, backend: object) -> None:
        _ = config
        _ = backend
        calls.append(op_id)

    monkeypatch.setattr(profile_tools, "ensure_prereqs_for_mcp", _record, raising=False)
    mcp = RecordingMcp()
    profile_tools.register_profile_tools(
        cast("FastMCP", mcp),
        cast("QueryBackendOrService", backend),
        config=cast("ServingConfig", SimpleNamespace(mode="local_db")),
    )
    cast("Callable[..., dict[str, object]]", mcp.registry["get_file_profile"])(rel_path="a.py")
    expect_equal(calls, [])
    monkeypatch.delenv("CODEINTEL_AUTO_PIPELINE", raising=False)
