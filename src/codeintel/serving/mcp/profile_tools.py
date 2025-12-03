"""Profile-oriented MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from mcp.server.fastmcp import FastMCP

from codeintel.serving.auto_pipeline import ensure_prereqs_for_mcp, is_auto_pipeline_enabled
from codeintel.serving.mcp.models import (
    FileProfileResponse,
    FunctionProfileResponse,
    ModuleProfileResponse,
    ProblemDetail,
)
from codeintel.serving.mcp.serialization import (
    ResponseFactory,
    SupportsFromDomain,
    SupportsModelDump,
    SupportsModelValidate,
)
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.operations import Operation, get_operation

if TYPE_CHECKING:
    from codeintel.config.serving_models import ServingConfig
    from codeintel.serving.mcp.backend import QueryBackend


def _require_spec(op_id: str, expected_tool: str) -> Operation:
    spec = get_operation(op_id)
    if spec is None or spec.tool_name != expected_tool:
        message = f"Operation {op_id} has mismatched tool name"
        raise ValueError(message)
    return spec


def register_profile_tools(
    mcp: FastMCP,
    backend: QueryBackendOrService,
    config: ServingConfig | None = None,
) -> None:
    """Register profile-oriented MCP tools.

    Parameters
    ----------
    mcp
        FastMCP instance to register tools against.
    backend
        Concrete MCP backend or QueryService implementation.
    config
        Optional serving config for auto-pipeline support.
    """
    # Check auto-pipeline conditions once
    auto_pipeline = (
        is_auto_pipeline_enabled() and config is not None and hasattr(backend, "gateway")
    )
    _ = _require_spec("profiles.function", "get_function_profile")
    _ = _require_spec("profiles.file", "get_file_profile")
    _ = _require_spec("profiles.module", "get_module_profile")

    @mcp.tool()
    @_wrap
    def get_function_profile(goid_h128: int) -> dict[str, object] | dict[str, ProblemDetail]:
        if auto_pipeline and config is not None:
            ensure_prereqs_for_mcp(
                op_id="profiles.function",
                config=config,
                backend=cast("QueryBackend", backend),
            )
        result = backend.get_function_profile(goid_h128=goid_h128)
        response = _coerce_response(
            result,
            FunctionProfileResponse,
        )
        return response.model_dump()

    @mcp.tool()
    @_wrap
    def get_file_profile(rel_path: str) -> dict[str, object] | dict[str, ProblemDetail]:
        if auto_pipeline and config is not None:
            ensure_prereqs_for_mcp(
                op_id="profiles.file",
                config=config,
                backend=cast("QueryBackend", backend),
            )
        result = backend.get_file_profile(rel_path=rel_path)
        response = _coerce_response(
            result,
            FileProfileResponse,
        )
        return response.model_dump()

    @mcp.tool()
    @_wrap
    def get_module_profile(module: str) -> dict[str, object] | dict[str, ProblemDetail]:
        if auto_pipeline and config is not None:
            ensure_prereqs_for_mcp(
                op_id="profiles.module",
                config=config,
                backend=cast("QueryBackend", backend),
            )
        result = backend.get_module_profile(module=module)
        response = _coerce_response(
            result,
            ModuleProfileResponse,
        )
        return response.model_dump()


def _coerce_response(
    payload: object,
    model_cls: ResponseFactory,
) -> SupportsModelDump:
    if hasattr(model_cls, "from_domain"):
        return cast("SupportsFromDomain", model_cls).from_domain(payload)
    return cast("SupportsModelValidate", model_cls).model_validate(payload)


__all__ = ["register_profile_tools"]
