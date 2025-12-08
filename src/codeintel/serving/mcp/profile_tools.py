"""Profile-oriented MCP tools.

Note: This module is maintained for backward compatibility. The core tool
building logic has been consolidated in ``tool_builder.py``. New code should
use ``register_tools_for_category`` from ``tools_base.py`` or the unified
``register_tools`` function.

See Also
--------
- ``codeintel.serving.mcp.tool_builder`` : Unified tool building
- ``codeintel.serving.mcp.tools_base`` : Top-level registration
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

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
from codeintel.serving.mcp.tool_builder import McpToolRegistrar
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.operations import Operation, get_operation

if TYPE_CHECKING:
    from codeintel.config.serving_models import ServingConfig
    from codeintel.serving.mcp.backend import QueryBackend


def _require_spec(
    op_id: str, expected_tool: str, operations: Iterable[Operation] | None
) -> Operation:
    if operations is not None:
        for spec in operations:
            if spec.id == op_id:
                if spec.tool_name != expected_tool:
                    message = f"Operation {op_id} has mismatched tool name"
                    raise ValueError(message)
                return spec
    spec = get_operation(op_id)
    if spec is None or spec.tool_name != expected_tool:
        message = f"Operation {op_id} has mismatched tool name"
        raise ValueError(message)
    return spec


@dataclass
class ProfileToolOptions:
    """Optional overrides for profile tool registration."""

    operations: Iterable[Operation] | None = None
    model_resolver: Callable[[str], ResponseFactory | None] | None = None
    prereq_runner: Callable[[str, ServingConfig, QueryBackend], object] | None = None


def register_profile_tools(
    mcp: McpToolRegistrar,
    backend: QueryBackendOrService,
    config: ServingConfig | None = None,
    options: ProfileToolOptions | None = None,
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
    options
        Optional overrides for operations, model resolution, and prereq runner.
    """
    opts = options or ProfileToolOptions()
    resolve_model = opts.model_resolver or (
        lambda name: getattr(  # type: ignore[misc]
            __import__("codeintel.serving.mcp.models", fromlist=[name]),
            name,
            None,
        )
    )
    run_prereqs = opts.prereq_runner or (
        lambda op_id, cfg, bkd: ensure_prereqs_for_mcp(op_id=op_id, config=cfg, backend=bkd)
    )
    # Check auto-pipeline conditions once
    auto_pipeline = (
        is_auto_pipeline_enabled() and config is not None and hasattr(backend, "gateway")
    )
    spec_function = _require_spec("profiles.function", "get_function_profile", opts.operations)
    spec_file = _require_spec("profiles.file", "get_file_profile", opts.operations)
    spec_module = _require_spec("profiles.module", "get_module_profile", opts.operations)

    fn_model_cls = cast("ResponseFactory | None", resolve_model(spec_function.output_model_name))
    file_model_cls = cast("ResponseFactory | None", resolve_model(spec_file.output_model_name))
    module_model_cls = cast("ResponseFactory | None", resolve_model(spec_module.output_model_name))

    @mcp.tool(
        name=spec_function.tool_name,
        description=spec_function.summary,
    )
    @_wrap
    def get_function_profile(goid_h128: int) -> dict[str, object] | dict[str, ProblemDetail]:
        if auto_pipeline and config is not None:
            run_prereqs("profiles.function", config, cast("QueryBackend", backend))
        result = backend.get_function_profile(goid_h128=goid_h128)
        response = _coerce_response(
            result,
            fn_model_cls or FunctionProfileResponse,
        )
        return response.model_dump()

    @mcp.tool(
        name=spec_file.tool_name,
        description=spec_file.summary,
    )
    @_wrap
    def get_file_profile(rel_path: str) -> dict[str, object] | dict[str, ProblemDetail]:
        if auto_pipeline and config is not None:
            run_prereqs("profiles.file", config, cast("QueryBackend", backend))
        result = backend.get_file_profile(rel_path=rel_path)
        response = _coerce_response(
            result,
            file_model_cls or FileProfileResponse,
        )
        return response.model_dump()

    @mcp.tool(
        name=spec_module.tool_name,
        description=spec_module.summary,
    )
    @_wrap
    def get_module_profile(module: str) -> dict[str, object] | dict[str, ProblemDetail]:
        if auto_pipeline and config is not None:
            run_prereqs("profiles.module", config, cast("QueryBackend", backend))
        result = backend.get_module_profile(module=module)
        response = _coerce_response(
            result,
            module_model_cls or ModuleProfileResponse,
        )
        return response.model_dump()


def _coerce_response(
    payload: object,
    model_cls: ResponseFactory,
) -> SupportsModelDump:
    """Coerce a backend response into a response model.

    Returns
    -------
    SupportsModelDump
        Response model instance with ``model_dump`` available.
    """
    if hasattr(model_cls, "from_domain"):
        return cast("SupportsFromDomain", model_cls).from_domain(payload)
    return cast("SupportsModelValidate", model_cls).model_validate(payload)


__all__ = ["ProfileToolOptions", "register_profile_tools"]
