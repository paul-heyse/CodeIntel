"""Coverage-focused tests for MCP tool builder helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.context import get_current_request_context
from codeintel.serving.mcp.tool_builder import (
    ToolRegistrationOptions,
    build_tool_from_operation,
    register_tools_for_category,
)
from codeintel.serving.operations.catalog import DataSourceType, Operation
from tests._helpers.assertions import expect_equal, expect_in, expect_is_instance, expect_true
from tests._helpers.gateway import GatewayFactory
from tests._helpers.mcp_tools import make_mcp_context

if TYPE_CHECKING:
    from codeintel.serving.mcp.backend import QueryBackend
    from codeintel.serving.mcp.tool_utils import QueryBackendOrService


@dataclass
class _ModelFromDomain:
    value: str

    @classmethod
    def from_domain(cls, payload: object) -> _ModelFromDomain:
        return cls(value=str(payload))

    def model_dump(self) -> dict[str, object]:
        return {"value": self.value}


class _Backend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.repo = "demo/repo"
        self.commit = "deadbeef"
        self.gateway = GatewayFactory().with_macros().open()
        self.limits = BackendLimits()

    def do_echo(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("do_echo", dict(kwargs)))
        return {"ok": True, **kwargs}

    def do_model(self, **kwargs: object) -> _ModelFromDomain:
        self.calls.append(("do_model", dict(kwargs)))
        return _ModelFromDomain.from_domain(kwargs.get("payload", "p"))


def _make_operation(
    op_id: str,
    backend_method: str,
    *,
    output_model_name: str = "",
    data_source: DataSourceType = DataSourceType.VIEW,
    category: str = "functions",
) -> Operation:
    return Operation(
        id=op_id,
        category=category,
        summary="test op",
        description=None,
        http_method=None,
        http_path=None,
        tool_name=f"tool_{op_id}",
        output_model_name=output_model_name,
        backend_method=backend_method,
        data_source=data_source,
        source_name=None,
        repository_method=None,
        required_datasets=(),
        required_graphs=(),
        exposed_datasets=(),
        supports_pagination=False,
        default_limit=None,
        max_limit=None,
    )


def test_build_tool_from_operation_model_serialization() -> None:
    """Tool builder should handle from_domain/model_dump paths."""
    backend = _Backend()
    typed_backend = cast("QueryBackendOrService", backend)
    spec = _make_operation("echo", "do_model", output_model_name="_ModelFromDomain")

    def _resolve_model(name: str) -> type[_ModelFromDomain] | None:
        if name == "_ModelFromDomain":
            return _ModelFromDomain
        return None

    tool = build_tool_from_operation(
        spec,
        typed_backend,
        config=None,
        model_resolver=_resolve_model,
    )
    response = tool(payload="hello", extra=1)
    expect_equal(response, {"value": "hello"})

    ctx = get_current_request_context()
    expect_true(ctx is None)


def test_build_tool_from_operation_missing_backend_method() -> None:
    """Missing backend method should raise TypeError."""
    backend = cast("QueryBackendOrService", _Backend())
    spec = _make_operation("missing", "nope")
    with pytest.raises(TypeError):
        build_tool_from_operation(spec, backend, config=None)


def test_register_tools_for_category_registers_expected_tools() -> None:
    """Only matching categories should be registered."""
    backend = cast("QueryBackendOrService", _Backend())
    specs = [
        _make_operation("fn.one", "do_echo"),
        _make_operation("datasets.list", "do_echo", category="datasets"),
    ]
    ctx = make_mcp_context(backend=backend, operations=specs)
    register_tools_for_category(
        ctx.mcp,
        ctx.backend,
        categories={"functions"},
        options=ToolRegistrationOptions(operations=ctx.operations),
    )
    names = {reg.name for reg in ctx.mcp.registrations.calls}
    expect_equal(names, {"tool_fn.one"})
    expect_true("tool_datasets.list" not in ctx.mcp.registry)

    registered_callable = ctx.mcp.registry["tool_fn.one"]
    result = registered_callable(message="hi")
    expect_is_instance(result, dict)


def test_build_tool_auto_pipeline_invokes_prereqs() -> None:
    """Auto-pipeline branch should trigger ensure_prereqs_for_mcp when enabled."""
    backend = _Backend()
    typed_backend = cast("QueryBackendOrService", backend)
    spec = _make_operation("echo", "do_echo")

    calls: list[tuple[str, str]] = []

    def _record_prereqs(op_id: str, cfg: ServingConfig, bkd: QueryBackend) -> None:
        _ = cfg
        calls.append((op_id, cast("_Backend", bkd).repo))

    previous = os.environ.get("CODEINTEL_AUTO_PIPELINE")
    os.environ["CODEINTEL_AUTO_PIPELINE"] = "1"
    try:
        config = ServingConfig(repo="demo", commit="c")
        tool = build_tool_from_operation(
            spec,
            typed_backend,
            config=config,
            prereq_runner=_record_prereqs,
        )
        _ = tool()
    finally:
        if previous is None:
            os.environ.pop("CODEINTEL_AUTO_PIPELINE", None)
        else:
            os.environ["CODEINTEL_AUTO_PIPELINE"] = previous

    expect_in(("echo", "demo/repo"), calls)
