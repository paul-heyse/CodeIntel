"""Tests for MCP auto-pipeline wrapping."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from codeintel.serving.mcp import auto_pipeline_wrapper
from tests._helpers.assertions import expect_equal

if TYPE_CHECKING:
    from codeintel.config.serving_models import ServingConfig
    from codeintel.serving.mcp.backend import QueryBackend


def test_wrap_tool_with_prereqs_invokes_prereq_then_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrapper should run ensure_prereqs_for_mcp before calling tool."""
    calls: list[str] = []

    def _ensure_prereqs_for_mcp(*, op_id: str, config: object, backend: object) -> None:
        calls.append(f"prereq:{op_id}:{config}:{backend}")

    monkeypatch.setattr(auto_pipeline_wrapper, "ensure_prereqs_for_mcp", _ensure_prereqs_for_mcp)

    def _tool(**kwargs: object) -> dict[str, object]:
        calls.append(f"tool:{kwargs.get('x')}")
        return {"ok": kwargs.get("x")}

    wrapped = auto_pipeline_wrapper.wrap_tool_with_prereqs(
        _tool,
        op_id="op.one",
        config=cast("ServingConfig", "cfg"),
        backend=cast("QueryBackend", "backend"),
    )

    result = wrapped(x=1)

    expect_equal(calls, ["prereq:op.one:cfg:backend", "tool:1"])
    expect_equal(result, {"ok": 1})
