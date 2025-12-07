"""Tests for meta MCP tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypedDict, cast

import pytest

from codeintel.config.datasets.dataflow import DataflowEdge, DataflowNode
from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp import meta_tools
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from codeintel.serving.operations.catalog import DataSourceType, Operation

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


@dataclass
class _OpMeta:
    id: str
    name: str
    table_key: str
    description: str
    schema_version: str
    family: str
    is_docs_view: bool
    is_read_only: bool
    default_limit: int
    max_limit: int
    is_view: bool = True


class _RecordingMcp:
    def __init__(self) -> None:
        self.registry: list[Callable[..., list[dict[str, object]]]] = []

    def tool(
        self, name: str | None = None, description: str | None = None
    ) -> Callable[
        [Callable[..., list[dict[str, object]]]],
        Callable[..., list[dict[str, object]]],
    ]:
        def _decorator(
            func: Callable[..., list[dict[str, object]]],
        ) -> Callable[..., list[dict[str, object]]]:
            _ = name
            _ = description
            self.registry.append(func)
            return func

        return _decorator


class _DataflowPayload(TypedDict):
    nodes: list[dict[str, object]]
    edges: list[dict[str, object]]


class _ExplainPayload(TypedDict):
    node: dict[str, object]
    incoming_edges: list[dict[str, object]]
    outgoing_edges: list[dict[str, object]]


def _make_operation(op_id: str, tool_name: str) -> Operation:
    return Operation(
        id=op_id,
        category="meta",
        summary=op_id,
        description=None,
        http_method=None,
        http_path=None,
        tool_name=tool_name,
        output_model_name="Model",
        backend_method="method",
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


def test_register_meta_tools_registers_expected_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Meta tools should register and return serialized payloads."""
    # Stub dataset metadata and operations
    monkeypatch.setattr(
        meta_tools,
        "build_dataset_meta",
        lambda _service, _limits: (
            SimpleNamespace(
                id="d1",
                name="Dataset One",
                table_key="table1",
                description="desc",
                schema_version="1",
                family="fam",
                is_docs_view=True,
                is_read_only=False,
                default_limit=10,
                max_limit=100,
            ),
        ),
    )
    monkeypatch.setattr(
        meta_tools,
        "iter_registry_operations",
        lambda: (_make_operation("op.one", "tool_one"),),
    )

    nodes = [
        DataflowNode(
            id="table1",
            kind="table",
            family="fam",
            owner_package="analytics",
            description="desc",
        ),
        DataflowNode(
            id="op.one",
            kind="operation",
            family="fam",
            owner_package="analytics",
            description="op",
        ),
    ]
    edges = [DataflowEdge(src="table1", dst="op.one", edge_type="reads")]

    monkeypatch.setattr(
        meta_tools,
        "build_serving_dataflow_graph",
        lambda: (nodes, edges),
    )

    backend = SimpleNamespace(limits=BackendLimits(), service=SimpleNamespace())
    mcp = _RecordingMcp()

    meta_tools.register_meta_tools(cast("FastMCP", mcp), cast("QueryBackendOrService", backend))

    assert len(mcp.registry) == 6
    (
        list_datasets,
        list_operations,
        list_dataflow,
        explain_dataset,
        explain_operation,
        explain_path,
    ) = mcp.registry

    datasets = cast("list[dict[str, object]]", list_datasets())
    dataflow = cast("list[_DataflowPayload]", list_dataflow())
    dataset_details = cast("list[_ExplainPayload]", explain_dataset("table1"))
    op_details = cast("list[_ExplainPayload]", explain_operation("op.one"))
    path = cast("list[_DataflowPayload]", explain_path("table1", "op.one", max_hops=2))

    assert datasets[0]["id"] == "d1"
    assert cast("list[dict[str, object]]", list_operations())[0]["id"] == "op.one"
    assert len(dataflow[0]["nodes"]) == 2
    assert dataset_details[0]["node"]["id"] == "table1"
    assert dataset_details[0]["incoming_edges"] == []
    assert len(dataset_details[0]["outgoing_edges"]) == 1
    assert len(op_details[0]["incoming_edges"]) == 1
    assert op_details[0]["outgoing_edges"] == []
    assert path[0]["edges"]


def test_explain_dataset_returns_error_for_unknown_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown dataset id should yield ProblemDetail payload."""
    monkeypatch.setattr(
        meta_tools,
        "build_dataset_meta",
        lambda _service, _limits: (),
    )
    monkeypatch.setattr(
        meta_tools,
        "iter_registry_operations",
        lambda: (),
    )
    nodes = [
        DataflowNode(
            id="existing",
            kind="table",
            family="fam",
            owner_package="analytics",
            description="desc",
        )
    ]
    edges: list[DataflowEdge] = []
    monkeypatch.setattr(meta_tools, "build_serving_dataflow_graph", lambda: (nodes, edges))
    backend = SimpleNamespace(limits=BackendLimits(), service=SimpleNamespace())
    mcp = _RecordingMcp()
    meta_tools.register_meta_tools(
        cast("FastMCP", mcp), cast("QueryBackendOrService", backend)
    )
    (
        _list_datasets,
        _list_operations,
        _list_dataflow,
        explain_dataset,
        _explain_operation,
        _explain_path,
    ) = mcp.registry

    result = cast("Callable[[str], object]", explain_dataset)("missing")
    assert isinstance(result, dict)
    assert "error" in result
