"""Tests for meta MCP tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TypedDict, cast

from codeintel.config.datasets.dataflow import DataflowEdge, DataflowNode
from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.meta_tools import MetaToolOptions, register_meta_tools
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from codeintel.serving.operations.catalog import DataSourceType, Operation
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_length,
    expect_true,
)


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
        self.registry: list[Callable[..., object]] = []

    def tool(
        self, name: str | None = None, **options: object
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        def _decorator(
            func: Callable[..., object],
        ) -> Callable[..., object]:
            _ = name
            _ = options
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


def test_register_meta_tools_registers_expected_tools() -> None:
    """Meta tools should register and return serialized payloads."""
    dataset_meta = (
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
    )
    operations = (_make_operation("op.one", "tool_one"),)

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

    backend = SimpleNamespace(limits=BackendLimits(), service=SimpleNamespace())
    mcp = _RecordingMcp()

    register_meta_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        options=MetaToolOptions(
            operations=operations,
            dataflow_builder=lambda: (nodes, edges),
            dataset_meta_builder=lambda _service, _limits: dataset_meta,
        ),
    )

    expect_length(mcp.registry, 6)
    (
        list_datasets,
        list_operations,
        list_dataflow,
        explain_dataset,
        explain_operation,
        explain_path,
    ) = mcp.registry

    expect_equal(cast("list[dict[str, object]]", list_datasets())[0]["id"], "d1")
    expect_equal(cast("list[dict[str, object]]", list_operations())[0]["id"], "op.one")
    expect_length(cast("list[_DataflowPayload]", list_dataflow())[0]["nodes"], 2)
    dataset_details = cast("list[_ExplainPayload]", explain_dataset("table1"))
    expect_equal(dataset_details[0]["node"]["id"], "table1")
    expect_equal(dataset_details[0]["incoming_edges"], [])
    expect_length(dataset_details[0]["outgoing_edges"], 1)
    op_details = cast("list[_ExplainPayload]", explain_operation("op.one"))
    expect_length(op_details[0]["incoming_edges"], 1)
    expect_equal(op_details[0]["outgoing_edges"], [])
    expect_true(
        bool(
            cast("list[_DataflowPayload]", explain_path("table1", "op.one", max_hops=2))[0]["edges"]
        )
    )


def test_explain_dataset_returns_error_for_unknown_id() -> None:
    """Unknown dataset id should yield ProblemDetail payload."""
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
    backend = SimpleNamespace(limits=BackendLimits(), service=SimpleNamespace())
    mcp = _RecordingMcp()
    register_meta_tools(
        mcp,
        cast("QueryBackendOrService", backend),
        options=MetaToolOptions(
            operations=(),
            dataflow_builder=lambda: (nodes, edges),
            dataset_meta_builder=lambda _service, _limits: (),
        ),
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
    expect_is_instance(result, dict)
    result_dict = cast("dict[str, object]", result)
    expect_in("error", result_dict)
