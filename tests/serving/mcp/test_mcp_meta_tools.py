"""Tests for MCP meta tools."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.config.datasets.dataflow import DataflowEdge, DataflowNode
from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.meta_tools import DatasetMetaLike, MetaToolOptions, register_meta_tools
from codeintel.serving.operations.catalog import iter_registry_operations
from codeintel.serving.services.query_service import LocalQueryService, QueryService
from tests._helpers.assertions import (
    assert_logged,
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_length,
)
from tests._helpers.mcp_registrar import RecordingMcpRegistrar
from tests.serving.mcp.conftest import McpBackendComponents

if TYPE_CHECKING:
    from tests.serving.mcp.conftest import McpBackendComponents

DEFAULT_LIMIT = 10
MAX_ROWS = 100


@dataclass(frozen=True)
class _DatasetMetaFake:
    id: str
    name: str
    table_key: str
    description: str
    schema_version: str | None
    family: str | None
    is_docs_view: bool
    is_read_only: bool
    default_limit: int
    max_limit: int


@dataclass(frozen=True)
class _OperationSpecFake:
    id: str
    category: str
    summary: str
    description: str | None
    http_method: str | None
    http_path: str | None
    tool_name: str | None
    output_model_name: str
    required_datasets: tuple[str, ...]
    required_graphs: tuple[str, ...]
    default_limit: int | None
    max_limit: int | None


def _sample_dataset_meta() -> list[_DatasetMetaFake]:
    return [
        _DatasetMetaFake(
            id="analytics.fn_metrics",
            name="Function Metrics Δ",
            table_key="analytics.fn_metrics",
            description="Unicode-friendly metrics table",
            schema_version="1.0.0",
            family="analytics",
            is_docs_view=False,
            is_read_only=False,
            default_limit=25,
            max_limit=250,
        ),
        _DatasetMetaFake(
            id="docs.fn_metrics_view",
            name="Docs View",
            table_key="docs.fn_metrics_view",
            description="Docs view of metrics",
            schema_version=None,
            family=None,
            is_docs_view=True,
            is_read_only=True,
            default_limit=10,
            max_limit=50,
        ),
    ]


def _sample_operations() -> list[_OperationSpecFake]:
    return [
        _OperationSpecFake(
            id="operation.summarize",
            category="meta",
            summary="Summarize datasets Δ",
            description="Summarize analytics datasets with unicode",
            http_method="GET",
            http_path="/meta/summarize",
            tool_name="meta.list_datasets",
            output_model_name="DatasetMetaResponse",
            required_datasets=("analytics.fn_metrics",),
            required_graphs=("graph://functions",),
            default_limit=None,
            max_limit=50,
        ),
        _OperationSpecFake(
            id="operation.graph",
            category="meta",
            summary="Build graph",
            description=None,
            http_method=None,
            http_path=None,
            tool_name="meta.dataflow_graph",
            output_model_name="DataflowGraphResponse",
            required_datasets=("analytics.fn_metrics", "docs.fn_metrics_view"),
            required_graphs=(),
            default_limit=5,
            max_limit=None,
        ),
    ]


def _dataflow_graph() -> tuple[list[DataflowNode], list[DataflowEdge]]:
    nodes = [
        DataflowNode(
            id="analytics.fn_metrics",
            kind="table",
            family="analytics",
            owner_package="analytics",
            description="Function metrics table",
        ),
        DataflowNode(
            id="docs.fn_metrics_view",
            kind="view",
            family="docs",
            owner_package="docs",
            description="Docs view",
        ),
        DataflowNode(
            id="operation.summarize",
            kind="operation",
            family="analytics",
            owner_package="analytics",
            description="Summarize metrics",
        ),
    ]
    edges = [
        DataflowEdge(src="analytics.fn_metrics", dst="docs.fn_metrics_view", edge_type="builds"),
        DataflowEdge(src="docs.fn_metrics_view", dst="operation.summarize", edge_type="reads"),
    ]
    return nodes, edges


def _meta_options() -> MetaToolOptions:
    """Build MetaToolOptions used across tests.

    Returns
    -------
    MetaToolOptions
        Options configured with sample operations and dataset metadata builder.
    """

    def _dataset_meta_builder(
        service: QueryService, limits: BackendLimits
    ) -> list[_DatasetMetaFake]:
        del service, limits
        return _sample_dataset_meta()

    return MetaToolOptions(
        operations=_sample_operations(),
        dataflow_builder=_dataflow_graph,
        dataset_meta_builder=_dataset_meta_builder,
    )


def _register_with_options(
    backend: LocalQueryService, options: MetaToolOptions | None = None
) -> RecordingMcpRegistrar:
    """Register meta tools with provided options and return the registrar.

    Returns
    -------
    RecordingMcpRegistrar
        Registrar with meta tools registered.
    """
    registrar = RecordingMcpRegistrar("meta-tools")
    register_meta_tools(registrar, backend, options=options or _meta_options())
    return registrar


@pytest.fixture
def backend(mcp_service: LocalQueryService) -> LocalQueryService:
    """Backend (query service) for meta tool tests."""
    return mcp_service


def test_meta_tools_list_payloads_with_unicode(
    backend: LocalQueryService,
) -> None:
    """Meta tools should surface unicode dataset metadata and limits."""
    registrar = _register_with_options(backend)

    tools = registrar.list_tools()
    expect_length(tools, 6)
    expect_in("meta.list_datasets", [tool.name for tool in tools])
    expect_in("meta.explain_path", [tool.name for tool in tools])

    datasets = cast("list[dict[str, object]]", registrar.registry["meta.list_datasets"]())
    expect_is_instance(datasets, list)
    expect_length(datasets, 2)
    expect_equal(datasets[0]["name"], "Function Metrics Δ")
    expect_equal(datasets[1]["schema_version"], None)
    expect_equal(datasets[0]["max_limit"], 250)

    operations = cast("list[dict[str, object]]", registrar.registry["meta.list_operations"]())
    expect_length(operations, 2)
    expect_equal(operations[0]["required_datasets"], ["analytics.fn_metrics"])
    expect_equal(operations[0]["default_limit"], DEFAULT_LIMIT)
    expect_equal(operations[1]["max_limit"], MAX_ROWS)

    graph_payload = cast("list[dict[str, object]]", registrar.registry["meta.dataflow_graph"]())
    expect_length(graph_payload, 1)
    graph_entry: dict[str, object] = graph_payload[0]
    nodes = cast("list[object]", graph_entry["nodes"])
    edges = cast("list[object]", graph_entry["edges"])
    expect_length(nodes, 3)
    expect_length(edges, 2)


def test_explain_tools_return_edges_and_nodes(
    backend: LocalQueryService,
) -> None:
    """Explain tools should return structured nodes/edges for datasets and ops."""
    registrar = _register_with_options(backend)

    dataset_payload = cast(
        "list[dict[str, object]]",
        registrar.registry["meta.explain_dataset"]("docs.fn_metrics_view"),
    )
    expect_length(dataset_payload, 1)
    dataset_entry: dict[str, object] = dataset_payload[0]
    expect_length(cast("list[object]", dataset_entry["incoming_edges"]), 1)
    expect_length(cast("list[object]", dataset_entry["outgoing_edges"]), 1)

    operation_payload = cast(
        "list[dict[str, object]]",
        registrar.registry["meta.explain_operation"]("operation.summarize"),
    )
    expect_length(operation_payload, 1)
    operation_entry: dict[str, object] = operation_payload[0]
    expect_length(cast("list[object]", operation_entry["incoming_edges"]), 1)
    expect_length(cast("list[object]", operation_entry["outgoing_edges"]), 0)

    path_payload = cast(
        "list[dict[str, object]]",
        registrar.registry["meta.explain_path"](
            "analytics.fn_metrics",
            "operation.summarize",
            4,
        ),
    )
    expect_length(path_payload, 1)
    path_entry: dict[str, object] = path_payload[0]
    expect_length(cast("list[object]", path_entry["nodes"]), 3)
    expect_length(cast("list[object]", path_entry["edges"]), 2)


def test_unknown_ids_return_problem_detail_and_log_warning(
    backend: LocalQueryService, caplog: pytest.LogCaptureFixture
) -> None:
    """Unknown IDs should yield problem details and log warnings."""
    registrar = _register_with_options(backend)

    with caplog.at_level("WARNING"):
        dataset_error = cast(
            "dict[str, object]", registrar.registry["meta.explain_dataset"]("unknown.dataset")
        )
    expect_in("error", dataset_error)
    dataset_error_detail = cast("dict[str, object]", dataset_error["error"])
    expect_in("Unknown dataset/docs node_id", cast("str", dataset_error_detail["detail"]))
    assert_logged(caplog.records, level="WARNING", containing="Unknown dataset/docs node_id")

    caplog.clear()
    with caplog.at_level("WARNING"):
        operation_error = cast(
            "dict[str, object]", registrar.registry["meta.explain_operation"]("unknown.operation")
        )
    expect_in("error", operation_error)
    operation_error_detail = cast("dict[str, object]", operation_error["error"])
    expect_in("Unknown operation id", cast("str", operation_error_detail["detail"]))
    assert_logged(caplog.records, level="WARNING", containing="Unknown operation id")

    caplog.clear()
    with caplog.at_level("WARNING"):
        path_error = cast(
            "dict[str, object]",
            registrar.registry["meta.explain_path"]("unknown.src", "operation.summarize"),
        )
    expect_in("error", path_error)
    path_error_detail = cast("dict[str, object]", path_error["error"])
    expect_in("Unknown src_id", cast("str", path_error_detail["detail"]))
    assert_logged(caplog.records, level="WARNING", containing="Unknown src_id")


def test_invalid_payloads_return_problem_detail_and_log_warning(
    backend: LocalQueryService, caplog: pytest.LogCaptureFixture
) -> None:
    """Invalid payloads should yield problem details and log warnings."""

    @dataclass
    class _BrokenDatasetMeta:
        id: str
        name: str
        table_key: str
        description: str = "Broken"

    def broken_meta_builder(
        service: QueryService, limits: BackendLimits
    ) -> Iterable[DatasetMetaLike]:
        del service, limits
        return cast(
            "Iterable[DatasetMetaLike]",
            [_BrokenDatasetMeta(id="broken", name="Broken", table_key="broken")],
        )

    registrar = _register_with_options(
        backend,
        options=MetaToolOptions(
            operations=iter_registry_operations(),
            dataflow_builder=_dataflow_graph,
            dataset_meta_builder=broken_meta_builder,
        ),
    )

    with caplog.at_level("WARNING"):
        error_payload = cast("dict[str, object]", registrar.registry["meta.list_datasets"]())
    expect_in("error", error_payload)
    error_detail = cast("dict[str, object]", error_payload["error"])
    expect_in("Dataset meta missing expected attribute", cast("str", error_detail["detail"]))
    assert_logged(
        caplog.records,
        level="WARNING",
        containing="Dataset meta missing expected attribute",
    )
