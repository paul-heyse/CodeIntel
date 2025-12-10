"""Tests for DataflowRepository."""

from __future__ import annotations

from typing import cast

import pytest

from codeintel.storage.repositories.dataflow import DataflowRepository
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_length,
    expect_true,
)
from tests._helpers.builders import (
    DatasetDataflowEdgeRow,
    DatasetDataflowNodeRow,
    insert_rows,
)
from tests._helpers.context import TestContext

DATAFLOW_NODES: tuple[DatasetDataflowNodeRow, ...] = (
    DatasetDataflowNodeRow(
        id="node1",
        kind="table",
        family="core",
        owner_package="pkg.one",
        description="First node",
    ),
    DatasetDataflowNodeRow(
        id="node2",
        kind="view",
        family="analytics",
        owner_package="pkg.two",
        description="Second node",
    ),
    DatasetDataflowNodeRow(
        id="node3",
        kind="table",
        family="graph",
        owner_package="pkg.three",
        description="Third node",
    ),
)

DATAFLOW_EDGES: tuple[DatasetDataflowEdgeRow, ...] = (
    DatasetDataflowEdgeRow(src="node1", dst="node2", edge_type="depends_on"),
    DatasetDataflowEdgeRow(src="node2", dst="node3", edge_type="depends_on"),
    DatasetDataflowEdgeRow(src="node1", dst="node3", edge_type="produces"),
)


@pytest.fixture
def dataflow_ctx(test_ctx: TestContext) -> TestContext:
    """Seed dataset dataflow tables for repository tests."""
    con = test_ctx.gateway.con
    base_counts = (
        len(con.execute("SELECT id FROM metadata.dataset_dataflow_nodes").fetchall()),
        len(con.execute("SELECT src FROM metadata.dataset_dataflow_edges").fetchall()),
    )
    insert_rows(test_ctx.gateway, DATAFLOW_NODES)
    insert_rows(test_ctx.gateway, DATAFLOW_EDGES)
    test_ctx.extra["dataflow_counts"] = base_counts
    return test_ctx


@pytest.fixture
def dataflow_repo(dataflow_ctx: TestContext) -> DataflowRepository:
    """Provide a DataflowRepository backed by a seeded TestContext."""
    return DataflowRepository(
        gateway=dataflow_ctx.gateway,
        repo=dataflow_ctx.repo,
        commit=dataflow_ctx.commit,
    )


def test_list_nodes_returns_nodes(
    dataflow_repo: DataflowRepository, dataflow_ctx: TestContext
) -> None:
    """Verify list_nodes returns dataflow nodes including seeded data."""
    nodes = dataflow_repo.list_nodes()

    base_node_count, _ = cast(
        "tuple[int, int]", dataflow_ctx.extra.get("dataflow_counts", (0, 0))
    )
    expect_length(nodes, base_node_count + len(DATAFLOW_NODES), label="node count")

    node_ids = [n["id"] for n in nodes]
    expect_in("node1", node_ids, label="node1 present")
    expect_in("node2", node_ids, label="node2 present")
    expect_in("node3", node_ids, label="node3 present")


def test_list_nodes_includes_expected_columns(
    dataflow_repo: DataflowRepository,
) -> None:
    """Verify list_nodes returns nodes with expected columns."""
    nodes = dataflow_repo.list_nodes()

    expect_true(len(nodes) > 0, message="nodes returned")
    first_node = nodes[0]
    expect_in("id", first_node, label="id key present")
    expect_in("kind", first_node, label="kind key present")
    expect_in("family", first_node, label="family key present")


def test_list_edges_returns_edges(
    dataflow_repo: DataflowRepository, dataflow_ctx: TestContext
) -> None:
    """Verify list_edges returns edges including bootstrapped data."""
    edges = dataflow_repo.list_edges()

    _, base_edge_count = cast(
        "tuple[int, int]", dataflow_ctx.extra.get("dataflow_counts", (0, 0))
    )
    expect_length(edges, base_edge_count + len(DATAFLOW_EDGES), label="edge count")


def test_list_edges_filtered_by_src(dataflow_repo: DataflowRepository) -> None:
    """Verify list_edges filters by source node."""
    edges = dataflow_repo.list_edges(src="node1")

    expected_count = 2
    expect_length(edges, expected_count, label="filtered edge count")
    expect_true(all(e["src"] == "node1" for e in edges), message="all edges match src")


def test_list_edges_filtered_by_dst(dataflow_repo: DataflowRepository) -> None:
    """Verify list_edges filters by destination node."""
    edges = dataflow_repo.list_edges(dst="node3")

    expected_count = 2
    expect_length(edges, expected_count, label="filtered edge count")
    expect_true(all(e["dst"] == "node3" for e in edges), message="all edges match dst")


def test_list_edges_filtered_by_both(dataflow_repo: DataflowRepository) -> None:
    """Verify list_edges filters by both src and dst."""
    edges = dataflow_repo.list_edges(src="node1", dst="node2")

    expect_length(edges, 1, label="filtered edges")
    expect_equal(edges[0]["src"], "node1", label="src")
    expect_equal(edges[0]["dst"], "node2", label="dst")
    expect_equal(edges[0]["edge_type"], "depends_on", label="edge_type")


def test_list_edges_returns_empty_when_no_match(
    dataflow_repo: DataflowRepository,
) -> None:
    """Verify list_edges returns empty list when filter has no matches."""
    edges = dataflow_repo.list_edges(src="nonexistent")

    expect_equal(edges, [], label="no matching edges")
