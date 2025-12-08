"""Tests for DataflowRepository."""

from __future__ import annotations

from duckdb import DuckDBPyConnection

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.repositories.dataflow import DataflowRepository
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_length,
    expect_true,
)


def _seed_dataflow_data(con: DuckDBPyConnection) -> None:
    """Seed dataflow tables with test data."""
    con.execute(
        """
        INSERT INTO metadata.dataset_dataflow_nodes (id, kind, family, owner_package, description)
        VALUES
            ('node1', 'table', 'core', 'pkg.one', 'First node'),
            ('node2', 'view', 'analytics', 'pkg.two', 'Second node'),
            ('node3', 'table', 'graph', 'pkg.three', 'Third node')
        """
    )

    con.execute(
        """
        INSERT INTO metadata.dataset_dataflow_edges (src, dst, edge_type)
        VALUES
            ('node1', 'node2', 'depends_on'),
            ('node2', 'node3', 'depends_on'),
            ('node1', 'node3', 'produces')
        """
    )


def test_list_nodes_returns_nodes(fresh_gateway: StorageGateway) -> None:
    """Verify list_nodes returns dataflow nodes including seeded data."""
    con = fresh_gateway.con

    initial_count = len(con.execute("SELECT id FROM metadata.dataset_dataflow_nodes").fetchall())

    _seed_dataflow_data(con)

    repo = DataflowRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    nodes = repo.list_nodes()

    expected_new_count = 3
    expect_length(nodes, initial_count + expected_new_count, label="node count")

    node_ids = [n["id"] for n in nodes]
    expect_in("node1", node_ids, label="node1 present")
    expect_in("node2", node_ids, label="node2 present")
    expect_in("node3", node_ids, label="node3 present")


def test_list_nodes_includes_expected_columns(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify list_nodes returns nodes with expected columns."""
    con = fresh_gateway.con
    _seed_dataflow_data(con)

    repo = DataflowRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    nodes = repo.list_nodes()

    expect_true(len(nodes) > 0, message="nodes returned")
    first_node = nodes[0]
    expect_in("id", first_node, label="id key present")
    expect_in("kind", first_node, label="kind key present")
    expect_in("family", first_node, label="family key present")


def test_list_edges_returns_edges(fresh_gateway: StorageGateway) -> None:
    """Verify list_edges returns edges including bootstrapped data."""
    con = fresh_gateway.con

    initial_count = len(con.execute("SELECT src FROM metadata.dataset_dataflow_edges").fetchall())

    _seed_dataflow_data(con)

    repo = DataflowRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    edges = repo.list_edges()

    expected_new_count = 3
    expect_true(len(edges) >= initial_count + expected_new_count, message="edge count increased")


def test_list_edges_filtered_by_src(fresh_gateway: StorageGateway) -> None:
    """Verify list_edges filters by source node."""
    con = fresh_gateway.con
    _seed_dataflow_data(con)

    repo = DataflowRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    edges = repo.list_edges(src="node1")

    expected_count = 2
    expect_length(edges, expected_count, label="filtered edge count")
    expect_true(all(e["src"] == "node1" for e in edges), message="all edges match src")


def test_list_edges_filtered_by_dst(fresh_gateway: StorageGateway) -> None:
    """Verify list_edges filters by destination node."""
    con = fresh_gateway.con
    _seed_dataflow_data(con)

    repo = DataflowRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    edges = repo.list_edges(dst="node3")

    expected_count = 2
    expect_length(edges, expected_count, label="filtered edge count")
    expect_true(all(e["dst"] == "node3" for e in edges), message="all edges match dst")


def test_list_edges_filtered_by_both(fresh_gateway: StorageGateway) -> None:
    """Verify list_edges filters by both src and dst."""
    con = fresh_gateway.con
    _seed_dataflow_data(con)

    repo = DataflowRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    edges = repo.list_edges(src="node1", dst="node2")

    expect_length(edges, 1, label="filtered edges")
    expect_equal(edges[0]["src"], "node1", label="src")
    expect_equal(edges[0]["dst"], "node2", label="dst")
    expect_equal(edges[0]["edge_type"], "depends_on", label="edge_type")


def test_list_edges_returns_empty_when_no_match(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify list_edges returns empty list when filter has no matches."""
    con = fresh_gateway.con
    _seed_dataflow_data(con)

    repo = DataflowRepository(
        gateway=fresh_gateway,
        repo="test/repo",
        commit="abc123",
    )

    edges = repo.list_edges(src="nonexistent")

    expect_equal(edges, [], label="no matching edges")
