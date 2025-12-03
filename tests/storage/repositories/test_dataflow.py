"""Tests for DataflowRepository."""

from __future__ import annotations

from duckdb import DuckDBPyConnection

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.repositories.dataflow import DataflowRepository


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
    assert len(nodes) == initial_count + expected_new_count

    node_ids = [n["id"] for n in nodes]
    assert "node1" in node_ids
    assert "node2" in node_ids
    assert "node3" in node_ids


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

    assert len(nodes) > 0
    first_node = nodes[0]
    assert "id" in first_node
    assert "kind" in first_node
    assert "family" in first_node


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
    assert len(edges) >= initial_count + expected_new_count


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
    assert len(edges) == expected_count
    assert all(e["src"] == "node1" for e in edges)


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
    assert len(edges) == expected_count
    assert all(e["dst"] == "node3" for e in edges)


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

    assert len(edges) == 1
    assert edges[0]["src"] == "node1"
    assert edges[0]["dst"] == "node2"
    assert edges[0]["edge_type"] == "depends_on"


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

    assert edges == []
