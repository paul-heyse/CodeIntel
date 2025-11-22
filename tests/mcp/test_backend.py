"""Backend behavior tests for MCP components."""

from __future__ import annotations

import duckdb
import pytest

from codeintel.mcp import errors
from codeintel.mcp.backend import DuckDBBackend


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """
    Create an in-memory DuckDB connection with minimal MCP tables.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Connection seeded with fixture tables for MCP backend tests.
    """
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA IF NOT EXISTS docs;")
    con.execute("CREATE SCHEMA IF NOT EXISTS graph;")
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")
    con.execute(
        """
        CREATE TABLE docs.v_function_summary (
            repo TEXT, commit TEXT, rel_path TEXT, qualname TEXT,
            urn TEXT, function_goid_h128 DECIMAL(38,0),
            start_line INTEGER
        );
        """
    )
    con.execute(
        """
        INSERT INTO docs.v_function_summary VALUES
        ('r', 'c', 'foo.py', 'foo', 'urn:foo', 1, 1);
        """
    )
    con.execute(
        """
        CREATE TABLE docs.v_call_graph_enriched (
            caller_goid_h128 DECIMAL(38,0),
            callee_goid_h128 DECIMAL(38,0),
            caller_qualname TEXT,
            callee_qualname TEXT
        );
        """
    )
    con.execute(
        """
        INSERT INTO docs.v_call_graph_enriched VALUES
        (1, 2, 'foo', 'bar'),
        (3, 1, 'baz', 'foo');
        """
    )
    con.execute(
        """
        CREATE TABLE docs.v_test_to_function (
            repo TEXT, commit TEXT, test_id TEXT, function_goid_h128 DECIMAL(38,0)
        );
        """
    )
    con.execute(
        """
        INSERT INTO docs.v_test_to_function VALUES
        ('r', 'c', 't1', 1);
        """
    )
    con.execute(
        """
        CREATE TABLE docs.v_file_summary (
            repo TEXT, commit TEXT, rel_path TEXT
        );
        """
    )
    con.execute(
        """
        INSERT INTO docs.v_file_summary VALUES
        ('r', 'c', 'foo.py');
        """
    )
    con.execute(
        """
        CREATE TABLE graph.call_graph_edges_dummy(d INT);
        """
    )
    return con


@pytest.fixture
def backend(con: duckdb.DuckDBPyConnection) -> DuckDBBackend:
    """
    Backend configured against the in-memory DuckDB.

    Returns
    -------
    DuckDBBackend
        Backend under test.
    """
    return DuckDBBackend(con=con, repo="r", commit="c")


def test_get_function_summary(backend: DuckDBBackend) -> None:
    """Function summary resolves when URN is provided."""
    resp = backend.get_function_summary(urn="urn:foo")
    if not resp.found or resp.summary is None:
        pytest.fail("Expected function summary to be present")
    if resp.summary["qualname"] != "foo":
        pytest.fail("Qualname mismatch for function summary")


def test_get_function_summary_invalid(backend: DuckDBBackend) -> None:
    """Missing identifiers yields an MCP error."""
    with pytest.raises(errors.McpError):
        backend.get_function_summary()


def test_callgraph_direction_validation(backend: DuckDBBackend) -> None:
    """Direction validation rejects unsupported values."""
    with pytest.raises(errors.McpError):
        backend.get_callgraph_neighbors(goid_h128=1, direction="bad")


def test_get_tests_for_function(backend: DuckDBBackend) -> None:
    """Tests for function returns seeded test rows."""
    resp = backend.get_tests_for_function(goid_h128=1)
    if len(resp.tests) != 1:
        pytest.fail("Expected exactly one test row")
    if resp.tests[0]["test_id"] != "t1":
        pytest.fail("Unexpected test_id in results")


def test_get_tests_for_function_invalid(backend: DuckDBBackend) -> None:
    """Missing identifiers for tests raises MCP error."""
    with pytest.raises(errors.McpError):
        backend.get_tests_for_function()


def test_read_dataset_rows_unknown(backend: DuckDBBackend) -> None:
    """Unknown dataset names raise MCP errors."""
    with pytest.raises(errors.McpError):
        backend.read_dataset_rows(dataset_name="missing")
