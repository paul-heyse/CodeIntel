"""Backend behavior tests for MCP components."""

from __future__ import annotations

import duckdb
import pytest

from codeintel.mcp import errors
from codeintel.mcp.backend import MAX_ROWS_LIMIT, DuckDBBackend


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
    con.execute(
        """
        CREATE TABLE analytics.function_validation(
            repo TEXT,
            commit TEXT,
            rel_path TEXT,
            qualname TEXT,
            issue TEXT,
            detail TEXT,
            created_at TIMESTAMP
        );
        """
    )
    con.execute(
        """
        INSERT INTO analytics.function_validation VALUES
        ('r', 'c', 'foo.py', 'foo', 'span_not_found', 'Span 1-2', CURRENT_TIMESTAMP);
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


def test_dataset_rows_clamping(backend: DuckDBBackend) -> None:
    """Limits/offsets clamp with messages instead of raising."""
    backend.dataset_tables = {"functions": "docs.v_function_summary"}
    resp = backend.read_dataset_rows(dataset_name="functions", limit=1000, offset=-2)
    if resp.limit != MAX_ROWS_LIMIT or resp.offset != 0:
        pytest.fail("Expected clamped limit/offset values")
    codes = {m.code for m in resp.meta.messages}
    if not {"limit_clamped", "offset_invalid"} <= codes:
        pytest.fail(f"Unexpected message codes: {codes}")
    if resp.rows:
        pytest.fail("Expected no rows when clamping with invalid offset")


def test_tests_for_function_not_found(backend: DuckDBBackend) -> None:
    """Missing test edges returns message instead of raising."""
    resp = backend.get_tests_for_function(goid_h128=999, limit=5)
    if resp.tests:
        pytest.fail("Expected no tests for missing function")
    codes = [m.code for m in resp.meta.messages]
    if "not_found" not in codes:
        pytest.fail(f"Expected not_found message; got {codes}")


def test_read_function_validation_dataset(backend: DuckDBBackend) -> None:
    """function_validation dataset should be readable via dataset APIs."""
    resp = backend.read_dataset_rows(dataset_name="function_validation", limit=5)
    if not resp.rows:
        pytest.fail("Expected rows for function_validation dataset")
    row = resp.rows[0]
    if row.get("issue") != "span_not_found":
        pytest.fail("Unexpected issue value in function_validation dataset")


def test_dataset_list_includes_function_validation(backend: DuckDBBackend) -> None:
    """Dataset registry should include function_validation."""
    datasets = backend.list_datasets()
    names = {ds.name for ds in datasets}
    if "function_validation" not in names:
        pytest.fail(f"function_validation missing from dataset list: {names}")


def test_function_validation_clamping(backend: DuckDBBackend) -> None:
    """Clamping should apply and return messages for function_validation reads."""
    resp = backend.read_dataset_rows(dataset_name="function_validation", limit=10, offset=-1)
    codes = {m.code for m in resp.meta.messages}
    if "offset_invalid" not in codes:
        pytest.fail(f"Expected offset_invalid message; got {codes}")
    if resp.offset != 0:
        pytest.fail("Expected offset to clamp to 0")
