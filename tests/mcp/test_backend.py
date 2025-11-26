"""Backend behavior tests for MCP components."""

from __future__ import annotations

import pytest

from codeintel.serving.mcp import errors
from codeintel.serving.mcp.backend import MAX_ROWS_LIMIT, DuckDBBackend
from codeintel.storage.gateway import StorageGateway
from tests._helpers.fixtures import seed_mcp_backend


@pytest.fixture
def gateway(fresh_gateway: StorageGateway) -> StorageGateway:
    """
    Create an in-memory DuckDB gateway with minimal MCP tables.

    Returns
    -------
    StorageGateway
        Connection seeded with fixture tables for MCP backend tests.
    """
    seed_mcp_backend(fresh_gateway, repo="r", commit="c")
    return fresh_gateway


@pytest.fixture
def backend(gateway: StorageGateway) -> DuckDBBackend:
    """
    Backend configured against the in-memory DuckDB.

    Returns
    -------
    DuckDBBackend
        Backend under test.
    """
    return DuckDBBackend(gateway=gateway, repo="r", commit="c")


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
    resp = backend.read_dataset_rows(dataset_name="function_validation", limit=1000, offset=-2)
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
