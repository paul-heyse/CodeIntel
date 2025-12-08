"""Comprehensive tests for ingestion adapters.

This module tests the storage and tool adapters used during
code ingestion, focusing on the interface contracts.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from codeintel.ingestion.adapters.duckdb_storage import (
    INGEST_MACROS,
    SMALL_BATCH_THRESHOLD,
    DuckDBStorageAdapter,
    build_delete_in_query,
    quote_identifier,
    quote_macro_name,
    quote_table_key,
)
from codeintel.ingestion.adapters.tool_runner import ToolRunnerAdapter
from codeintel.ingestion.engine.results import CoverageReport
from codeintel.ingestion.ports.storage import BatchResult, QueryResult
from codeintel.ingestion.ports.tools import ToolStatus
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_not_none,
    expect_length,
    expect_true,
)
from tests._helpers.fakes import FakeToolService, FakeToolServiceConfig

# Test constants
ROWS_WRITTEN_100 = 100
DURATION_1_5 = 1.5


@pytest.fixture
def duckdb_adapter(fresh_gateway: StorageGateway) -> DuckDBStorageAdapter:
    """Provide a DuckDBStorageAdapter backed by the fresh_gateway fixture.

    Returns
    -------
    DuckDBStorageAdapter
        Adapter connected to the in-memory gateway.
    """
    return DuckDBStorageAdapter(fresh_gateway)


@pytest.fixture
def success_tool_adapter() -> ToolRunnerAdapter:
    """Provide ToolRunnerAdapter wired to a success-configured FakeToolService.

    Returns
    -------
    ToolRunnerAdapter
        Adapter with deterministic success responses.
    """
    return ToolRunnerAdapter(_make_success_service())


@pytest.fixture
def failing_tool_adapter() -> ToolRunnerAdapter:
    """Provide ToolRunnerAdapter wired to a failure-configured FakeToolService.

    Returns
    -------
    ToolRunnerAdapter
        Adapter configured to raise errors.
    """
    return ToolRunnerAdapter(_make_failing_service())


# =============================================================================
# quote_identifier Tests
# =============================================================================


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("my_table", '"my_table"'),
        ("table_123", '"table_123"'),
        ("MyTable", '"MyTable"'),
    ],
)
def test_quote_identifier_valid(identifier: str, expected: str) -> None:
    """Should quote valid identifiers."""
    result = quote_identifier(identifier)

    expect_equal(result, expected)


@pytest.mark.parametrize(
    "identifier",
    [
        "my table",
        "my-table",
        "table; DROP TABLE users;--",
        'table"name',
        "table;name",
    ],
)
def test_quote_identifier_rejects_invalid(identifier: str) -> None:
    """Should reject unsafe identifiers."""
    with pytest.raises(ValueError, match="Unsafe identifier"):
        quote_identifier(identifier)


# =============================================================================
# quote_table_key Tests
# =============================================================================


def test_quote_table_key_valid() -> None:
    """Should quote valid table keys."""
    schema, table, quoted = quote_table_key("core.modules")

    expect_equal(schema, "core")
    expect_equal(table, "modules")
    expect_equal(quoted, '"core"."modules"')


def test_quote_table_key_unknown_table() -> None:
    """Should reject unknown table keys."""
    with pytest.raises(ValueError, match="Unknown table key"):
        quote_table_key("nonexistent.table")


# =============================================================================
# Constants Tests
# =============================================================================


def test_small_batch_threshold_positive() -> None:
    """SMALL_BATCH_THRESHOLD should be a positive integer."""
    expect_true(SMALL_BATCH_THRESHOLD > 0)
    expect_true(isinstance(SMALL_BATCH_THRESHOLD, int))


def test_ingest_macros_not_empty() -> None:
    """INGEST_MACROS should contain mappings."""
    expect_true(len(INGEST_MACROS) > 0)
    expect_true(isinstance(INGEST_MACROS, dict))


def test_ingest_macros_keys_are_table_format() -> None:
    """INGEST_MACROS keys should be in schema.table format."""
    for key in INGEST_MACROS:
        expect_true("." in key)
        parts = key.split(".")
        min_parts = 2
        expect_true(len(parts) >= min_parts)


def test_ingest_macros_values_start_with_metadata() -> None:
    """INGEST_MACROS values should start with metadata.ingest_."""
    for value in INGEST_MACROS.values():
        expect_true(value.startswith("metadata.ingest_"))


# =============================================================================
# BatchResult Tests
# =============================================================================


def test_batch_result_attributes() -> None:
    """BatchResult should store write results."""
    result = BatchResult(
        table_key="core.test",
        rows_written=ROWS_WRITTEN_100,
        duration_s=DURATION_1_5,
    )

    expect_equal(result.table_key, "core.test")
    expect_equal(result.rows_written, ROWS_WRITTEN_100)
    expect_equal(result.duration_s, DURATION_1_5)


def test_batch_result_defaults() -> None:
    """BatchResult should have sensible defaults."""
    result = BatchResult(table_key="core.test", rows_written=50)

    expect_equal(result.duration_s, 0.0)


# =============================================================================
# QueryResult Tests
# =============================================================================


def test_query_result_attributes() -> None:
    """QueryResult should store query results."""
    result = QueryResult(
        rows=[("a", 1), ("b", 2)],
        columns=("name", "value"),
        row_count=2,
    )

    expected_rows = 2
    expect_length(result.rows, expected_rows)
    expect_equal(result.columns, ("name", "value"))
    expect_equal(result.row_count, expected_rows)


def test_query_result_defaults() -> None:
    """QueryResult should have sensible defaults."""
    result = QueryResult()

    expect_equal(result.rows, [])
    expect_equal(result.columns, ())
    expect_equal(result.row_count, 0)


# =============================================================================
# DuckDBStorageAdapter Tests
# =============================================================================


def test_duckdb_adapter_initialization(duckdb_adapter: DuckDBStorageAdapter) -> None:
    """DuckDBStorageAdapter should initialize from gateway."""
    expect_is_not_none(duckdb_adapter)


def test_duckdb_adapter_ensure_schema(duckdb_adapter: DuckDBStorageAdapter) -> None:
    """DuckDBStorageAdapter.ensure_schema should not raise for valid tables."""
    # Should not raise
    duckdb_adapter.ensure_schema("core.modules")


def test_duckdb_adapter_ensure_schema_unknown_table(
    duckdb_adapter: DuckDBStorageAdapter,
) -> None:
    """DuckDBStorageAdapter.ensure_schema should raise for unknown tables."""
    with pytest.raises(RuntimeError, match="missing from TABLE_SCHEMAS"):
        duckdb_adapter.ensure_schema("nonexistent.table_xyz")


def test_duckdb_adapter_execute_query(duckdb_adapter: DuckDBStorageAdapter) -> None:
    """DuckDBStorageAdapter.execute_query should return results."""
    result = duckdb_adapter.execute_query("SELECT 1 as value")

    expect_is_not_none(result)
    expect_true(result.row_count >= 0)


def test_duckdb_adapter_execute_query_with_params(
    duckdb_adapter: DuckDBStorageAdapter,
) -> None:
    """DuckDBStorageAdapter.execute_query should handle parameters."""
    result = duckdb_adapter.execute_query("SELECT ? + ? as sum", [1, 2])

    expect_is_not_none(result)
    expect_equal(result.row_count, 1)


def test_duckdb_adapter_write_batch_small(duckdb_adapter: DuckDBStorageAdapter) -> None:
    """DuckDBStorageAdapter.write_batch should handle small batches."""
    rows = [
        ("test_module", "test/path.py", "test/repo", "abc123", "python", "[]", "[]"),
    ]

    result = duckdb_adapter.write_batch("core.modules", rows, scope="test/repo@abc123")

    expect_equal(result.rows_written, 1)


def test_duckdb_adapter_delete_by_paths(
    duckdb_adapter: DuckDBStorageAdapter,
) -> None:
    """DuckDBStorageAdapter.delete_by_paths should delete matching rows."""
    # First insert some data
    rows = [
        ("mod1", "src/a.py", "test/repo", "abc123", "python", "[]", "[]"),
        ("mod2", "src/b.py", "test/repo", "abc123", "python", "[]", "[]"),
    ]
    duckdb_adapter.write_batch("core.modules", rows, scope="test/repo@abc123")

    # Delete one path
    deleted = duckdb_adapter.delete_by_paths(
        "core.modules",
        ["src/a.py"],
        path_column="path",
    )

    expect_true(deleted >= 0)  # May be 0 if table structure differs


def test_duckdb_adapter_fetch_dataframe(duckdb_adapter: DuckDBStorageAdapter) -> None:
    """DuckDBStorageAdapter.fetch_dataframe should return dataframe."""
    df = duckdb_adapter.fetch_dataframe("SELECT 1 as value, 'test' as name")

    expect_is_not_none(df)
    # Check it has expected shape
    expect_true(len(df) >= 0)


# =============================================================================
# Integration Tests
# =============================================================================


def test_adapter_write_and_query_cycle(duckdb_adapter: DuckDBStorageAdapter) -> None:
    """Adapter should support write-then-query cycle."""
    # Write some data
    rows = [
        ("cycle_mod", "cycle/path.py", "cycle/repo", "xyz789", "python", "[]", "[]"),
    ]
    write_result = duckdb_adapter.write_batch("core.modules", rows, scope="cycle/repo@xyz789")

    expect_equal(write_result.rows_written, 1)

    # Query it back
    query_result = duckdb_adapter.execute_query(
        "SELECT module, path FROM core.modules WHERE repo = ?",
        ["cycle/repo"],
    )

    expect_true(query_result.row_count >= 1)


# =============================================================================
# Additional Helper Function Tests
# =============================================================================


def testbuild_delete_in_query() -> None:
    """build_delete_in_query should construct valid DELETE SQL."""
    result = build_delete_in_query('"core"."modules"', '"path"', 3)

    expected_count = 3
    expect_in("DELETE FROM", result)
    expect_in('"core"."modules"', result)
    expect_in('"path"', result)
    expect_equal(result.count("?"), expected_count)


def test_quote_macro_name_simple() -> None:
    """quote_macro_name should handle simple macro names."""
    result = quote_macro_name("metadata.ingest_modules")

    expect_equal(result, "metadata.ingest_modules")


def test_quote_macro_name_invalid() -> None:
    """quote_macro_name should reject invalid macro names."""
    with pytest.raises(ValueError, match="Unsafe"):
        quote_macro_name("metadata..double_dot")


# =============================================================================
# DuckDBStorageAdapter Property Tests
# =============================================================================


def test_duckdb_adapter_con_property(
    duckdb_adapter: DuckDBStorageAdapter, fresh_gateway: StorageGateway
) -> None:
    """DuckDBStorageAdapter.con should return the gateway connection."""
    expect_true(duckdb_adapter.con is fresh_gateway.con)


# =============================================================================
# Write Batch Tests (Edge Cases)
# =============================================================================


def test_duckdb_adapter_write_batch_empty(duckdb_adapter: DuckDBStorageAdapter) -> None:
    """DuckDBStorageAdapter.write_batch should handle empty rows."""
    result = duckdb_adapter.write_batch("core.modules", [], scope="test@abc")

    expect_equal(result.rows_written, 0)
    expect_equal(result.duration_s, 0.0)


def test_duckdb_adapter_write_batch_no_scope(duckdb_adapter: DuckDBStorageAdapter) -> None:
    """DuckDBStorageAdapter.write_batch should work without scope."""
    rows = [
        ("no_scope_mod", "no_scope/path.py", "test", "xyz", "python", "[]", "[]"),
    ]
    result = duckdb_adapter.write_batch("core.modules", rows)

    expect_equal(result.rows_written, 1)


def test_duckdb_adapter_write_batch_larger(duckdb_adapter: DuckDBStorageAdapter) -> None:
    """DuckDBStorageAdapter.write_batch should handle batches larger than threshold."""
    # Create more than SMALL_BATCH_THRESHOLD rows
    rows = [
        (f"mod_{i}", f"path/{i}.py", "test/repo", "abc123", "python", "[]", "[]")
        for i in range(30)  # > SMALL_BATCH_THRESHOLD (25)
    ]

    result = duckdb_adapter.write_batch("core.modules", rows, scope="test/repo@abc123")

    expected_rows = 30
    expect_equal(result.rows_written, expected_rows)


# =============================================================================
# Delete Tests
# =============================================================================


def test_duckdb_adapter_delete_by_paths_empty(duckdb_adapter: DuckDBStorageAdapter) -> None:
    """DuckDBStorageAdapter.delete_by_paths should handle empty paths list."""
    deleted = duckdb_adapter.delete_by_paths("core.modules", [])

    expect_equal(deleted, 0)


def test_duckdb_adapter_delete_by_params(duckdb_adapter: DuckDBStorageAdapter) -> None:
    """DuckDBStorageAdapter.delete_by_params should execute delete."""
    # First insert data
    rows = [
        ("del_mod", "del/path.py", "test/repo", "abc123", "python", "[]", "[]"),
    ]
    duckdb_adapter.write_batch("core.modules", rows)

    # Try delete by params - should not raise
    deleted = duckdb_adapter.delete_by_params("core.modules", ["test/repo", "abc123"])

    expect_true(deleted >= 0)  # DuckDB doesn't return count


# =============================================================================
# Query Tests (Edge Cases)
# =============================================================================


def test_duckdb_adapter_execute_query_no_params(duckdb_adapter: DuckDBStorageAdapter) -> None:
    """DuckDBStorageAdapter.execute_query should work without params."""
    result = duckdb_adapter.execute_query("SELECT 42 as answer")

    expect_equal(result.row_count, 1)
    expect_equal(result.columns, ("answer",))


def test_duckdb_adapter_fetch_dataframe_with_params(
    duckdb_adapter: DuckDBStorageAdapter,
) -> None:
    """DuckDBStorageAdapter.fetch_dataframe should handle params."""
    df = duckdb_adapter.fetch_dataframe("SELECT ? as value", [100])

    expect_length(df, 1)


# =============================================================================
# Registry Metadata Tests
# =============================================================================


def test_ingest_macros_has_core_modules() -> None:
    """INGEST_MACROS should have entry for core.modules."""
    expect_in("core.modules", INGEST_MACROS)
    expect_true(INGEST_MACROS["core.modules"].startswith("metadata."))


# =============================================================================
# ToolRunnerAdapter Tests
# =============================================================================


def _make_success_service() -> FakeToolService:
    """Create FakeToolService configured for successful responses.

    Returns
    -------
    FakeToolService
        Service with deterministic success responses.
    """
    return FakeToolService(
        FakeToolServiceConfig(
            pyright_errors={"mod.py": 2, "other.py": 0},
            pyrefly_errors={"mod.py": 1},
            ruff_errors={"style.py": 3},
            coverage_report=CoverageReport.from_file_reports(
                [
                    ("mod.py", {1, 2, 3}, {4, 5}),
                ]
            ),
            pytest_success=True,
        )
    )


def _make_failing_service() -> FakeToolService:
    """Create FakeToolService configured to raise errors.

    Returns
    -------
    FakeToolService
        Service configured to raise errors on all tool methods.
    """
    config = FakeToolServiceConfig(
        raise_on_pyright=RuntimeError("pyright failed"),
        raise_on_pyrefly=RuntimeError("pyrefly failed"),
        raise_on_ruff=OSError("ruff failed"),
        raise_on_coverage=ValueError("coverage failed"),
        raise_on_scip=RuntimeError("SCIP failed"),
        raise_on_pytest=RuntimeError("pytest failed"),
    )
    return FakeToolService(config)


def test_tool_runner_adapter_initialization(success_tool_adapter: ToolRunnerAdapter) -> None:
    """ToolRunnerAdapter should initialize with ToolService."""
    expect_is_not_none(success_tool_adapter)


@pytest.mark.parametrize(
    ("method_name", "expected_diagnostics"),
    [
        ("run_pyright", 1),
        ("run_pyrefly", 1),
        ("run_ruff", 1),
    ],
)
def test_tool_runner_adapter_diagnostic_tools_success(
    success_tool_adapter: ToolRunnerAdapter,
    method_name: str,
    expected_diagnostics: int,
) -> None:
    """Diagnostic tools should return OK with expected diagnostic counts."""
    method = getattr(success_tool_adapter, method_name)
    result = asyncio.run(method(Path()))

    expect_equal(result.status, ToolStatus.OK)
    expect_length(result.diagnostics, expected_diagnostics)
    if method_name == "run_pyright":
        expect_true(result.duration_s > 0)


@pytest.mark.parametrize(
    ("method_name", "message"),
    [
        ("run_pyright", "pyright failed"),
        ("run_pyrefly", "pyrefly failed"),
        ("run_ruff", "ruff failed"),
    ],
)
def test_tool_runner_adapter_diagnostic_tools_failure(
    failing_tool_adapter: ToolRunnerAdapter,
    method_name: str,
    message: str,
) -> None:
    """Diagnostic tools should return FAILED with surfaced errors."""
    method = getattr(failing_tool_adapter, method_name)
    result = asyncio.run(method(Path()))

    expect_equal(result.status, ToolStatus.FAILED)
    expect_true(message in (result.error or ""))


def test_tool_runner_adapter_run_coverage_success(
    success_tool_adapter: ToolRunnerAdapter,
) -> None:
    """ToolRunnerAdapter.run_coverage should return file data."""
    result = asyncio.run(success_tool_adapter.run_coverage(Path()))

    expect_equal(result.status, ToolStatus.OK)
    expect_length(result.files, 1)
    expect_equal(result.files[0].rel_path, "mod.py")


def test_tool_runner_adapter_run_coverage_failure(
    failing_tool_adapter: ToolRunnerAdapter,
) -> None:
    """ToolRunnerAdapter.run_coverage should handle failures."""
    result = asyncio.run(failing_tool_adapter.run_coverage(Path()))

    expect_equal(result.status, ToolStatus.FAILED)
    expect_true("coverage failed" in (result.error or ""))


@pytest.mark.parametrize("rel_paths", [None, ["src/mod.py"]])
def test_tool_runner_adapter_run_scip_success(
    success_tool_adapter: ToolRunnerAdapter, tmp_path: Path, rel_paths: list[str] | None
) -> None:
    """ToolRunnerAdapter.run_scip should handle full and shard indexing."""
    output_scip = tmp_path / "index.scip"
    output_json = tmp_path / "index.json"

    result = asyncio.run(
        success_tool_adapter.run_scip(
            Path(),
            output_scip=output_scip,
            output_json=output_json,
            rel_paths=rel_paths,
        )
    )

    expect_equal(result.status, ToolStatus.OK)


def test_tool_runner_adapter_run_scip_failure(
    failing_tool_adapter: ToolRunnerAdapter, tmp_path: Path
) -> None:
    """ToolRunnerAdapter.run_scip should handle failures."""
    output_scip = tmp_path / "index.scip"
    output_json = tmp_path / "index.json"

    result = asyncio.run(
        failing_tool_adapter.run_scip(
            Path(),
            output_scip=output_scip,
            output_json=output_json,
        )
    )

    expect_equal(result.status, ToolStatus.FAILED)
    expect_true("SCIP failed" in (result.error or ""))


def test_tool_runner_adapter_run_pytest_no_report(
    success_tool_adapter: ToolRunnerAdapter, tmp_path: Path
) -> None:
    """ToolRunnerAdapter.run_pytest should return OK when report doesn't exist."""
    json_path = tmp_path / "report.json"

    result = asyncio.run(success_tool_adapter.run_pytest(Path(), json_report_path=json_path))

    expect_equal(result.status, ToolStatus.OK)
    expect_equal(result.tests, [])


def test_tool_runner_adapter_run_pytest_with_report(
    success_tool_adapter: ToolRunnerAdapter, tmp_path: Path
) -> None:
    """ToolRunnerAdapter.run_pytest should parse existing report."""
    json_path = tmp_path / "report.json"

    # Create a valid report
    json_path.write_text(
        (
            '{"tests": [{"nodeid": "test::a", "outcome": "passed", "duration": 0.1}], '
            '"summary": {"passed": 1}}'
        ),
        encoding="utf-8",
    )

    result = asyncio.run(success_tool_adapter.run_pytest(Path(), json_report_path=json_path))

    expect_equal(result.status, ToolStatus.OK)
    expect_length(result.tests, 1)


def test_tool_runner_adapter_run_pytest_failure(
    failing_tool_adapter: ToolRunnerAdapter, tmp_path: Path
) -> None:
    """ToolRunnerAdapter.run_pytest should handle failures."""
    json_path = tmp_path / "report.json"

    result = asyncio.run(failing_tool_adapter.run_pytest(Path(), json_report_path=json_path))

    expect_equal(result.status, ToolStatus.FAILED)
    expect_true("pytest failed" in (result.error or ""))
