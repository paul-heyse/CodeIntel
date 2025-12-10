"""Coverage-related test assertion helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions.expectation_assertions import expect_equal

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


def assert_single_edge(con: DuckDBPyConnection) -> None:
    """Assert a single populated test coverage edge exists.

    Parameters
    ----------
    con
        DuckDB connection.

    Raises
    ------
    AssertionError
        If the edge count or contents do not match expectations.
    """
    rows = con.execute(
        "SELECT test_goid_h128, coverage_ratio, last_status FROM analytics.test_coverage_edges"
    ).fetchall()
    if len(rows) != 1:
        message = f"Expected 1 edge row, got {len(rows)}"
        raise AssertionError(message)
    test_goid, cov_ratio, status = rows[0]
    if test_goid is None:
        message = "Expected test_goid_h128 to be populated"
        raise AssertionError(message)
    tolerance = 1e-6
    if abs(float(cov_ratio) - 1.0) > tolerance:
        message = f"Unexpected coverage_ratio {cov_ratio}"
        raise AssertionError(message)
    if status != "passed":
        message = f"Unexpected last_status {status}"
        raise AssertionError(message)


def assert_coverage_lines(
    con: DuckDBPyConnection,
    *,
    snapshot: SnapshotRef,
    rel_path: str,
    executable: int,
    covered: int,
) -> None:
    """Assert coverage_functions row matches expected counts.

    Raises
    ------
    AssertionError
        If the coverage_functions row is missing or counts differ.
    """
    row = con.execute(
        """
        SELECT executable_lines, covered_lines
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND rel_path = ?
        """,
        [snapshot.repo, snapshot.commit, rel_path],
    ).fetchone()
    if row is None:
        message = f"coverage_functions row missing for {rel_path}"
        raise AssertionError(message)
    expect_equal(row[0], executable)
    expect_equal(row[1], covered)


def assert_function_loc(
    con: DuckDBPyConnection,
    *,
    goid: int,
    loc: int,
    logical_loc: int,
) -> None:
    """Assert function_metrics rows contain expected LOC values.

    Raises
    ------
    AssertionError
        If the metrics row is missing or LOC values differ.
    """
    row = con.execute(
        """
        SELECT loc, logical_loc
        FROM analytics.function_metrics
        WHERE function_goid_h128 = ?
        """,
        [goid],
    ).fetchone()
    if row is None:
        message = f"function_metrics row missing for {goid}"
        raise AssertionError(message)
    expect_equal(row[0], loc)
    expect_equal(row[1], logical_loc)


def assert_typedness_bucket(
    con: DuckDBPyConnection,
    *,
    goid: int,
    bucket: str,
) -> None:
    """Assert typedness bucket for a given GOID from risk factors.

    Raises
    ------
    AssertionError
        If typedness does not match the expected bucket.
    """
    row = con.execute(
        """
        SELECT typedness_bucket
        FROM analytics.goid_risk_factors
        WHERE function_goid_h128 = ?
        """,
        [goid],
    ).fetchone()
    if row is None:
        message = f"typedness row missing for {goid}"
        raise AssertionError(message)
    expect_equal(row[0], bucket)


__all__ = [
    "assert_coverage_lines",
    "assert_function_loc",
    "assert_single_edge",
    "assert_typedness_bucket",
]
