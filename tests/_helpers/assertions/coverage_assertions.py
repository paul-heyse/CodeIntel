"""Coverage-related test assertion helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

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


__all__ = [
    "assert_single_edge",
]
