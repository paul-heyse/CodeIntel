"""Tests for analytics.coverage_analytics module.

Covers the compute_coverage_functions function which aggregates line-level
coverage data into function-level coverage statistics.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.coverage_analytics import compute_coverage_functions
from codeintel.config import ConfigBuilder
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway, open_memory_gateway

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


# ---------------------------------------------------------------------------
# Test Constants
# ---------------------------------------------------------------------------

EXPECTED_EXECUTABLE_5 = 5
EXPECTED_EXECUTABLE_3 = 3
EXPECTED_EXECUTABLE_6 = 6
EXPECTED_COVERED_2 = 2
EXPECTED_COVERED_3 = 3
EXPECTED_FUNCTIONS_COUNT = 4
COVERAGE_TOLERANCE = 0.001
DIVIDE_COVERAGE_LOW = 0.5
DIVIDE_COVERAGE_HIGH = 0.6
DIVIDE_ERROR_LINE_START = 22

# Hash constants for test GOIDs (DECIMAL(38,0) in schema)
HASH_1 = 1001
HASH_2 = 1002
HASH_3 = 1003
HASH_4 = 1004
HASH_A = 1005
HASH_B = 1006
HASH_METHOD = 1007
HASH_CLASS = 1008
HASH_IDEM = 1009
HASH_REPO1 = 1010
HASH_REPO2 = 1011
HASH_SINGLE = 1012
HASH_CALC_CLASS = 2001
HASH_CALC_INIT = 2002
HASH_CALC_ADD = 2003
HASH_CALC_DIVIDE = 2004
HASH_HELPER = 2005


# ---------------------------------------------------------------------------
# Test Data Classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoidSeedData:
    """Data for seeding a GOID into the database."""

    urn: str
    rel_path: str
    kind: str
    qualname: str
    goid_h128: int
    start_line: int
    end_line: int | None
    language: str = "python"


@dataclass(frozen=True)
class CoverageLineSeedData:
    """Data for seeding a coverage line into the database."""

    rel_path: str
    line: int
    is_executable: bool
    is_covered: bool


@dataclass(frozen=True)
class CoverageRangeSeedData:
    """Data for seeding a range of coverage lines."""

    rel_path: str
    start: int
    end: int
    is_executable: bool = True
    is_covered: bool = True


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot() -> SnapshotRef:
    """
    Create a standard test snapshot reference.

    Returns
    -------
    SnapshotRef
        A snapshot reference for testing.
    """
    return SnapshotRef(repo="test-repo", commit="abc123", repo_root=Path.cwd())


@pytest.fixture
def analytics_gateway() -> Iterator[StorageGateway]:
    """
    Create an in-memory gateway with full schema for coverage analytics tests.

    Yields
    ------
    StorageGateway
        Gateway backed by in-memory DuckDB with full schema.
    """
    gw = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    try:
        yield gw
    finally:
        gw.close()


# ---------------------------------------------------------------------------
# Schema Setup Helpers
# ---------------------------------------------------------------------------


def _seed_goid(
    con: DuckDBPyConnection,
    snapshot: SnapshotRef,
    data: GoidSeedData,
) -> None:
    """
    Seed a single GOID into core.goids.

    Parameters
    ----------
    con
        DuckDB connection.
    snapshot
        Snapshot reference for repo/commit.
    data
        GOID data to seed.
    """
    con.execute(
        """
        INSERT INTO core.goids (
            urn, repo, commit, rel_path, language, kind, qualname, goid_h128,
            start_line, end_line, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())
        """,
        [
            data.urn,
            snapshot.repo,
            snapshot.commit,
            data.rel_path,
            data.language,
            data.kind,
            data.qualname,
            data.goid_h128,
            data.start_line,
            data.end_line,
        ],
    )


def _seed_coverage_line(
    con: DuckDBPyConnection,
    snapshot: SnapshotRef,
    data: CoverageLineSeedData,
) -> None:
    """
    Seed a single coverage line into analytics.coverage_lines.

    Parameters
    ----------
    con
        DuckDB connection.
    snapshot
        Snapshot reference for repo/commit.
    data
        Coverage line data to seed.
    """
    # hits = 1 if covered, 0 otherwise; context_count = 0 for simple test data
    hits = 1 if data.is_covered else 0
    context_count = 0
    con.execute(
        """
        INSERT INTO analytics.coverage_lines (
            repo, commit, rel_path, line, is_executable, is_covered, hits, context_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NOW())
        """,
        [
            snapshot.repo,
            snapshot.commit,
            data.rel_path,
            data.line,
            data.is_executable,
            data.is_covered,
            hits,
            context_count,
        ],
    )


def _seed_coverage_lines_range(
    con: DuckDBPyConnection,
    snapshot: SnapshotRef,
    data: CoverageRangeSeedData,
) -> None:
    """
    Seed multiple coverage lines for a range of line numbers.

    Parameters
    ----------
    con
        DuckDB connection.
    snapshot
        Snapshot reference for repo/commit.
    data
        Coverage range data to seed.
    """
    for line in range(data.start, data.end):
        _seed_coverage_line(
            con,
            snapshot,
            CoverageLineSeedData(
                rel_path=data.rel_path,
                line=line,
                is_executable=data.is_executable,
                is_covered=data.is_covered,
            ),
        )


def _query_coverage_function(
    con: DuckDBPyConnection,
    snapshot: SnapshotRef,
    goid_h128: int,
) -> tuple[object, ...] | None:
    """
    Query a coverage function by hash.

    Parameters
    ----------
    con
        DuckDB connection.
    snapshot
        Snapshot reference for repo/commit.
    goid_h128
        GOID hash to query.

    Returns
    -------
    tuple[object, ...] | None
        Row tuple or None if not found.
    """
    return con.execute(
        """
        SELECT
            executable_lines, covered_lines, coverage_ratio, tested, untested_reason
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, goid_h128],
    ).fetchone()


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_empty_goids_produces_no_rows(
    snapshot: SnapshotRef, analytics_gateway: StorageGateway
) -> None:
    """Verify that no GOIDs results in no coverage function rows."""
    con = analytics_gateway.con

    cfg = ConfigBuilder.from_snapshot(
            repo=snapshot.repo, commit=snapshot.commit, repo_root=snapshot.repo_root
        ).coverage_analytics()
    compute_coverage_functions(analytics_gateway, cfg)

    result = con.execute(
        """
        SELECT COUNT(*) FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ?
        """,
        [snapshot.repo, snapshot.commit],
    ).fetchone()

    assert result is not None
    assert result[0] == 0


def test_single_function_fully_covered(
    snapshot: SnapshotRef, analytics_gateway: StorageGateway
) -> None:
    """Verify coverage ratio for a fully covered function."""
    con = analytics_gateway.con

    _seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:test:func1", "module.py", "function", "my_function", HASH_1, 1, 5),
    )

    _seed_coverage_lines_range(con, snapshot, CoverageRangeSeedData("module.py", 1, 6))

    cfg = ConfigBuilder.from_snapshot(
            repo=snapshot.repo, commit=snapshot.commit, repo_root=snapshot.repo_root
        ).coverage_analytics()
    compute_coverage_functions(analytics_gateway, cfg)

    result = _query_coverage_function(con, snapshot, HASH_1)

    assert result is not None
    assert result[0] == EXPECTED_EXECUTABLE_5
    assert result[1] == EXPECTED_EXECUTABLE_5
    assert result[2] == 1.0
    assert result[3] is True
    assert not result[4]


def test_single_function_partially_covered(
    snapshot: SnapshotRef, analytics_gateway: StorageGateway
) -> None:
    """Verify coverage ratio for a partially covered function."""
    con = analytics_gateway.con

    _seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:test:func2", "partial.py", "function", "partial_func", HASH_2, 1, 4),
    )

    # 4 lines: 3 executable, 2 covered
    _seed_coverage_line(
        con, snapshot, CoverageLineSeedData("partial.py", 1, is_executable=True, is_covered=True)
    )
    _seed_coverage_line(
        con, snapshot, CoverageLineSeedData("partial.py", 2, is_executable=True, is_covered=True)
    )
    _seed_coverage_line(
        con, snapshot, CoverageLineSeedData("partial.py", 3, is_executable=True, is_covered=False)
    )
    _seed_coverage_line(
        con, snapshot, CoverageLineSeedData("partial.py", 4, is_executable=False, is_covered=False)
    )

    cfg = ConfigBuilder.from_snapshot(
            repo=snapshot.repo, commit=snapshot.commit, repo_root=snapshot.repo_root
        ).coverage_analytics()
    compute_coverage_functions(analytics_gateway, cfg)

    result = _query_coverage_function(con, snapshot, HASH_2)

    assert result is not None
    assert result[0] == EXPECTED_EXECUTABLE_3
    assert result[1] == EXPECTED_COVERED_2
    assert isinstance(result[2], (int, float))
    assert abs(result[2] - (2 / 3)) < COVERAGE_TOLERANCE
    assert result[3] is True
    assert not result[4]


def test_function_no_coverage_data(
    snapshot: SnapshotRef, analytics_gateway: StorageGateway
) -> None:
    """Verify function with no coverage lines is marked untested."""
    con = analytics_gateway.con

    _seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:test:func3", "uncovered.py", "function", "uncovered_func", HASH_3, 1, 10),
    )

    cfg = ConfigBuilder.from_snapshot(
            repo=snapshot.repo, commit=snapshot.commit, repo_root=snapshot.repo_root
        ).coverage_analytics()
    compute_coverage_functions(analytics_gateway, cfg)

    result = _query_coverage_function(con, snapshot, HASH_3)

    assert result is not None
    assert result[0] == 0
    assert result[1] == 0
    assert result[2] is None
    assert result[3] is False
    assert result[4] == "no_executable_code"


def test_function_with_executable_but_no_covered_lines(
    snapshot: SnapshotRef, analytics_gateway: StorageGateway
) -> None:
    """Verify function with executable but zero covered lines."""
    con = analytics_gateway.con

    _seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:test:func4", "no_tests.py", "function", "no_tests_func", HASH_4, 5, 10),
    )

    _seed_coverage_lines_range(
        con, snapshot, CoverageRangeSeedData("no_tests.py", 5, 11, is_covered=False)
    )

    cfg = ConfigBuilder.from_snapshot(
            repo=snapshot.repo, commit=snapshot.commit, repo_root=snapshot.repo_root
        ).coverage_analytics()
    compute_coverage_functions(analytics_gateway, cfg)

    result = _query_coverage_function(con, snapshot, HASH_4)

    assert result is not None
    assert result[0] == EXPECTED_EXECUTABLE_6
    assert result[1] == 0
    assert result[2] == 0.0
    assert result[3] is False
    assert result[4] == "no_tests"


def test_method_kind_included(
    snapshot: SnapshotRef, analytics_gateway: StorageGateway
) -> None:
    """Verify that 'method' kind GOIDs are included in aggregation."""
    con = analytics_gateway.con

    _seed_goid(
        con,
        snapshot,
        GoidSeedData(
            "urn:test:method1", "class_mod.py", "method", "MyClass.my_method", HASH_METHOD, 10, 15
        ),
    )

    _seed_coverage_lines_range(con, snapshot, CoverageRangeSeedData("class_mod.py", 10, 16))

    cfg = ConfigBuilder.from_snapshot(
            repo=snapshot.repo, commit=snapshot.commit, repo_root=snapshot.repo_root
        ).coverage_analytics()
    compute_coverage_functions(analytics_gateway, cfg)

    result = con.execute(
        """
        SELECT kind, coverage_ratio, tested
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_METHOD],
    ).fetchone()

    assert result is not None
    assert result[0] == "method"
    assert result[1] == 1.0
    assert result[2] is True


def test_class_kind_excluded(
    snapshot: SnapshotRef, analytics_gateway: StorageGateway
) -> None:
    """Verify that 'class' kind GOIDs are NOT included in aggregation."""
    con = analytics_gateway.con

    _seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:test:class1", "class_def.py", "class", "MyClass", HASH_CLASS, 1, 50),
    )

    _seed_coverage_lines_range(con, snapshot, CoverageRangeSeedData("class_def.py", 1, 51))

    cfg = ConfigBuilder.from_snapshot(
            repo=snapshot.repo, commit=snapshot.commit, repo_root=snapshot.repo_root
        ).coverage_analytics()
    compute_coverage_functions(analytics_gateway, cfg)

    result = con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_CLASS],
    ).fetchone()

    assert result is not None
    assert result[0] == 0


def test_multiple_functions_same_file(
    snapshot: SnapshotRef, analytics_gateway: StorageGateway
) -> None:
    """Verify multiple functions in the same file are aggregated separately."""
    con = analytics_gateway.con

    _seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:test:func_a", "multi.py", "function", "func_a", HASH_A, 1, 5),
    )

    _seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:test:func_b", "multi.py", "function", "func_b", HASH_B, 10, 15),
    )

    _seed_coverage_lines_range(
        con, snapshot, CoverageRangeSeedData("multi.py", 1, 6, is_covered=True)
    )
    _seed_coverage_lines_range(
        con, snapshot, CoverageRangeSeedData("multi.py", 10, 16, is_covered=False)
    )

    cfg = ConfigBuilder.from_snapshot(
            repo=snapshot.repo, commit=snapshot.commit, repo_root=snapshot.repo_root
        ).coverage_analytics()
    compute_coverage_functions(analytics_gateway, cfg)

    result_a = con.execute(
        """
        SELECT coverage_ratio, tested
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_A],
    ).fetchone()

    assert result_a is not None
    assert result_a[0] == 1.0
    assert result_a[1] is True

    result_b = con.execute(
        """
        SELECT coverage_ratio, tested
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_B],
    ).fetchone()

    assert result_b is not None
    assert result_b[0] == 0.0
    assert result_b[1] is False


def test_idempotent_rerun_deletes_old_rows(
    snapshot: SnapshotRef, analytics_gateway: StorageGateway
) -> None:
    """Verify that re-running deletes existing rows for the same repo/commit."""
    con = analytics_gateway.con

    _seed_goid(
        con,
        snapshot,
        GoidSeedData(
            "urn:test:func_idem", "idempotent.py", "function", "idem_func", HASH_IDEM, 1, 3
        ),
    )

    _seed_coverage_line(
        con, snapshot, CoverageLineSeedData("idempotent.py", 1, is_executable=True, is_covered=True)
    )
    _seed_coverage_line(
        con, snapshot, CoverageLineSeedData("idempotent.py", 2, is_executable=True, is_covered=False)
    )
    _seed_coverage_line(
        con, snapshot, CoverageLineSeedData("idempotent.py", 3, is_executable=True, is_covered=False)
    )

    cfg = ConfigBuilder.from_snapshot(
            repo=snapshot.repo, commit=snapshot.commit, repo_root=snapshot.repo_root
        ).coverage_analytics()
    compute_coverage_functions(analytics_gateway, cfg)

    result1 = con.execute(
        """
        SELECT covered_lines FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_IDEM],
    ).fetchone()
    assert result1 is not None
    assert result1[0] == 1

    con.execute(
        """
        UPDATE analytics.coverage_lines
        SET is_covered = TRUE
        WHERE repo = ? AND commit = ?
        """,
        [snapshot.repo, snapshot.commit],
    )

    compute_coverage_functions(analytics_gateway, cfg)

    result2 = con.execute(
        """
        SELECT covered_lines FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_IDEM],
    ).fetchone()
    assert result2 is not None
    assert result2[0] == EXPECTED_COVERED_3

    count = con.execute(
        """
        SELECT COUNT(*) FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_IDEM],
    ).fetchone()
    assert count is not None
    assert count[0] == 1


def test_different_repos_isolated(
    snapshot: SnapshotRef, analytics_gateway: StorageGateway
) -> None:
    """Verify that different repos are isolated from each other."""
    con = analytics_gateway.con

    _seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:repo1:func", "module.py", "function", "func1", HASH_REPO1, 1, 5),
    )

    other_snapshot = SnapshotRef(repo="other-repo", commit="other-commit", repo_root=Path.cwd())
    _seed_goid(
        con,
        other_snapshot,
        GoidSeedData("urn:repo2:func", "module.py", "function", "func2", HASH_REPO2, 1, 5),
    )

    _seed_coverage_lines_range(con, snapshot, CoverageRangeSeedData("module.py", 1, 6))

    cfg = ConfigBuilder.from_snapshot(
            repo=snapshot.repo, commit=snapshot.commit, repo_root=snapshot.repo_root
        ).coverage_analytics()
    compute_coverage_functions(analytics_gateway, cfg)

    result = con.execute(
        """
        SELECT COUNT(*) FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ?
        """,
        [snapshot.repo, snapshot.commit],
    ).fetchone()
    assert result is not None
    assert result[0] == 1

    result_other = con.execute(
        """
        SELECT COUNT(*) FROM analytics.coverage_functions
        WHERE repo = 'other-repo'
        """
    ).fetchone()
    assert result_other is not None
    assert result_other[0] == 0


def test_function_with_null_end_line(
    snapshot: SnapshotRef, analytics_gateway: StorageGateway
) -> None:
    """Verify handling of GOIDs with NULL end_line (single-line functions)."""
    con = analytics_gateway.con

    _seed_goid(
        con,
        snapshot,
        GoidSeedData(
            "urn:test:single_line", "single.py", "function", "single_line_func", HASH_SINGLE, 5, None
        ),
    )

    _seed_coverage_line(
        con, snapshot, CoverageLineSeedData("single.py", 5, is_executable=True, is_covered=True)
    )

    cfg = ConfigBuilder.from_snapshot(
            repo=snapshot.repo, commit=snapshot.commit, repo_root=snapshot.repo_root
        ).coverage_analytics()
    compute_coverage_functions(analytics_gateway, cfg)

    result = con.execute(
        """
        SELECT executable_lines, covered_lines, coverage_ratio
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_SINGLE],
    ).fetchone()

    assert result is not None
    assert result[0] == 1
    assert result[1] == 1
    assert result[2] == 1.0


def test_realistic_module_with_mixed_coverage(
    snapshot: SnapshotRef, analytics_gateway: StorageGateway
) -> None:
    """Test a realistic module with multiple functions and varied coverage."""
    con = analytics_gateway.con
    calc_path = "utils/calculator.py"

    # Class (should not be included)
    _seed_goid(
        con,
        snapshot,
        GoidSeedData(
            "urn:utils:Calculator", calc_path, "class", "Calculator", HASH_CALC_CLASS, 1, 50
        ),
    )

    # Constructor method
    _seed_goid(
        con,
        snapshot,
        GoidSeedData(
            "urn:utils:Calculator.__init__",
            calc_path,
            "method",
            "Calculator.__init__",
            HASH_CALC_INIT,
            5,
            10,
        ),
    )

    # Add method (well tested)
    _seed_goid(
        con,
        snapshot,
        GoidSeedData(
            "urn:utils:Calculator.add", calc_path, "method", "Calculator.add", HASH_CALC_ADD, 12, 15
        ),
    )

    # Divide method (partially tested)
    _seed_goid(
        con,
        snapshot,
        GoidSeedData(
            "urn:utils:Calculator.divide",
            calc_path,
            "method",
            "Calculator.divide",
            HASH_CALC_DIVIDE,
            17,
            25,
        ),
    )

    # Helper function (untested)
    _seed_goid(
        con,
        snapshot,
        GoidSeedData(
            "urn:utils:_internal_helper",
            calc_path,
            "function",
            "_internal_helper",
            HASH_HELPER,
            30,
            35,
        ),
    )

    # __init__: fully covered
    _seed_coverage_lines_range(con, snapshot, CoverageRangeSeedData(calc_path, 5, 11))

    # add: fully covered
    _seed_coverage_lines_range(con, snapshot, CoverageRangeSeedData(calc_path, 12, 16))

    # divide: partially covered
    for line in range(17, 26):
        is_covered = line < DIVIDE_ERROR_LINE_START
        _seed_coverage_line(
            con,
            snapshot,
            CoverageLineSeedData(calc_path, line, is_executable=True, is_covered=is_covered),
        )

    # _internal_helper: not covered at all
    _seed_coverage_lines_range(
        con, snapshot, CoverageRangeSeedData(calc_path, 30, 36, is_covered=False)
    )

    cfg = ConfigBuilder.from_snapshot(
            repo=snapshot.repo, commit=snapshot.commit, repo_root=snapshot.repo_root
        ).coverage_analytics()
    compute_coverage_functions(analytics_gateway, cfg)

    results = con.execute(
        """
        SELECT function_goid_h128, coverage_ratio, tested, untested_reason
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ?
        ORDER BY function_goid_h128
        """,
        [snapshot.repo, snapshot.commit],
    ).fetchall()

    assert len(results) == EXPECTED_FUNCTIONS_COUNT

    results_by_hash = {int(row[0]): row for row in results}

    init_row = results_by_hash[HASH_CALC_INIT]
    assert init_row[1] == 1.0
    assert init_row[2] is True

    add_row = results_by_hash[HASH_CALC_ADD]
    assert add_row[1] == 1.0
    assert add_row[2] is True

    divide_row = results_by_hash[HASH_CALC_DIVIDE]
    assert DIVIDE_COVERAGE_LOW < divide_row[1] < DIVIDE_COVERAGE_HIGH
    assert divide_row[2] is True

    helper_row = results_by_hash[HASH_HELPER]
    assert helper_row[1] == 0.0
    assert helper_row[2] is False
    assert helper_row[3] == "no_tests"
