"""Combined coverage aggregation tests using shared coverage packs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.analytics.compute.coverage import compute_coverage_functions
from codeintel.analytics.testing.coverage import edges as coverage_edges
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_analytics import CoverageAnalyticsStepConfig, TestCoverageStepConfig
from tests._helpers import coverage_ready_context
from tests._helpers.assertions import (
    assert_coverage_function_row,
    expect_equal,
    expect_is_not_none,
    expect_length,
    expect_true,
)
from tests._helpers.assertions.logging_assertions import assert_logged
from tests._helpers.config_factory import coverage_analytics_cfg
from tests._helpers.context import TestContext
from tests._helpers.coverage import (
    CoverageLineSeedData,
    CoverageRangeSeedData,
    GoidSeedData,
    seed_coverage_line,
    seed_coverage_lines_range,
    seed_goid,
)

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


@pytest.fixture
def coverage_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Provide a coverage-ready TestContext with clean coverage tables.

    Yields
    ------
    TestContext
        Context with coverage and metrics seeds applied.
    """
    ctx = coverage_ready_context(tmp_path)
    con = ctx.con
    con.execute("DELETE FROM analytics.coverage_functions")
    con.execute("DELETE FROM analytics.coverage_lines")
    con.execute("DELETE FROM analytics.test_catalog")
    con.execute("DELETE FROM analytics.test_coverage_edges")
    con.execute(
        "DELETE FROM core.goids WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    )
    try:
        yield ctx
    finally:
        ctx.close()


def _coverage_config(snapshot: SnapshotRef) -> CoverageAnalyticsStepConfig:
    return coverage_analytics_cfg(snapshot)


def test_load_coverage_data_missing_file(
    tmp_path: Path,
    coverage_ctx: TestContext,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing coverage files should emit a warning and return None."""
    cfg = TestCoverageStepConfig(
        snapshot=coverage_ctx.to_snapshot_ref(),
        coverage_file=tmp_path / "nonexistent.coverage",
    )
    caplog.set_level("WARNING")

    result = coverage_edges.load_coverage_data(cfg)

    expect_true(result is None)
    assert_logged(
        caplog.records,
        level="WARNING",
        containing="Coverage file",
    )


def test_empty_goids_produces_no_rows(coverage_ctx: TestContext) -> None:
    """Verify that no GOIDs results in no coverage function rows."""
    cfg = _coverage_config(coverage_ctx.to_snapshot_ref())
    compute_coverage_functions(coverage_ctx.gateway, cfg)

    count = coverage_ctx.con.execute(
        """
        SELECT COUNT(*) FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ?
        """,
        [coverage_ctx.repo, coverage_ctx.commit],
    ).fetchone()

    count = expect_is_not_none(count)
    expect_equal(count[0], 0)


def test_single_function_fully_covered(coverage_ctx: TestContext) -> None:
    """Verify coverage ratio for a fully covered function."""
    con = coverage_ctx.con
    snapshot = coverage_ctx.to_snapshot_ref()

    seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:test:func1", "module.py", "function", "my_function", HASH_1, 1, 5),
    )
    seed_coverage_lines_range(con, snapshot, CoverageRangeSeedData("module.py", 1, 6))

    cfg = _coverage_config(snapshot)
    compute_coverage_functions(coverage_ctx.gateway, cfg)

    assert_coverage_function_row(
        con,
        snapshot=snapshot,
        goid=HASH_1,
        executable=EXPECTED_EXECUTABLE_5,
        covered=EXPECTED_EXECUTABLE_5,
        ratio=1.0,
        tested=True,
        untested_reason="",
    )


def test_single_function_partially_covered(coverage_ctx: TestContext) -> None:
    """Verify coverage ratio for a partially covered function."""
    con = coverage_ctx.con
    snapshot = coverage_ctx.to_snapshot_ref()

    seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:test:func2", "partial.py", "function", "partial_func", HASH_2, 1, 4),
    )

    seed_coverage_line(
        con, snapshot, CoverageLineSeedData("partial.py", 1, is_executable=True, is_covered=True)
    )
    seed_coverage_line(
        con, snapshot, CoverageLineSeedData("partial.py", 2, is_executable=True, is_covered=True)
    )
    seed_coverage_line(
        con, snapshot, CoverageLineSeedData("partial.py", 3, is_executable=True, is_covered=False)
    )
    seed_coverage_line(
        con, snapshot, CoverageLineSeedData("partial.py", 4, is_executable=False, is_covered=False)
    )

    cfg = _coverage_config(snapshot)
    compute_coverage_functions(coverage_ctx.gateway, cfg)

    assert_coverage_function_row(
        con,
        snapshot=snapshot,
        goid=HASH_2,
        executable=EXPECTED_EXECUTABLE_3,
        covered=EXPECTED_COVERED_2,
        ratio=0.6666666666666666,
        tested=True,
        untested_reason="",
    )


def test_function_no_coverage_data(coverage_ctx: TestContext) -> None:
    """Verify function with no coverage lines is marked untested."""
    con = coverage_ctx.con
    snapshot = coverage_ctx.to_snapshot_ref()

    seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:test:func3", "uncovered.py", "function", "uncovered_func", HASH_3, 1, 10),
    )

    cfg = _coverage_config(snapshot)
    compute_coverage_functions(coverage_ctx.gateway, cfg)

    assert_coverage_function_row(
        con,
        snapshot=snapshot,
        goid=HASH_3,
        executable=0,
        covered=0,
        ratio=None,
        tested=False,
        untested_reason="no_executable_code",
    )


def test_function_with_executable_but_no_covered_lines(coverage_ctx: TestContext) -> None:
    """Verify function with executable but zero covered lines."""
    con = coverage_ctx.con
    snapshot = coverage_ctx.to_snapshot_ref()

    seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:test:func4", "no_tests.py", "function", "no_tests_func", HASH_4, 5, 10),
    )

    seed_coverage_lines_range(
        con, snapshot, CoverageRangeSeedData("no_tests.py", 5, 11, is_covered=False)
    )

    cfg = _coverage_config(snapshot)
    compute_coverage_functions(coverage_ctx.gateway, cfg)

    assert_coverage_function_row(
        con,
        snapshot=snapshot,
        goid=HASH_4,
        executable=EXPECTED_EXECUTABLE_6,
        covered=0,
        ratio=0.0,
        tested=False,
        untested_reason="no_tests",
    )


def test_method_kind_included(coverage_ctx: TestContext) -> None:
    """Verify that 'method' kind GOIDs are included in aggregation."""
    con = coverage_ctx.con
    snapshot = coverage_ctx.to_snapshot_ref()

    seed_goid(
        con,
        snapshot,
        GoidSeedData(
            "urn:test:method1", "class_mod.py", "method", "MyClass.my_method", HASH_METHOD, 10, 15
        ),
    )
    seed_coverage_lines_range(con, snapshot, CoverageRangeSeedData("class_mod.py", 10, 16))

    cfg = _coverage_config(snapshot)
    compute_coverage_functions(coverage_ctx.gateway, cfg)

    result = coverage_ctx.con.execute(
        """
        SELECT kind, coverage_ratio, tested
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_METHOD],
    ).fetchone()

    result = expect_is_not_none(result)
    expect_equal(result[0], "method")
    expect_equal(result[1], 1.0)
    expect_true(result[2] is True)


def test_class_kind_excluded(coverage_ctx: TestContext) -> None:
    """Verify that 'class' kind GOIDs are NOT included in aggregation."""
    con = coverage_ctx.con
    snapshot = coverage_ctx.to_snapshot_ref()

    seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:test:class1", "class_def.py", "class", "MyClass", HASH_CLASS, 1, 50),
    )
    seed_coverage_lines_range(con, snapshot, CoverageRangeSeedData("class_def.py", 1, 51))

    cfg = _coverage_config(snapshot)
    compute_coverage_functions(coverage_ctx.gateway, cfg)

    result = coverage_ctx.con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_CLASS],
    ).fetchone()

    result = expect_is_not_none(result)
    expect_equal(result[0], 0)


def test_multiple_functions_same_file(coverage_ctx: TestContext) -> None:
    """Verify multiple functions in the same file are aggregated separately."""
    con = coverage_ctx.con
    snapshot = coverage_ctx.to_snapshot_ref()

    seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:test:func_a", "multi.py", "function", "func_a", HASH_A, 1, 5),
    )
    seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:test:func_b", "multi.py", "function", "func_b", HASH_B, 10, 15),
    )
    seed_coverage_lines_range(
        con, snapshot, CoverageRangeSeedData("multi.py", 1, 6, is_covered=True)
    )
    seed_coverage_lines_range(
        con, snapshot, CoverageRangeSeedData("multi.py", 10, 16, is_covered=False)
    )

    cfg = _coverage_config(snapshot)
    compute_coverage_functions(coverage_ctx.gateway, cfg)

    result_a = coverage_ctx.con.execute(
        """
        SELECT coverage_ratio, tested
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_A],
    ).fetchone()
    result_a = expect_is_not_none(result_a)
    expect_equal(result_a[0], 1.0)
    expect_true(result_a[1] is True)

    result_b = coverage_ctx.con.execute(
        """
        SELECT coverage_ratio, tested
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_B],
    ).fetchone()
    result_b = expect_is_not_none(result_b)
    expect_equal(result_b[0], 0.0)
    expect_true(result_b[1] is False)


def test_idempotent_rerun_deletes_old_rows(coverage_ctx: TestContext) -> None:
    """Verify that re-running deletes existing rows for the same repo/commit."""
    con = coverage_ctx.con
    snapshot = coverage_ctx.to_snapshot_ref()

    seed_goid(
        con,
        snapshot,
        GoidSeedData(
            "urn:test:func_idem", "idempotent.py", "function", "idem_func", HASH_IDEM, 1, 3
        ),
    )
    seed_coverage_line(
        con, snapshot, CoverageLineSeedData("idempotent.py", 1, is_executable=True, is_covered=True)
    )
    seed_coverage_line(
        con,
        snapshot,
        CoverageLineSeedData("idempotent.py", 2, is_executable=True, is_covered=False),
    )
    seed_coverage_line(
        con,
        snapshot,
        CoverageLineSeedData("idempotent.py", 3, is_executable=True, is_covered=False),
    )

    cfg = _coverage_config(snapshot)
    compute_coverage_functions(coverage_ctx.gateway, cfg)

    result1 = coverage_ctx.con.execute(
        """
        SELECT covered_lines FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_IDEM],
    ).fetchone()
    result1 = expect_is_not_none(result1)
    expect_equal(result1[0], 1)

    coverage_ctx.con.execute(
        """
        UPDATE analytics.coverage_lines
        SET is_covered = TRUE
        WHERE repo = ? AND commit = ?
        """,
        [snapshot.repo, snapshot.commit],
    )

    compute_coverage_functions(coverage_ctx.gateway, cfg)

    result2 = coverage_ctx.con.execute(
        """
        SELECT covered_lines FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_IDEM],
    ).fetchone()
    result2 = expect_is_not_none(result2)
    expect_equal(result2[0], EXPECTED_COVERED_3)

    count = coverage_ctx.con.execute(
        """
        SELECT COUNT(*) FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_IDEM],
    ).fetchone()
    count = expect_is_not_none(count)
    expect_equal(count[0], 1)


def test_different_repos_isolated(coverage_ctx: TestContext) -> None:
    """Verify that different repos are isolated from each other."""
    con = coverage_ctx.con
    snapshot = coverage_ctx.to_snapshot_ref()
    other_snapshot = SnapshotRef(
        repo="other-repo",
        commit="other-commit",
        repo_root=snapshot.repo_root,
    )

    seed_goid(
        con,
        snapshot,
        GoidSeedData("urn:repo1:func", "module.py", "function", "func1", HASH_REPO1, 1, 5),
    )
    seed_goid(
        con,
        other_snapshot,
        GoidSeedData("urn:repo2:func", "module.py", "function", "func2", HASH_REPO2, 1, 5),
    )
    seed_coverage_lines_range(con, snapshot, CoverageRangeSeedData("module.py", 1, 6))

    cfg = _coverage_config(snapshot)
    compute_coverage_functions(coverage_ctx.gateway, cfg)

    result = con.execute(
        """
        SELECT COUNT(*) FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ?
        """,
        [snapshot.repo, snapshot.commit],
    ).fetchone()
    result = expect_is_not_none(result)
    expect_equal(result[0], 1)

    result_other = con.execute(
        """
        SELECT COUNT(*) FROM analytics.coverage_functions
        WHERE repo = 'other-repo'
        """
    ).fetchone()
    result_other = expect_is_not_none(result_other)
    expect_equal(result_other[0], 0)


def test_function_with_null_end_line(coverage_ctx: TestContext) -> None:
    """Verify handling of GOIDs with NULL end_line (single-line functions)."""
    con = coverage_ctx.con
    snapshot = coverage_ctx.to_snapshot_ref()

    seed_goid(
        con,
        snapshot,
        GoidSeedData(
            "urn:test:single_line",
            "single.py",
            "function",
            "single_line_func",
            HASH_SINGLE,
            5,
            None,
        ),
    )
    seed_coverage_line(
        con, snapshot, CoverageLineSeedData("single.py", 5, is_executable=True, is_covered=True)
    )

    cfg = _coverage_config(snapshot)
    compute_coverage_functions(coverage_ctx.gateway, cfg)

    result = coverage_ctx.con.execute(
        """
        SELECT executable_lines, covered_lines, coverage_ratio
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [snapshot.repo, snapshot.commit, HASH_SINGLE],
    ).fetchone()

    result = expect_is_not_none(result)
    expect_equal(result[0], 1)
    expect_equal(result[1], 1)
    expect_equal(result[2], 1.0)


def test_realistic_module_with_mixed_coverage(coverage_ctx: TestContext) -> None:
    """Test a realistic module with multiple functions and varied coverage."""
    con = coverage_ctx.con
    snapshot = coverage_ctx.to_snapshot_ref()
    calc_path = "utils/calculator.py"

    seed_goid(
        con,
        snapshot,
        GoidSeedData(
            "urn:utils:Calculator", calc_path, "class", "Calculator", HASH_CALC_CLASS, 1, 50
        ),
    )
    seed_goid(
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
    seed_goid(
        con,
        snapshot,
        GoidSeedData(
            "urn:utils:Calculator.add", calc_path, "method", "Calculator.add", HASH_CALC_ADD, 12, 15
        ),
    )
    seed_goid(
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
    seed_goid(
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

    seed_coverage_lines_range(con, snapshot, CoverageRangeSeedData(calc_path, 5, 11))
    seed_coverage_lines_range(con, snapshot, CoverageRangeSeedData(calc_path, 12, 16))
    for line in range(17, 26):
        seed_coverage_line(
            con,
            snapshot,
            CoverageLineSeedData(
                calc_path,
                line,
                is_executable=True,
                is_covered=line < DIVIDE_ERROR_LINE_START,
            ),
        )
    seed_coverage_lines_range(
        con, snapshot, CoverageRangeSeedData(calc_path, 30, 36, is_covered=False)
    )

    cfg = _coverage_config(snapshot)
    compute_coverage_functions(coverage_ctx.gateway, cfg)

    results = con.execute(
        """
        SELECT function_goid_h128, coverage_ratio, tested, untested_reason
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ?
        ORDER BY function_goid_h128
        """,
        [snapshot.repo, snapshot.commit],
    ).fetchall()

    expect_length(results, EXPECTED_FUNCTIONS_COUNT)
    results_by_hash = {int(row[0]): row for row in results}

    init_row = results_by_hash[HASH_CALC_INIT]
    expect_equal(init_row[1], 1.0)
    expect_true(init_row[2] is True)

    add_row = results_by_hash[HASH_CALC_ADD]
    expect_equal(add_row[1], 1.0)
    expect_true(add_row[2] is True)

    divide_row = results_by_hash[HASH_CALC_DIVIDE]
    expect_true(DIVIDE_COVERAGE_LOW < divide_row[1] < DIVIDE_COVERAGE_HIGH)
    expect_true(divide_row[2] is True)

    helper_row = results_by_hash[HASH_HELPER]
    expect_equal(helper_row[1], 0.0)
    expect_true(helper_row[2] is False)
    expect_equal(helper_row[3], "no_tests")


def test_compute_coverage_functions_smoke(coverage_ctx: TestContext) -> None:
    """Aggregate executable and covered lines into coverage_functions."""
    snapshot = coverage_ctx.to_snapshot_ref()
    con = coverage_ctx.con
    seed_goid(
        con,
        snapshot,
        GoidSeedData(
            urn="urn:func",
            rel_path="pkg/mod.py",
            kind="function",
            qualname="pkg.mod.fn",
            goid_h128=1,
            start_line=1,
            end_line=3,
        ),
    )
    seed_goid(
        con,
        snapshot,
        GoidSeedData(
            urn="urn:method",
            rel_path="pkg/mod.py",
            kind="method",
            qualname="pkg.mod.method",
            goid_h128=2,
            start_line=10,
            end_line=12,
        ),
    )
    seed_coverage_line(
        con,
        snapshot,
        CoverageLineSeedData("pkg/mod.py", 1, is_executable=True, is_covered=True),
    )
    seed_coverage_line(
        con,
        snapshot,
        CoverageLineSeedData("pkg/mod.py", 2, is_executable=True, is_covered=False),
    )
    seed_coverage_line(
        con,
        snapshot,
        CoverageLineSeedData("pkg/mod.py", 3, is_executable=False, is_covered=False),
    )

    cfg = _coverage_config(snapshot)
    compute_coverage_functions(coverage_ctx.gateway, cfg)

    rows_raw = con.execute(
        """
        SELECT function_goid_h128, executable_lines, covered_lines,
               coverage_ratio, tested, untested_reason
        FROM analytics.coverage_functions
        WHERE repo = ? AND commit = ?
        ORDER BY function_goid_h128
        """,
        [snapshot.repo, snapshot.commit],
    ).fetchall()

    rows = [
        (int(goid), exec_lines, covered, ratio, tested, reason)
        for goid, exec_lines, covered, ratio, tested, reason in rows_raw
    ]
    expect_equal(
        rows,
        [
            (1, 2, 1, pytest.approx(0.5), True, ""),
            (2, 0, 0, None, False, "no_executable_code"),
        ],
    )
