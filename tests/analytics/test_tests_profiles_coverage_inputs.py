"""Richer coverage input tests using in-memory DuckDB fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from codeintel.analytics.tests_profiles import coverage_inputs
from codeintel.config import BehavioralCoverageStepConfig, TestProfileStepConfig
from codeintel.config.primitives import SnapshotRef

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


def _snapshot_cfg() -> tuple[TestProfileStepConfig, BehavioralCoverageStepConfig]:
    snapshot = SnapshotRef(repo="r", commit="c", repo_root=Path.cwd())
    return TestProfileStepConfig(snapshot=snapshot), BehavioralCoverageStepConfig(snapshot=snapshot)


def _seed_sample_data(con: DuckDBPyConnection) -> None:
    edges = [
        ("t1", 1, "mod.a", 5, 10, "r", "c", "a.py", "A::t1a"),
        ("t1", 2, "mod.b", 8, 10, "r", "c", "b.py", "B::t1b"),
        ("t2", 2, "mod.b", 6, 10, "r", "c", "b.py", "B::t2"),
    ]
    con.executemany(
        """
        INSERT INTO analytics.test_coverage_edges
        (test_id, function_goid_h128, module, covered_lines, executable_lines, repo, commit, rel_path, qualname)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        edges,
    )
    catalog = [
        ("t1", "r", "c", "passed", 500.0, False),
        ("t2", "r", "c", "failed", 2500.0, True),
    ]
    con.executemany(
        "INSERT INTO analytics.test_catalog VALUES (?, ?, ?, ?, ?, ?)",
        catalog,
    )
    subsystems = [
        ("mod.a", "subA"),
        ("mod.b", "subB"),
    ]
    con.executemany(
        "INSERT INTO analytics.subsystem_modules VALUES (?, ?, 'r', 'c')",
        subsystems,
    )
    con.executemany(
        "INSERT INTO analytics.subsystems VALUES (?, ?, ?, 'r', 'c')",
        [("subA", "Subsystem A", 0.2), ("subB", "Subsystem B", 0.8)],
    )


def _aggregate_functions(con: DuckDBPyConnection, repo: str, commit: str) -> dict[str, Any]:
    """
    Compute function coverage summaries from the fixture tables.

    Parameters
    ----------
    con
        DuckDB connection.
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    dict[str, Any]
        Mapping of test_id to function coverage summary.
    """
    rows = con.execute(
        """
        SELECT
            test_id,
            function_goid_h128,
            covered_lines,
            executable_lines
        FROM analytics.test_coverage_edges
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()

    by_test: dict[str, list[tuple[int, int, int]]] = {}
    for test_id, func_goid, covered, executable in rows:
        by_test.setdefault(test_id, []).append((int(func_goid), int(covered), int(executable or 0)))

    primary_threshold = 0.4
    result: dict[str, Any] = {}
    for test_id, funcs in by_test.items():
        functions = [
            {"function_goid_h128": goid, "covered_lines": cov, "executable_lines": exe}
            for goid, cov, exe in funcs
        ]
        primary = [goid for goid, cov, exe in funcs if exe > 0 and (cov / exe) >= primary_threshold]
        result[test_id] = {"functions": functions, "count": len(functions), "primary": primary}
    return result


def _aggregate_subsystems(con: DuckDBPyConnection, repo: str, commit: str) -> dict[str, Any]:
    """
    Compute subsystem coverage summaries from the fixture tables.

    Parameters
    ----------
    con
        DuckDB connection.
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    dict[str, Any]
        Mapping of test_id to subsystem coverage summary.
    """
    rows = con.execute(
        """
        SELECT
            e.test_id,
            sm.subsystem_id
        FROM analytics.test_coverage_edges AS e
        LEFT JOIN analytics.subsystem_modules AS sm
          ON sm.module = e.module AND sm.repo = e.repo AND sm.commit = e.commit
        WHERE e.repo = ? AND e.commit = ?
        """,
        [repo, commit],
    ).fetchall()

    by_test: dict[str, list[str | None]] = {}
    for test_id, subsystem_id in rows:
        by_test.setdefault(test_id, []).append(subsystem_id)

    result: dict[str, Any] = {}
    for test_id, subs in by_test.items():
        subsystems = [{"subsystem_id": sid} for sid in subs if sid is not None]
        primary_subsystem = subsystems[0]["subsystem_id"] if subsystems else None
        result[test_id] = {
            "subsystems": subsystems,
            "count": len(subsystems),
            "primary_subsystem_id": primary_subsystem,
            "max_risk_score": None,
        }
    return result


def test_aggregate_test_coverage_by_function_in_memory(
    coverage_profiles_conn: DuckDBPyConnection,
) -> None:
    """Validate function coverage aggregation against small DuckDB fixture."""
    _seed_sample_data(coverage_profiles_conn)
    test_cfg, _ = _snapshot_cfg()
    result = coverage_inputs.aggregate_test_coverage_by_function(
        coverage_profiles_conn, test_cfg, loader=_aggregate_functions
    )
    if set(result.keys()) != {"t1", "t2"}:
        pytest.fail("Expected both tests t1 and t2 in coverage results.")
    t1 = result["t1"]
    t2 = result["t2"]
    expected_t1_count = 2
    expected_t2_count = 1
    if t1.count != expected_t1_count or t2.count != expected_t2_count:
        pytest.fail("Function counts did not match expectations.")
    primary_expected = {1, 2}
    if set(t1.primary) != primary_expected or t2.primary != [2]:
        pytest.fail("Primary function selection did not match expectations.")


def test_aggregate_test_coverage_by_subsystem_in_memory(
    coverage_profiles_conn: DuckDBPyConnection,
) -> None:
    """Validate subsystem coverage aggregation against small DuckDB fixture."""
    _seed_sample_data(coverage_profiles_conn)
    _, beh_cfg = _snapshot_cfg()
    result = coverage_inputs.aggregate_test_coverage_by_subsystem(
        coverage_profiles_conn, beh_cfg, loader=_aggregate_subsystems
    )
    if set(result.keys()) != {"t1", "t2"}:
        pytest.fail("Expected both tests t1 and t2 in subsystem results.")
    t1 = result["t1"]
    t2 = result["t2"]
    expected_t1_count = 2
    expected_t2_count = 1
    if t1.count != expected_t1_count or t2.count != expected_t2_count:
        pytest.fail("Subsystem counts did not match expectations.")
    primary_subs = {"subA", "subB"}
    if t1.primary_subsystem_id not in primary_subs:
        pytest.fail("Primary subsystem for t1 not in expected set.")
    if t2.primary_subsystem_id != "subB":
        pytest.fail("Primary subsystem for t2 did not match expectations.")
