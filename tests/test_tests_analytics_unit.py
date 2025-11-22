"""Unit tests for analytics.tests_analytics helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest
from coverage import Coverage

from codeintel.analytics.tests_analytics import (
    EdgeContext,
    FunctionRow,
    TestCoverageConfig,
    backfill_test_goids_for_catalog,
    build_edges_for_file_for_tests,
    compute_test_coverage_edges,
)

cast("Any", TestCoverageConfig).__test__ = False  # prevent pytest from collecting the dataclass


def test_backfill_test_goids_updates_catalog() -> None:
    """Ensure test_catalog entries receive GOIDs and URNs when matched to GOIDs."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA core;")
    con.execute("CREATE SCHEMA analytics;")
    con.execute(
        """
        CREATE TABLE core.goids (
            goid_h128 DECIMAL(38,0),
            urn TEXT,
            repo TEXT,
            commit TEXT,
            rel_path TEXT,
            qualname TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.test_catalog (
            test_id TEXT,
            rel_path TEXT,
            qualname TEXT,
            repo TEXT,
            commit TEXT,
            test_goid_h128 DECIMAL(38,0),
            urn TEXT
        );
        """
    )

    cfg = TestCoverageConfig(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=Path.cwd(),
    )

    con.execute(
        """
        INSERT INTO core.goids (goid_h128, urn, repo, commit, rel_path, qualname)
        VALUES
            (1, 'goid:demo/repo#python:function:test_mod.test_func', ?, ?, 'tests/test_mod.py', 'tests.test_mod.test_func'),
            (2, 'goid:demo/repo#python:function:other', ?, ?, 'tests/test_mod.py', 'tests.test_mod.other')
        """,
        [cfg.repo, cfg.commit, cfg.repo, cfg.commit],
    )
    con.execute(
        """
        INSERT INTO analytics.test_catalog (test_id, rel_path, qualname, repo, commit)
        VALUES ('tests/test_mod.py::test_func', 'tests/test_mod.py', 'tests.test_mod.test_func', ?, ?)
        """,
        [cfg.repo, cfg.commit],
    )

    goid_map, urn_map = backfill_test_goids_for_catalog(con, cfg)

    expected_goid_map = {"tests/test_mod.py::test_func": 1}
    expected_urn_map = {
        "tests/test_mod.py::test_func": "goid:demo/repo#python:function:test_mod.test_func"
    }
    if goid_map != expected_goid_map:
        pytest.fail(f"goid map mismatch: {goid_map}")
    if urn_map != expected_urn_map:
        pytest.fail(f"urn map mismatch: {urn_map}")

    row = con.execute(
        """
        SELECT test_goid_h128, urn
        FROM analytics.test_catalog
        WHERE test_id = 'tests/test_mod.py::test_func'
        """
    ).fetchone()
    if row is None:
        pytest.fail("Row missing in test_catalog")
    if int(row[0]) != 1:
        pytest.fail(f"Unexpected test_goid_h128 {row[0]}")
    if row[1] != "goid:demo/repo#python:function:test_mod.test_func":
        pytest.fail(f"Unexpected urn {row[1]}")


def test_edges_for_file_uses_test_meta() -> None:
    """_edges_for_file should carry through mapped test GOIDs/URNs and coverage ratios."""
    file_funcs: list[FunctionRow] = [
        FunctionRow(
            start_line=1,
            end_line=2,
            goid_h128=123,
            urn="goid:demo/repo#python:function:q",
            qualname="pkg.mod.func",
            rel_path="tests/test_mod.py",
        )
    ]
    statements_set = {1, 2}
    contexts_by_lineno = {1: {"tests/test_mod.py::test_func"}, 2: {"tests/test_mod.py::test_func"}}
    ctx = EdgeContext(
        status_by_test={"tests/test_mod.py::test_func": "passed"},
        cfg=TestCoverageConfig(repo="demo/repo", commit="deadbeef", repo_root=Path.cwd()),
        now=datetime(2024, 1, 1, tzinfo=UTC),
        test_meta_by_id={
            "tests/test_mod.py::test_func": (456, "goid:demo/repo#python:function:test")
        },
    )

    edges = build_edges_for_file_for_tests(
        file_funcs=file_funcs,
        statements_set=statements_set,
        contexts_by_lineno=contexts_by_lineno,
        rel_path="tests/test_mod.py",
        ctx=ctx,
    )

    if len(edges) != 1:
        pytest.fail(f"Expected one edge, got {len(edges)}")

    edge = edges[0]
    expected_test_goid = 456
    expected_urn = "goid:demo/repo#python:function:test"
    expected_cov_ratio = 1.0
    if edge["test_goid_h128"] != expected_test_goid:
        pytest.fail(f"Expected test_goid_h128 {expected_test_goid}, got {edge['test_goid_h128']}")
    if edge["urn"] != expected_urn:
        pytest.fail(f"Expected URN {expected_urn}, got {edge['urn']}")
    if edge["coverage_ratio"] != expected_cov_ratio:
        pytest.fail(f"Expected coverage_ratio {expected_cov_ratio}, got {edge['coverage_ratio']}")


def test_compute_test_coverage_edges_with_fake_coverage(tmp_path: Path) -> None:
    """compute_test_coverage_edges should join coverage contexts with test GOIDs."""
    repo_root = tmp_path / "repo"
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    target_file = pkg_dir / "mod.py"
    target_file.write_text("def func():\n    return 1\n", encoding="utf-8")

    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA core;")
    con.execute("CREATE SCHEMA analytics;")
    con.execute(
        """
        CREATE TABLE core.goids (
            goid_h128 DECIMAL(38,0),
            urn TEXT,
            repo TEXT,
            commit TEXT,
            rel_path TEXT,
            language TEXT,
            kind TEXT,
            qualname TEXT,
            start_line INTEGER,
            end_line INTEGER
        );
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.test_catalog (
            test_id TEXT,
            rel_path TEXT,
            qualname TEXT,
            repo TEXT,
            commit TEXT,
            test_goid_h128 DECIMAL(38,0),
            urn TEXT,
            status TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.test_coverage_edges (
            test_id TEXT,
            test_goid_h128 DECIMAL(38,0),
            function_goid_h128 DECIMAL(38,0),
            urn TEXT,
            repo TEXT,
            commit TEXT,
            rel_path TEXT,
            qualname TEXT,
            covered_lines INTEGER,
            executable_lines INTEGER,
            coverage_ratio DOUBLE,
            last_status TEXT,
            created_at TIMESTAMP
        );
        """
    )

    cfg = TestCoverageConfig(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=repo_root,
        coverage_file=tmp_path / ".coverage",  # unused with monkeypatch
    )

    con.execute(
        """
        INSERT INTO core.goids (goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line)
        VALUES
            (1, 'goid:demo/repo#python:function:pkg.mod.func', ?, ?, 'pkg/mod.py', 'python', 'function', 'pkg.mod.func', 1, 2),
            (99, 'goid:demo/repo#python:function:pkg.mod.test_func', ?, ?, 'pkg/mod.py', 'python', 'test', 'pkg.mod.test_func', 1, 2)
        """,
        [cfg.repo, cfg.commit, cfg.repo, cfg.commit],
    )
    con.execute(
        """
        INSERT INTO analytics.test_catalog (test_id, rel_path, qualname, repo, commit, status)
        VALUES ('pkg/mod.py::test_func', 'pkg/mod.py', 'pkg.mod.test_func', ?, ?, 'passed')
        """,
        [cfg.repo, cfg.commit],
    )

    class _FakeData:
        def __init__(self, paths: list[str], contexts: dict[int, set[str]]) -> None:
            self._paths = paths
            self._contexts = contexts

        def measured_files(self) -> list[str]:
            return self._paths

        def contexts_by_lineno(self, path: str) -> dict[int, set[str]]:
            return self._contexts if path in self._paths else {}

    class _FakeCoverage:
        def __init__(self, path: Path) -> None:
            self.path = path
            self._statements = [1, 2]
            self._contexts = {1: {"pkg/mod.py::test_func"}, 2: {"pkg/mod.py::test_func"}}

        def analysis2(self, path: str) -> tuple[str, list[int], list[int], list[int], list[int]]:
            return path, self._statements, [], [], self._statements

        def get_data(self) -> _FakeData:
            return _FakeData([str(target_file)], self._contexts)

    def fake_loader(cfg_arg: TestCoverageConfig) -> Coverage:
        return _FakeCoverage(cfg_arg.repo_root)  # type: ignore[return-value]

    compute_test_coverage_edges(con, cfg, coverage_loader=fake_loader)

    _assert_single_edge(con)


def _assert_single_edge(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        "SELECT test_goid_h128, coverage_ratio, last_status FROM analytics.test_coverage_edges"
    ).fetchall()
    if len(rows) != 1:
        pytest.fail(f"Expected 1 edge row, got {len(rows)}")
    test_goid, cov_ratio, status = rows[0]
    if test_goid is None:
        pytest.fail("Expected test_goid_h128 to be populated")
    tolerance = 1e-6
    if abs(float(cov_ratio) - 1.0) > tolerance:
        pytest.fail(f"Unexpected coverage_ratio {cov_ratio}")
    if status != "passed":
        pytest.fail(f"Unexpected last_status {status}")
