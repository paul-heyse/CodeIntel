"""Unit tests for analytics.tests.coverage_edges helpers."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from coverage import Coverage

from codeintel.analytics.tests.coverage_edges import (
    EdgeContext,
    FunctionRow,
    backfill_test_goids_for_catalog,
    build_edges_for_file_for_tests,
    compute_test_coverage_edges,
)
from codeintel.config import ConfigBuilder, TestCoverageStepConfig
from codeintel.storage.gateway import DuckDBConnection
from tests._helpers.fixtures import (
    ProvisionOptions,
    provision_existing_repo,
    provision_graph_ready_repo,
)
from tests._helpers.tooling import generate_coverage_for_function


def _insert_goids(con: DuckDBConnection, cfg: TestCoverageStepConfig) -> None:
    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO core.goids (
            goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, created_at
        )
        VALUES
            (1, 'goid:demo/repo#python:function:test_mod.test_func', ?, ?, 'tests/test_mod.py', 'python', 'function', 'tests.test_mod.test_func', 1, 2, ?),
            (2, 'goid:demo/repo#python:function:other', ?, ?, 'tests/test_mod.py', 'python', 'function', 'tests.test_mod.other', 1, 2, ?)
        """,
        [cfg.repo, cfg.commit, now, cfg.repo, cfg.commit, now],
    )


def test_backfill_test_goids_updates_catalog() -> None:
    """Ensure test_catalog entries receive GOIDs and URNs when matched to GOIDs."""
    repo_root = Path(tempfile.mkdtemp())
    ctx = provision_graph_ready_repo(
        repo_root,
        repo="demo/repo",
        commit="deadbeef",
        options=ProvisionOptions(include_seed_goid=False, build_graph_metrics=True),
    )
    gateway = ctx.gateway
    con = gateway.con

    builder = ConfigBuilder.from_snapshot(
        repo=ctx.repo,
        commit=ctx.commit,
        repo_root=repo_root,
        build_dir=ctx.build_dir,
    )
    cfg = builder.test_coverage()

    _insert_goids(con, cfg)
    con.execute(
        """
        INSERT INTO analytics.test_catalog (test_id, rel_path, qualname, repo, commit, status, created_at)
        VALUES ('tests/test_mod.py::test_func', 'tests/test_mod.py', 'tests.test_mod.test_func', ?, ?, 'passed', ?)
        """,
        [cfg.repo, cfg.commit, datetime.now(UTC)],
    )

    goid_map, urn_map = backfill_test_goids_for_catalog(gateway, cfg)

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
    temp_root = Path(tempfile.mkdtemp())
    cfg = ConfigBuilder.from_snapshot(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=temp_root,
    ).test_coverage()
    ctx = EdgeContext(
        status_by_test={"tests/test_mod.py::test_func": "passed"},
        cfg=cfg,
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


def test_compute_test_coverage_edges_with_real_coverage(tmp_path: Path) -> None:
    """compute_test_coverage_edges should join coverage contexts with test GOIDs."""
    repo_root = tmp_path / "repo"
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    target_file = pkg_dir / "mod.py"
    target_file.write_text("def func():\n    return 1\n", encoding="utf-8")

    gateway = provision_existing_repo(
        repo_root,
        repo="demo/repo",
        commit="deadbeef",
        options=ProvisionOptions(include_seed_goid=False, build_graph_metrics=True),
    ).gateway
    con = gateway.con

    coverage_artifact = generate_coverage_for_function(
        repo_root=repo_root,
        module_import="pkg.mod",
        function_name="func",
        test_id="pkg/mod.py::test_func",
        coverage_file=tmp_path / ".coverage",
    )
    cfg = ConfigBuilder.from_snapshot(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=repo_root,
    ).test_coverage(coverage_file=coverage_artifact.coverage_file)

    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO core.goids (goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line, created_at)
        VALUES
            (1, 'goid:demo/repo#python:function:pkg.mod.func', ?, ?, 'pkg/mod.py', 'python', 'function', 'pkg.mod.func', 1, 2, ?),
            (99, 'goid:demo/repo#python:function:pkg.mod.test_func', ?, ?, 'pkg/mod.py', 'python', 'test', 'pkg.mod.test_func', 1, 2, ?)
        """,
        [cfg.repo, cfg.commit, now, cfg.repo, cfg.commit, now],
    )
    con.execute(
        """
        INSERT INTO analytics.test_catalog (test_id, rel_path, qualname, repo, commit, status, created_at)
        VALUES ('pkg/mod.py::test_func', 'pkg/mod.py', 'pkg.mod.test_func', ?, ?, 'passed', ?)
        """,
        [cfg.repo, cfg.commit, now],
    )

    compute_test_coverage_edges(gateway, cfg)
    _assert_single_edge(con)


def _assert_single_edge(con: DuckDBConnection) -> None:
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


def test_compute_test_coverage_edges_respects_injected_loader(tmp_path: Path) -> None:
    """compute_test_coverage_edges should call injected loader when provided."""
    repo_root = tmp_path / "repo"
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    target_file = pkg_dir / "mod.py"
    target_file.write_text("def func():\n    return 1\n", encoding="utf-8")

    gateway = provision_existing_repo(
        repo_root,
        repo="demo/repo",
        commit="deadbeef",
        options=ProvisionOptions(include_seed_goid=False, build_graph_metrics=True),
    ).gateway
    con = gateway.con

    builder = ConfigBuilder.from_snapshot(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=repo_root,
    )
    cfg = builder.test_coverage()

    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO core.goids (goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line, created_at)
        VALUES
            (1, 'goid:demo/repo#python:function:pkg.mod.func', ?, ?, 'pkg/mod.py', 'python', 'function', 'pkg.mod.func', 1, 2, ?),
            (99, 'goid:demo/repo#python:function:pkg.mod.test_func', ?, ?, 'pkg/mod.py', 'python', 'test', 'pkg.mod.test_func', 1, 2, ?)
        """,
        [cfg.repo, cfg.commit, now, cfg.repo, cfg.commit, now],
    )
    con.execute(
        """
        INSERT INTO analytics.test_catalog (test_id, rel_path, qualname, repo, commit, status, created_at)
        VALUES ('pkg/mod.py::test_func', 'pkg/mod.py', 'pkg.mod.test_func', ?, ?, 'passed', ?)
        """,
        [cfg.repo, cfg.commit, now],
    )

    artifact = generate_coverage_for_function(
        repo_root=repo_root,
        module_import="pkg.mod",
        function_name="func",
        test_id="pkg/mod.py::test_func",
    )

    def _coverage_loader(_cfg: TestCoverageStepConfig) -> Coverage:
        cov = Coverage(data_file=str(artifact.coverage_file))
        cov.load()
        return cov

    compute_test_coverage_edges(gateway, cfg, coverage_loader=_coverage_loader)
    _assert_single_edge(con)
