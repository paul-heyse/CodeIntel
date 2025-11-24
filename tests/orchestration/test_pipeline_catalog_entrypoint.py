"""Integration test ensuring pipeline steps use the shared function catalog."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from coverage import Coverage

from codeintel.analytics.tests_analytics import compute_test_coverage_edges
from codeintel.config.models import (
    CallGraphConfig,
    CFGBuilderConfig,
    SymbolUsesConfig,
    TestCoverageConfig,
)
from codeintel.graphs.callgraph_builder import build_call_graph
from codeintel.graphs.cfg_builder import build_cfg_and_dfg
from codeintel.graphs.function_catalog import load_function_catalog
from codeintel.graphs.symbol_uses import build_symbol_use_edges
from codeintel.ingestion.common import run_batch
from codeintel.orchestration.steps import AstStep, GoidsStep, PipelineContext, RepoScanStep
from codeintel.storage.gateway import StorageConfig, open_gateway
from tests._helpers.fakes import FakeCoverage

REPO: Final = "demo/repo"
COMMIT: Final = "deadbeef"


def _write_repo(repo_root: Path) -> tuple[int, int]:
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf8")
    (pkg_dir / "a.py").write_text("def callee():\n    return 1\n", encoding="utf8")
    (pkg_dir / "b.py").write_text(
        "from pkg.a import callee\n\ndef caller():\n    return callee()\n",
        encoding="utf8",
    )
    return 3, 4


def test_pipeline_steps_use_function_catalog(tmp_path: Path) -> None:
    """Ensure pipeline steps and builders all consume the shared function catalog."""
    repo_root = tmp_path / "repo"
    caller_start, caller_end = _write_repo(repo_root)

    db_path = tmp_path / "db.duckdb"
    con = open_gateway(
        StorageConfig(db_path=db_path, apply_schema=True, ensure_views=False, validate_schema=True)
    ).con

    run_batch(
        con,
        "core.modules",
        [
            ("pkg.a", "pkg/a.py", REPO, COMMIT, "python", "[]", "[]"),
            ("pkg.b", "pkg/b.py", REPO, COMMIT, "python", "[]", "[]"),
        ],
        delete_params=None,
    )

    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO analytics.test_catalog (test_id, rel_path, qualname, repo, commit, status, created_at)
        VALUES ('tests/test_sample.py::test_caller', 'pkg/b.py', 'pkg.b.caller', ?, ?, 'passed', ?)
        """,
        [REPO, COMMIT, now],
    )

    ctx = PipelineContext(
        repo_root=repo_root,
        db_path=db_path,
        build_dir=tmp_path / "build",
        repo=REPO,
        commit=COMMIT,
    )

    RepoScanStep().run(ctx, con)
    AstStep().run(ctx, con)
    GoidsStep().run(ctx, con)

    build_call_graph(con, CallGraphConfig.from_paths(repo=REPO, commit=COMMIT, repo_root=repo_root))
    build_cfg_and_dfg(
        con, CFGBuilderConfig.from_paths(repo=REPO, commit=COMMIT, repo_root=repo_root)
    )

    scip_json = ctx.build_dir / "scip" / "index.scip.json"
    scip_json.parent.mkdir(parents=True, exist_ok=True)
    scip_json.write_text(
        """
        [
          {
            "relative_path": "pkg/a.py",
            "occurrences": [
              { "symbol": "sym#def", "symbol_roles": 1 }
            ]
          },
          {
            "relative_path": "pkg/b.py",
            "occurrences": [
              { "symbol": "sym#def", "symbol_roles": 2 }
            ]
          }
        ]
        """.strip(),
        encoding="utf8",
    )
    build_symbol_use_edges(
        con,
        SymbolUsesConfig.from_paths(
            repo_root=repo_root,
            repo=REPO,
            commit=COMMIT,
            scip_json_path=scip_json,
        ),
    )

    def _load_fake(_cfg: TestCoverageConfig) -> Coverage:
        abs_b = str((repo_root / "pkg" / "b.py").resolve())
        statements = {abs_b: [caller_start, caller_end]}
        contexts = {abs_b: {caller_start: {"tests/test_sample.py::test_caller"}}}
        return FakeCoverage(statements, contexts)

    compute_test_coverage_edges(
        con,
        TestCoverageConfig.from_paths(
            repo=REPO,
            commit=COMMIT,
            repo_root=repo_root,
            coverage_file=None,
        ),
        coverage_loader=_load_fake,
    )

    def _assert(condition: object, *, detail: str) -> None:
        if condition:
            return
        raise AssertionError(detail)

    catalog = load_function_catalog(con, repo=REPO, commit=COMMIT)
    callee_goid = catalog.lookup_goid("pkg/a.py", 1, 2, "pkg.a.callee")
    caller_goid = catalog.lookup_goid("pkg/b.py", caller_start, caller_end, "pkg.b.caller")
    _assert(callee_goid is not None, detail="Catalog missing callee GOID")
    _assert(caller_goid is not None, detail="Catalog missing caller GOID")
    _assert(
        catalog.module_by_path.get("pkg/a.py") == "pkg.a",
        detail="Catalog module mapping missing pkg.a",
    )

    edge_goids = {
        row[0]
        for row in con.execute("SELECT caller_goid_h128 FROM graph.call_graph_edges").fetchall()
    }
    _assert(
        caller_goid in edge_goids,
        detail=f"Call graph edges missing caller GOID {caller_goid}",
    )

    cfg_goids = {
        row[0]
        for row in con.execute(
            "SELECT function_goid_h128 FROM graph.cfg_blocks WHERE file_path = 'pkg/b.py'"
        ).fetchall()
    }
    _assert(caller_goid in cfg_goids, detail=f"CFG blocks missing GOID {caller_goid}")

    sym_use_paths = {
        row[0]
        for row in con.execute(
            "SELECT use_path FROM graph.symbol_use_edges WHERE def_path = 'pkg/a.py'"
        ).fetchall()
    }
    _assert(
        sym_use_paths == {"pkg/b.py"},
        detail=f"Symbol uses not populated as expected: {sym_use_paths}",
    )

    coverage_goids = {
        row[0]
        for row in con.execute(
            "SELECT function_goid_h128 FROM analytics.test_coverage_edges"
        ).fetchall()
    }
    _assert(coverage_goids == {caller_goid}, detail=f"Coverage GOIDs mismatch: {coverage_goids}")
