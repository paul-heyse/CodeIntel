"""Integration test ensuring span alignment across graph builders."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast
from coverage import Coverage

from codeintel.analytics.tests import compute_test_coverage_edges
from codeintel.config import (
    ConfigBuilder,
    CallGraphStepConfig,
    CFGBuilderStepConfig,
    SymbolUsesStepConfig,
    TestCoverageStepConfig,
)
from codeintel.graphs.callgraph_builder import build_call_graph
from codeintel.graphs.cfg_builder import build_cfg_and_dfg
from codeintel.graphs.symbol_uses import build_symbol_use_edges
from codeintel.storage.gateway import StorageGateway
from tests._helpers.fakes import FakeCoverage


def _write_repo(repo_root: Path) -> tuple[int, int]:
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf8")
    (pkg_dir / "a.py").write_text("def callee():\n    return 1\n", encoding="utf8")
    (pkg_dir / "b.py").write_text(
        "from pkg.a import callee\n\ndef caller():\n    return callee()\n",
        encoding="utf8",
    )
    # Line numbers for caller function span (3-4).
    return 3, 4


def test_span_alignment_across_components(tmp_path: Path, fresh_gateway: StorageGateway) -> None:
    """
    Ensure call graph, CFG/DFG, and test coverage edges agree on function GOIDs.

    Raises
    ------
    AssertionError
        If any component produces mismatched GOIDs for the same function spans.
    """
    repo_root = tmp_path / "repo"
    caller_start, caller_end = _write_repo(repo_root)

    repo: Final = "demo/repo"
    commit: Final = "deadbeef"

    gateway = fresh_gateway
    con = gateway.con

    # Seed modules for module mapping.
    con.executemany(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, 'python', '[]', '[]')
        """,
        [
            ("pkg.a", "pkg/a.py", repo, commit),
            ("pkg.b", "pkg/b.py", repo, commit),
        ],
    )

    # Seed GOIDs for caller and callee.
    now = datetime.now(UTC)
    con.executemany(
        """
        INSERT INTO core.goids (
            goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                100,
                "urn:pkg.a.callee",
                repo,
                commit,
                "pkg/a.py",
                "python",
                "function",
                "pkg.a.callee",
                1,
                2,
                now,
            ),
            (
                200,
                "urn:pkg.b.caller",
                repo,
                commit,
                "pkg/b.py",
                "python",
                "function",
                "pkg.b.caller",
                caller_start,
                caller_end,
                now,
            ),
        ],
    )

    # Minimal test_catalog row to supply status.
    con.execute(
        """
        INSERT INTO analytics.test_catalog (test_id, rel_path, qualname, repo, commit, status)
        VALUES ('tests/test_sample.py::test_caller', 'pkg/b.py', 'pkg.b.caller', ?, ?, 'passed')
        """,
        [repo, commit],
    )

    # Build call graph and CFG using shared spans.
    build_call_graph(
        gateway, CallGraphStepConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    )
    build_cfg_and_dfg(
        gateway,
        CFGBuilderStepConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root),
    )
    # Symbol uses from SCIP JSON.
    scip_json = repo_root / "build" / "scip" / "index.scip.json"
    scip_json.parent.mkdir(parents=True, exist_ok=True)
    scip_json.write_text(
        (
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
            """
        ).strip(),
        encoding="utf8",
    )
    build_symbol_use_edges(
        gateway,
        SymbolUsesStepConfig.from_paths(
            repo_root=repo_root,
            repo=repo,
            commit=commit,
            scip_json_path=scip_json,
        ),
    )

    def _load_fake(_cfg: TestCoverageStepConfig) -> Coverage:
        abs_b = str((repo_root / "pkg" / "b.py").resolve())
        statements = {abs_b: [caller_start, caller_end]}
        contexts = {abs_b: {caller_start: {"tests/test_sample.py::test_caller"}}}
        return cast(Coverage, FakeCoverage(statements, contexts))

    compute_test_coverage_edges(
        gateway,
        TestCoverageStepConfig.from_paths(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
            coverage_file=None,
        ),
        coverage_loader=_load_fake,
    )

    cfg_goids = {
        row[0]
        for row in con.execute(
            "SELECT function_goid_h128 FROM graph.cfg_blocks WHERE file_path = 'pkg/b.py'"
        ).fetchall()
    }
    callgraph_goids = {
        row[0]
        for row in con.execute(
            "SELECT goid_h128 FROM graph.call_graph_nodes WHERE rel_path = 'pkg/b.py'"
        ).fetchall()
    }
    coverage_goids = {
        row[0]
        for row in con.execute(
            "SELECT function_goid_h128 FROM analytics.test_coverage_edges"
        ).fetchall()
    }
    symbol_use_paths = {
        row[0]
        for row in con.execute(
            "SELECT use_path FROM graph.symbol_use_edges WHERE def_path = 'pkg/a.py'"
        ).fetchall()
    }

    expected = {200}
    if cfg_goids != expected:
        message = f"CFG goids mismatch: expected {expected}, got {cfg_goids}"
        raise AssertionError(message)
    if callgraph_goids != expected:
        message = f"Call graph goids mismatch: expected {expected}, got {callgraph_goids}"
        raise AssertionError(message)
    if coverage_goids != expected:
        message = f"Coverage goids mismatch: expected {expected}, got {coverage_goids}"
        raise AssertionError(message)
    if symbol_use_paths != {"pkg/b.py"}:
        message = f"Symbol use mapping mismatch: expected pkg/b.py, got {symbol_use_paths}"
        raise AssertionError(message)
