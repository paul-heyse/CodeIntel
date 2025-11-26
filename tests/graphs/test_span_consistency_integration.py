"""Integration test ensuring span alignment across graph builders."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import duckdb

from codeintel.analytics.tests import compute_test_coverage_edges
from codeintel.config import ConfigBuilder
from codeintel.graphs.callgraph_builder import build_call_graph
from codeintel.graphs.cfg_builder import build_cfg_and_dfg
from codeintel.graphs.symbol_uses import build_symbol_use_edges
from codeintel.storage.gateway import StorageGateway
from tests._helpers.tooling import generate_coverage_for_function

REPO: Final = "demo/repo"
COMMIT: Final = "deadbeef"


@dataclass(frozen=True)
class SpanSnapshot:
    """Collected GOID/symbol-use state for alignment assertions."""

    cfg_goids: set[int]
    callgraph_goids: set[int]
    coverage_goids: set[int]
    symbol_use_paths: set[str]


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

    gateway = fresh_gateway
    con = gateway.con

    expected_goid = _seed_modules_and_goids(con, caller_start, caller_end)
    _seed_test_catalog(con)

    builder = ConfigBuilder.from_snapshot(repo=REPO, commit=COMMIT, repo_root=repo_root)

    # Build call graph and CFG using shared spans.
    build_call_graph(gateway, builder.call_graph())
    build_cfg_and_dfg(gateway, builder.cfg_builder())
    # Symbol uses from SCIP JSON.
    scip_json = builder.paths.scip_dir / "index.scip.json"
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
        builder.symbol_uses(scip_json_path=scip_json),
    )

    _load_pkg_for_coverage(repo_root)

    coverage_artifact = generate_coverage_for_function(
        repo_root=repo_root,
        module_import="pkg.b",
        function_name="caller",
        test_id="tests/test_sample.py::test_caller",
    )
    compute_test_coverage_edges(
        gateway,
        builder.test_coverage(coverage_file=coverage_artifact.coverage_file),
    )

    snapshot = _collect_span_snapshot(con)

    expected = {expected_goid}
    if snapshot.cfg_goids != expected:
        message = f"CFG goids mismatch: expected {expected}, got {snapshot.cfg_goids}"
        raise AssertionError(message)
    if snapshot.callgraph_goids != expected:
        message = f"Call graph goids mismatch: expected {expected}, got {snapshot.callgraph_goids}"
        raise AssertionError(message)
    if snapshot.coverage_goids != expected:
        message = f"Coverage goids mismatch: expected {expected}, got {snapshot.coverage_goids}"
        raise AssertionError(message)
    if snapshot.symbol_use_paths != {"pkg/b.py"}:
        message = f"Symbol use mapping mismatch: expected pkg/b.py, got {snapshot.symbol_use_paths}"
        raise AssertionError(message)


def _seed_modules_and_goids(con: object, caller_start: int, caller_end: int) -> int:
    con = _as_duckdb(con)
    now = datetime.now(UTC)
    con.executemany(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, 'python', '[]', '[]')
        """,
        [
            ("pkg.a", "pkg/a.py", REPO, COMMIT),
            ("pkg.b", "pkg/b.py", REPO, COMMIT),
        ],
    )
    expected_goid = 200
    con.execute(
        """
        INSERT INTO core.goids (
            goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            expected_goid,
            "urn:pkg.b.caller",
            REPO,
            COMMIT,
            "pkg/b.py",
            "python",
            "function",
            "pkg.b.caller",
            caller_start,
            caller_end,
            now,
        ),
    )
    return expected_goid


def _seed_test_catalog(con: object) -> None:
    con = _as_duckdb(con)
    con.execute(
        """
        INSERT INTO analytics.test_catalog (test_id, rel_path, qualname, repo, commit, status)
        VALUES ('tests/test_sample.py::test_caller', 'pkg/b.py', 'pkg.b.caller', ?, ?, 'passed')
        """,
        [REPO, COMMIT],
    )


def _load_pkg_for_coverage(repo_root: Path) -> None:
    pkg_init = repo_root / "pkg" / "__init__.py"
    pkg_spec = importlib.util.spec_from_file_location("pkg", pkg_init)
    if pkg_spec is None or pkg_spec.loader is None:
        message = "Unable to load pkg package for coverage"
        raise RuntimeError(message)
    pkg_module = importlib.util.module_from_spec(pkg_spec)
    sys.modules["pkg"] = pkg_module
    pkg_spec.loader.exec_module(pkg_module)


def _collect_span_snapshot(con: object) -> SpanSnapshot:
    con = _as_duckdb(con)
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
    return SpanSnapshot(
        cfg_goids=cfg_goids,
        callgraph_goids=callgraph_goids,
        coverage_goids=coverage_goids,
        symbol_use_paths=symbol_use_paths,
    )


def _as_duckdb(con: object) -> duckdb.DuckDBPyConnection:
    if isinstance(con, duckdb.DuckDBPyConnection):
        return con
    message = f"Unexpected connection type: {type(con)}"
    raise TypeError(message)
