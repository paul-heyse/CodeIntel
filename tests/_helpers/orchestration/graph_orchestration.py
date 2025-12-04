"""Graph test environment orchestration functions."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from codeintel.config import ConfigBuilder
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.engine import GraphKind, NxGraphEngine
from codeintel.graphs.plugins.builders.callgraph import get_callgraph_builder_plugin
from codeintel.graphs.plugins.builders.cfg_dfg import build_cfg_and_dfg
from codeintel.graphs.plugins.builders.symbol_uses import build_symbol_use_edges
from codeintel.graphs.plugins.runner import GraphPluginRunner
from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders import GoidRow, ModuleRow, TestCatalogRow
from tests._helpers.configs.graph_config import (
    COMMIT,
    REPO,
    GraphEngineSeed,
    SpanSnapshot,
    SpanTestEnv,
)
from tests._helpers.row_protocol import insert_rows
from tests._helpers.tooling import CoverageArtifact, generate_coverage_for_function


def build_seeded_graph_engine(gateway: StorageGateway, seed: GraphEngineSeed) -> NxGraphEngine:
    """Construct an NxGraphEngine seeded with provided graphs.

    Parameters
    ----------
    gateway
        Storage gateway backing the engine.
    seed
        Seed configuration containing graphs and snapshot metadata.

    Returns
    -------
    NxGraphEngine
        Engine seeded with the provided graphs and ready for use.
    """
    snapshot = SnapshotRef(
        repo=seed.repo,
        commit=seed.commit,
        repo_root=seed.repo_root or Path.cwd(),
    )
    engine = NxGraphEngine(gateway=gateway, snapshot=snapshot)
    if seed.call_graph is not None:
        engine.seed(GraphKind.CALL_GRAPH, seed.call_graph)
    if seed.import_graph is not None:
        engine.seed(GraphKind.IMPORT_GRAPH, seed.import_graph)
    return engine


def create_span_test_env(tmp_path: Path, gateway: StorageGateway) -> SpanTestEnv:
    """Create a reusable environment for span alignment checks.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.
    gateway
        Storage gateway for database operations.

    Returns
    -------
    SpanTestEnv
        Prepared environment with builder, gateway, and expected GOID.
    """
    repo_root = tmp_path / "repo"
    caller_start, caller_end = _write_repo(repo_root)
    expected_goid = _seed_modules_and_goids(gateway, caller_start, caller_end)
    _seed_test_catalog(gateway)
    builder = ConfigBuilder.from_snapshot(repo=REPO, commit=COMMIT, repo_root=repo_root)
    return SpanTestEnv(
        repo_root=repo_root,
        builder=builder,
        gateway=gateway,
        expected_goid=expected_goid,
    )


def build_span_graph_components(env: SpanTestEnv) -> None:
    """Run call graph, CFG/DFG, and symbol-use builders for the span test.

    Parameters
    ----------
    env
        Span test environment.
    """
    call_graph_cfg = env.builder.call_graph()
    runner = GraphPluginRunner(gateway=env.gateway)
    plugin = get_callgraph_builder_plugin()
    exec_ctx = runner.build_context(call_graph_cfg.snapshot)
    runner.run_plugin(plugin, exec_ctx)
    build_cfg_and_dfg(env.gateway, env.builder.cfg_builder())
    scip_json = env.builder.paths.scip_dir / "index.scip.json"
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
        env.gateway,
        env.builder.symbol_uses(scip_json_path=scip_json),
    )


def generate_span_coverage(repo_root: Path) -> CoverageArtifact:
    """Generate coverage artifact for the test caller function.

    Parameters
    ----------
    repo_root
        Path to the repository root.

    Returns
    -------
    CoverageArtifact
        Coverage artifact containing the created coverage file.
    """
    _load_pkg_for_coverage(repo_root)
    return generate_coverage_for_function(
        repo_root=repo_root,
        module_import="pkg.b",
        function_name="caller",
        test_id="tests/test_sample.py::test_caller",
    )


def collect_span_snapshot(con: object) -> SpanSnapshot:
    """Collect span-related GOIDs and symbol uses from the gateway.

    Parameters
    ----------
    con
        DuckDB connection object.

    Returns
    -------
    SpanSnapshot
        Snapshot of GOIDs and symbol-use paths for the test repo.
    """
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


def _seed_modules_and_goids(gateway: StorageGateway, caller_start: int, caller_end: int) -> int:
    now = datetime.now(UTC)
    insert_rows(
        gateway,
        [
            ModuleRow(module="pkg.a", path="pkg/a.py", repo=REPO, commit=COMMIT),
            ModuleRow(module="pkg.b", path="pkg/b.py", repo=REPO, commit=COMMIT),
        ],
    )
    expected_goid = 200
    insert_rows(
        gateway,
        [
            GoidRow(
                goid_h128=expected_goid,
                urn="urn:pkg.b.caller",
                repo=REPO,
                commit=COMMIT,
                rel_path="pkg/b.py",
                kind="function",
                qualname="pkg.b.caller",
                start_line=caller_start,
                end_line=caller_end,
                created_at=now,
            )
        ],
    )
    return expected_goid


def _seed_test_catalog(gateway: StorageGateway) -> None:
    now = datetime.now(UTC)
    insert_rows(
        gateway,
        [
            TestCatalogRow(
                test_id="tests/test_sample.py::test_caller",
                repo=REPO,
                commit=COMMIT,
                rel_path="pkg/b.py",
                qualname="pkg.b.caller",
                status="passed",
                created_at=now,
            )
        ],
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


def _as_duckdb(con: object) -> duckdb.DuckDBPyConnection:
    if isinstance(con, duckdb.DuckDBPyConnection):
        return con
    message = f"Unexpected connection type: {type(con)}"
    raise TypeError(message)


__all__ = [
    "build_seeded_graph_engine",
    "build_span_graph_components",
    "collect_span_snapshot",
    "create_span_test_env",
    "generate_span_coverage",
]
