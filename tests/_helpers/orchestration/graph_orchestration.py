"""Graph test environment orchestration functions."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

import duckdb

from codeintel.build.config import BuildConfig
from codeintel.build.graphs.engine import GraphKind, NxGraphEngine
from codeintel.build.providers import create_default_providers
from codeintel.config import SnapshotInit
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.storage.query_results import coerce_int
from tests._helpers.assertions import ModulesAssertions, assert_target_ok
from tests._helpers.configs.graph_config import SpanSnapshot, SpanTestEnv
from tests._helpers.context import TestContext
from tests._helpers.fixtures.rows import (
    ModuleRow,
    RepoMapRow,
    TestCatalogRow,
    insert_rows,
    insert_symbol_use_edges,
)
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness
from tests._helpers.modules_expectations import modules_expected_from_repo_tree

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.configs.graph_config import (
        GraphEngineSeed,
    )


REPO: Final[str] = DEFAULT_VARIANT.repo
COMMIT: Final[str] = DEFAULT_VARIANT.commit


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
    engine = NxGraphEngine(
        dataset_root_dir=gateway.datasets.dataset_root_dir,
        snapshot=snapshot,
    )
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
        Prepared environment with snapshot, gateway, and expected GOID.
    """
    repo_root = tmp_path / "repo"
    _write_repo(repo_root)
    _seed_modules_and_goids(gateway, repo_root)
    snapshot = SnapshotRef(repo=REPO, commit=COMMIT, repo_root=repo_root)
    ModulesAssertions(gateway, snapshot).inventory_consistent()
    _seed_test_catalog(gateway)
    _seed_symbol_use_edges(gateway)
    snapshot = SnapshotInit(repo=REPO, commit=COMMIT, repo_root=repo_root).to_snapshot_ref()
    return SpanTestEnv(
        repo_root=repo_root,
        snapshot=snapshot,
        gateway=gateway,
        expected_goid=None,
    )


def build_span_graph_components(env: SpanTestEnv) -> None:
    """Run call graph and CFG builders for the span test.

    Executes the native Hamilton graph compute nodes to build call graph and CFG data.

    Parameters
    ----------
    env
        Span test environment containing repo_root and gateway.

    Raises
    ------
    RuntimeError
        If graph computation fails execution.
    """
    build_dir = env.repo_root / ".build"
    build_dir.mkdir(parents=True, exist_ok=True)
    paths = BuildPaths.from_explicit(build_dir=build_dir)

    snapshot = SnapshotRef(
        repo=REPO,
        commit=COMMIT,
        repo_root=env.repo_root,
    )
    providers = create_default_providers(ToolsConfig.default())
    ctx = TestContext(
        snapshot=snapshot,
        gateway=env.gateway,
        build_paths=paths,
    )
    harness = HamiltonBuildHarness.wrap(
        ctx,
        providers=providers,
        build_config=BuildConfig.empty(),
    )
    result = harness.run_targets(["call_graph", "cfg"])
    call_graph_record = harness.record("call_graph", result=result)
    try:
        assert_target_ok(call_graph_record)
    except AssertionError as exc:
        message = f"call_graph extraction failed: {call_graph_record.error}"
        raise RuntimeError(message) from exc
    cfg_record = harness.record("cfg", result=result)
    try:
        assert_target_ok(cfg_record)
    except AssertionError as exc:
        message = f"cfg extraction failed: {cfg_record.error}"
        raise RuntimeError(message) from exc


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
        _coerce_goid(row[0])
        for row in con.execute(
            "SELECT function_goid_h128 FROM graph.cfg_blocks WHERE file_path = 'pkg/b.py'"
        ).fetchall()
        if row[0] is not None
    }
    callgraph_goids = {
        _coerce_goid(row[0])
        for row in con.execute(
            "SELECT goid_h128 FROM graph.call_graph_nodes WHERE rel_path = 'pkg/b.py'"
        ).fetchall()
        if row[0] is not None
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
        symbol_use_paths=symbol_use_paths,
    )


def _coerce_goid(value: object) -> int:
    return coerce_int(value, ctx="span snapshot goid")


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


def _seed_modules_and_goids(
    gateway: StorageGateway,
    repo_root: Path,
) -> None:
    path_map = modules_expected_from_repo_tree(repo_root)
    module_map = {module: path for path, module in path_map.items()}
    if not module_map:
        module_map = {
            "pkg.a": "pkg/a.py",
            "pkg.b": "pkg/b.py",
        }
    gateway.con.execute(
        "DELETE FROM core.modules WHERE repo = ? AND commit = ?",
        [REPO, COMMIT],
    )
    gateway.con.execute(
        "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?",
        [REPO, COMMIT],
    )
    insert_rows(
        gateway,
        [
            ModuleRow(module=module, path=path, repo=REPO, commit=COMMIT)
            for module, path in sorted(module_map.items())
        ],
    )
    insert_rows(
        gateway,
        [
            RepoMapRow(
                repo=REPO,
                commit=COMMIT,
                modules=module_map,
            )
        ],
    )


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


def _seed_symbol_use_edges(gateway: StorageGateway) -> None:
    """Seed symbol use edges for span alignment test.

    Creates an edge showing pkg/b.py uses a symbol from pkg/a.py,
    which is what the test expects to find.

    The tuple format is: (symbol, def_path, use_path, same_file, same_module).
    """
    insert_symbol_use_edges(
        gateway,
        [
            ("pkg.a.callee", "pkg/a.py", "pkg/b.py", False, False),
        ],
        repo=REPO,
        commit=COMMIT,
    )


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
]
