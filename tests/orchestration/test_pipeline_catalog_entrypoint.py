"""Integration test ensuring pipeline steps use the shared function catalog."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

from coverage import Coverage

from codeintel.analytics.tests import compute_test_coverage_edges
from codeintel.config import (
    BuildPaths,
    ConfigBuilder,
    ExecutionConfig,
    ScanProfiles,
    SnapshotRef,
    TestCoverageStepConfig,
)
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import GraphBackendConfig
from codeintel.graphs.callgraph_builder import build_call_graph
from codeintel.graphs.cfg_builder import build_cfg_and_dfg
from codeintel.graphs.function_catalog import load_function_catalog
from codeintel.graphs.symbol_uses import build_symbol_use_edges
from codeintel.ingestion.common import run_batch
from codeintel.ingestion.source_scanner import default_code_profile, default_config_profile
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
    caller_lines = _write_repo(repo_root)

    gateway = open_gateway(
        StorageConfig(
            db_path=tmp_path / "db.duckdb",
            apply_schema=True,
            ensure_views=True,
            validate_schema=True,
        )
    )

    run_batch(
        gateway,
        "core.modules",
        [
            ("pkg.a", "pkg/a.py", REPO, COMMIT, "python", "[]", "[]"),
            ("pkg.b", "pkg/b.py", REPO, COMMIT, "python", "[]", "[]"),
        ],
        delete_params=None,
    )

    now = datetime.now(UTC)
    gateway.con.execute(
        """
        INSERT INTO analytics.test_catalog (test_id, rel_path, qualname, repo, commit, status, created_at)
        VALUES ('tests/test_sample.py::test_caller', 'pkg/b.py', 'pkg.b.caller', ?, ?, 'passed', ?)
        """,
        [REPO, COMMIT, now],
    )

    snapshot = SnapshotRef(repo_root=repo_root, repo=REPO, commit=COMMIT)
    execution = ExecutionConfig.for_default_pipeline(
        build_dir=tmp_path / "build",
        tools=ToolsConfig.model_validate({}),
        profiles=ScanProfiles(
            code=default_code_profile(repo_root),
            config=default_config_profile(repo_root),
        ),
        graph_backend=GraphBackendConfig(),
    )
    build_paths = BuildPaths.from_layout(
        repo_root=repo_root,
        build_dir=execution.build_dir,
        db_path=gateway.config.db_path,
    )
    ctx = PipelineContext(
        snapshot=snapshot,
        execution=execution,
        paths=build_paths,
        gateway=gateway,
    )
    builder = ConfigBuilder.from_snapshot(
        repo=REPO,
        commit=COMMIT,
        repo_root=repo_root,
        build_dir=build_paths.build_dir,
    )

    RepoScanStep().run(ctx)
    AstStep().run(ctx)
    GoidsStep().run(ctx)

    build_call_graph(gateway, builder.call_graph())
    build_cfg_and_dfg(gateway, builder.cfg_builder())

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
        gateway,
        builder.symbol_uses(scip_json_path=scip_json),
    )

    def _load_fake(_cfg: TestCoverageStepConfig) -> Coverage:
        abs_b = str((repo_root / "pkg" / "b.py").resolve())
        statements = {abs_b: [caller_lines[0], caller_lines[1]]}
        contexts = {abs_b: {caller_lines[0]: {"tests/test_sample.py::test_caller"}}}
        return cast("Coverage", FakeCoverage(statements, contexts))

    compute_test_coverage_edges(
        gateway,
        builder.test_coverage(coverage_loader=_load_fake),
        coverage_loader=_load_fake,
    )

    def _assert(condition: object, *, detail: str) -> None:
        if condition:
            return
        raise AssertionError(detail)

    catalog = load_function_catalog(gateway, repo=REPO, commit=COMMIT)
    callee_goid = catalog.lookup_goid("pkg/a.py", 1, 2, "pkg.a.callee")
    caller_goid = catalog.lookup_goid("pkg/b.py", caller_lines[0], caller_lines[1], "pkg.b.caller")
    _assert(callee_goid is not None, detail="Catalog missing callee GOID")
    _assert(caller_goid is not None, detail="Catalog missing caller GOID")
    _assert(
        catalog.module_by_path.get("pkg/a.py") == "pkg.a",
        detail="Catalog module mapping missing pkg.a",
    )

    _assert(
        caller_goid
        in {
            row[0]
            for row in gateway.con.execute(
                "SELECT caller_goid_h128 FROM graph.call_graph_edges"
            ).fetchall()
        },
        detail=f"Call graph edges missing caller GOID {caller_goid}",
    )

    _assert(
        caller_goid
        in {
            row[0]
            for row in gateway.con.execute(
                "SELECT function_goid_h128 FROM graph.cfg_blocks WHERE file_path = 'pkg/b.py'"
            ).fetchall()
        },
        detail=f"CFG blocks missing GOID {caller_goid}",
    )

    _assert(
        {
            row[0]
            for row in gateway.con.execute(
                "SELECT use_path FROM graph.symbol_use_edges WHERE def_path = 'pkg/a.py'"
            ).fetchall()
        }
        == {"pkg/b.py"},
        detail="Symbol uses not populated as expected: %s"
        % {
            row[0]
            for row in gateway.con.execute(
                "SELECT use_path FROM graph.symbol_use_edges WHERE def_path = 'pkg/a.py'"
            ).fetchall()
        },
    )

    _assert(
        {
            row[0]
            for row in gateway.con.execute(
                "SELECT function_goid_h128 FROM analytics.test_coverage_edges"
            ).fetchall()
        }
        == {caller_goid},
        detail="Coverage GOIDs mismatch: %s"
        % {
            row[0]
            for row in gateway.con.execute(
                "SELECT function_goid_h128 FROM analytics.test_coverage_edges"
            ).fetchall()
        },
    )
