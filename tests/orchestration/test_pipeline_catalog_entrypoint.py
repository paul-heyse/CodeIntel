"""Integration test ensuring pipeline steps use the shared function catalog."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pytest
from coverage import Coverage

from codeintel.analytics.tests import compute_test_coverage_edges
from codeintel.config import (
    BuildPaths,
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
from codeintel.pipeline.orchestration.steps import AstStep, GoidsStep, PipelineContext, RepoScanStep
from codeintel.storage.gateway import StorageConfig, open_gateway
from tests._helpers.tooling import generate_coverage_for_function

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

    gateway.con.execute(
        """
        INSERT INTO analytics.test_catalog (test_id, rel_path, qualname, repo, commit, status, created_at)
        VALUES ('tests/test_sample.py::test_caller', 'pkg/b.py', 'pkg.b.caller', ?, ?, 'passed', ?)
        """,
        [REPO, COMMIT, datetime.now(UTC)],
    )

    snapshot = SnapshotRef(repo_root=repo_root, repo=REPO, commit=COMMIT)
    profiles = ScanProfiles(
        code=default_code_profile(repo_root),
        config=default_config_profile(repo_root),
    )
    build_paths = BuildPaths.from_layout(
        repo_root=repo_root,
        build_dir=tmp_path / "build",
        db_path=gateway.config.db_path,
    )
    ctx = PipelineContext(
        snapshot=snapshot,
        paths=build_paths,
        gateway=gateway,
        tools=ToolsConfig.model_validate({}),
        code_profile_cfg=profiles.code,
        config_profile_cfg=profiles.config,
        graph_backend_cfg=GraphBackendConfig(),
    )
    RepoScanStep().run(ctx)
    AstStep().run(ctx)
    GoidsStep().run(ctx)

    build_call_graph(gateway, ctx.config_builder().call_graph())
    build_cfg_and_dfg(gateway, ctx.config_builder().cfg_builder())

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
        ctx.config_builder().symbol_uses(scip_json_path=scip_json),
    )

    pkg_init = repo_root / "pkg" / "__init__.py"
    pkg_spec = importlib.util.spec_from_file_location("pkg", pkg_init)
    if pkg_spec is None or pkg_spec.loader is None:
        pytest.fail("Unable to load pkg package for coverage")
    sys.modules["pkg"] = importlib.util.module_from_spec(pkg_spec)
    pkg_spec.loader.exec_module(sys.modules["pkg"])

    coverage_file = build_paths.build_dir / ".coverage"
    generate_coverage_for_function(
        repo_root=repo_root,
        module_import="pkg.b",
        function_name="caller",
        test_id="tests/test_sample.py::test_caller",
        coverage_file=coverage_file,
    )

    def _load_coverage(_cfg: TestCoverageStepConfig) -> Coverage:
        cov = Coverage(data_file=str(coverage_file), config_file=False)
        cov.load()
        return cov

    compute_test_coverage_edges(
        gateway,
        ctx.config_builder().test_coverage(
            coverage_file=coverage_file,
            coverage_loader=_load_coverage,
        ),
        coverage_loader=_load_coverage,
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

    coverage_goids = {
        row[0]
        for row in gateway.con.execute(
            "SELECT function_goid_h128 FROM analytics.test_coverage_edges"
        ).fetchall()
    }
    _assert(
        coverage_goids and caller_goid in coverage_goids,
        detail=f"Coverage GOIDs missing caller: {coverage_goids}",
    )
