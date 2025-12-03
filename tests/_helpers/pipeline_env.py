"""Shared helpers for pipeline graph/coverage integration tests."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from coverage import Coverage

from codeintel.config import BuildPaths, ScanProfiles, SnapshotRef, TestCoverageStepConfig
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import GraphBackendConfig
from codeintel.graphs.plugins.builders.callgraph import get_callgraph_builder_plugin
from codeintel.graphs.plugins.builders.cfg_dfg import build_cfg_and_dfg
from codeintel.graphs.plugins.builders.symbol_uses import build_symbol_use_edges
from codeintel.graphs.plugins.runner import GraphPluginRunner
from codeintel.ingestion.infrastructure_utilities.source_scanner import (
    default_code_profile,
    default_config_profile,
)
from codeintel.pipeline.orchestration.core import PipelineContext
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway
from tests._helpers.tooling import generate_coverage_for_function

REPO = "demo/repo"
COMMIT = "deadbeef"


@dataclass
class PipelineEnv:
    """Reusable environment for pipeline graph/coverage assertions."""

    repo_root: Path
    build_paths: BuildPaths
    gateway: StorageGateway
    ctx: PipelineContext
    caller_lines: tuple[int, int]


def create_pipeline_env(tmp_path: Path) -> PipelineEnv:
    """Construct a pipeline environment with seeded catalog rows.

    Returns
    -------
    PipelineEnv
        Environment containing repo paths, gateway, and pipeline context.
    """
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
    return PipelineEnv(
        repo_root=repo_root,
        build_paths=build_paths,
        gateway=gateway,
        ctx=ctx,
        caller_lines=caller_lines,
    )


def build_graph_and_symbols(env: PipelineEnv) -> None:
    """Run pipeline graph steps and symbol-use generation."""
    _seed_modules_and_goids(env.gateway, env.caller_lines)

    builder = env.ctx.config_builder()
    call_graph_cfg = builder.call_graph()
    runner = GraphPluginRunner(gateway=env.gateway)
    plugin = get_callgraph_builder_plugin()
    exec_ctx = runner.build_context(call_graph_cfg.snapshot)
    runner.run_plugin(plugin, exec_ctx)
    build_cfg_and_dfg(env.gateway, builder.cfg_builder())

    scip_json = builder.paths.scip_dir / "index.scip.json"
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
        env.ctx.config_builder().symbol_uses(scip_json_path=scip_json),
    )


def generate_pipeline_coverage(env: PipelineEnv) -> Path:
    """Generate coverage artifact for the pipeline test.

    Returns
    -------
    Path
        Path to the generated coverage file.
    """
    pkg_init = env.repo_root / "pkg" / "__init__.py"
    pkg_spec = importlib.util.spec_from_file_location("pkg", pkg_init)
    if pkg_spec is None or pkg_spec.loader is None:
        pytest.fail("Unable to load pkg package for coverage")
    sys.modules["pkg"] = importlib.util.module_from_spec(pkg_spec)
    pkg_spec.loader.exec_module(sys.modules["pkg"])

    coverage_file = env.build_paths.build_dir / ".coverage"
    generate_coverage_for_function(
        repo_root=env.repo_root,
        module_import="pkg.b",
        function_name="caller",
        test_id="tests/test_sample.py::test_caller",
        coverage_file=coverage_file,
    )
    return coverage_file


def load_coverage(coverage_file: Path, _cfg: TestCoverageStepConfig | None = None) -> Coverage:
    """Load a Coverage object from disk for test coverage processing.

    Returns
    -------
    Coverage
        Loaded coverage object ready for analysis.
    """
    cov = Coverage(data_file=str(coverage_file), config_file=False)
    cov.load()
    return cov


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


def _seed_modules_and_goids(gateway: StorageGateway, caller_lines: tuple[int, int]) -> None:
    caller_start, caller_end = caller_lines
    now = datetime.now(UTC)
    gateway.con.executemany(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, 'python', '[]', '[]')
        """,
        [
            ("pkg.a", "pkg/a.py", REPO, COMMIT),
            ("pkg.b", "pkg/b.py", REPO, COMMIT),
        ],
    )
    gateway.con.executemany(
        """
        INSERT INTO core.goids (
            goid_h128, urn, repo, commit, rel_path, language, kind, qualname, start_line, end_line, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                100,
                "urn:pkg.a.callee",
                REPO,
                COMMIT,
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
        ],
    )


__all__ = [
    "PipelineEnv",
    "build_graph_and_symbols",
    "create_pipeline_env",
    "generate_pipeline_coverage",
    "load_coverage",
]
