"""Ensure pipeline steps reuse a single FunctionCatalog instance."""

from __future__ import annotations

from pathlib import Path

from codeintel.config import (
    BuildPaths,
    ExecutionConfig,
    ScanProfiles,
    SnapshotRef,
)
from codeintel.config.models import GraphBackendConfig, ToolsConfig
from codeintel.ingestion.source_scanner import default_code_profile, default_config_profile
from codeintel.orchestration.steps import (
    AstStep,
    CallGraphStep,
    CFGStep,
    GoidsStep,
    PipelineContext,
    RepoScanStep,
    SymbolUsesStep,
)
from codeintel.storage.gateway import StorageConfig, open_gateway


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    raise AssertionError(detail)


def _write_repo(repo_root: Path) -> None:
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf8")
    (pkg_dir / "a.py").write_text("def callee():\n    return 1\n", encoding="utf8")
    (pkg_dir / "b.py").write_text(
        "from pkg.a import callee\n\ndef caller():\n    return callee()\n",
        encoding="utf8",
    )


def test_steps_share_function_catalog(tmp_path: Path) -> None:
    """Pipeline steps should reuse a single cached function catalog."""
    repo_root = tmp_path / "repo"
    _write_repo(repo_root)

    db_path = tmp_path / "db.duckdb"
    gateway = open_gateway(
        StorageConfig(db_path=db_path, apply_schema=True, ensure_views=True, validate_schema=True)
    )
    snapshot = SnapshotRef(repo_root=repo_root, repo="r", commit="c")
    profiles = ScanProfiles(
        code=default_code_profile(repo_root),
        config=default_config_profile(repo_root),
    )
    execution = ExecutionConfig.for_default_pipeline(
        build_dir=tmp_path / "build",
        tools=ToolsConfig.model_validate({}),
        profiles=profiles,
        graph_backend=GraphBackendConfig(),
    )
    paths = BuildPaths.from_layout(
        repo_root=repo_root,
        build_dir=execution.build_dir,
        db_path=db_path,
    )
    ctx = PipelineContext(
        snapshot=snapshot,
        execution=execution,
        paths=paths,
        gateway=gateway,
    )

    RepoScanStep().run(ctx)
    AstStep().run(ctx)
    GoidsStep().run(ctx)

    CallGraphStep().run(ctx)
    first_catalog = ctx.function_catalog
    _expect(condition=first_catalog is not None, detail="function_catalog was not set")

    CFGStep().run(ctx)
    _expect(condition=ctx.function_catalog is first_catalog, detail="catalog was not reused")

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
    SymbolUsesStep().run(ctx)
    _expect(condition=ctx.function_catalog is first_catalog, detail="catalog was not reused")
