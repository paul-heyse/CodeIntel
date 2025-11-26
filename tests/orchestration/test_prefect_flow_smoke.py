"""Smoke tests for Prefect orchestration using public entry points."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pytest

from codeintel.cli.main import main as cli_main
from codeintel.pipeline.orchestration.prefect_flow import ExportArgs, export_docs_flow
from tests._helpers.fixtures import seed_docs_export_minimal
from tests._helpers.gateway import open_fresh_duckdb


def test_prefect_flow_imports() -> None:
    """Prefect flow can be imported without side effects."""
    if not callable(export_docs_flow):
        pytest.fail("export_docs_flow is not callable")


def test_prefect_flow_preflight_only(tmp_path: Path, prefect_quiet_env: None) -> None:
    """
    Run the Prefect flow with no targets to ensure preflight completes via the public entry.

    This exercises the flow wiring without invoking internal helpers directly.
    """
    _ = prefect_quiet_env
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir(parents=True, exist_ok=True)
    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    db_path = build_dir / "db" / "codeintel.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    previous_env = {key: value for key, value in os.environ.items() if key.startswith("CODEINTEL_")}
    os.environ["CODEINTEL_SKIP_SCIP"] = "true"
    try:
        export_docs_flow(
            args=ExportArgs(
                repo_root=repo_root,
                repo="demo/repo",
                commit="deadbeef",
                db_path=db_path,
                build_dir=build_dir,
            ),
            targets=[],
        )
    finally:
        for key in list(os.environ.keys()):
            if key.startswith("CODEINTEL_") and key not in previous_env:
                os.environ.pop(key, None)
        for key, value in previous_env.items():
            os.environ[key] = value


def _run_cli(argv: Iterable[str]) -> int:
    """
    Run the real CLI entry point in tests.

    Returns
    -------
    int
        CLI exit code.
    """
    return cli_main(list(argv))


def test_cli_docs_export_with_validation(tmp_path: Path, prefect_quiet_env: None) -> None:
    """
    Export via the real CLI with validation enabled against a minimal seeded DB.

    This mirrors the production entry point instead of calling internal helpers.
    """
    _ = prefect_quiet_env
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir(parents=True, exist_ok=True)
    build_dir = repo_root / "build"
    db_path = build_dir / "db" / "codeintel.duckdb"
    gateway = open_fresh_duckdb(db_path)
    seed_docs_export_minimal(gateway, repo="demo/repo", commit="deadbeef")
    gateway.close()

    document_output_dir = repo_root / "Document Output"
    argv = [
        "docs",
        "export",
        "--repo-root",
        str(repo_root),
        "--repo",
        "demo/repo",
        "--commit",
        "deadbeef",
        "--db-path",
        str(db_path),
        "--build-dir",
        str(build_dir),
        "--document-output-dir",
        str(document_output_dir),
        "--validate",
        "--schema",
        "function_profile",
    ]

    exit_code = _run_cli(argv)
    if exit_code != 0:
        pytest.fail(f"CLI docs export failed with exit code {exit_code}")

    manifest = document_output_dir / "index.json"
    if not manifest.exists():
        pytest.fail("Expected Document Output manifest from CLI export")
