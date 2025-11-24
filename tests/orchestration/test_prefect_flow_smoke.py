"""Smoke test to ensure Prefect flow scaffolding imports and runs no-op path."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from codeintel.orchestration.prefect_flow import ExportArgs, export_docs_flow


def test_prefect_flow_imports() -> None:
    """Prefect flow can be imported without side effects."""
    if not callable(export_docs_flow):
        pytest.fail("export_docs_flow is not callable")


def test_prefect_flow_minimal(tmp_path: Path, prefect_quiet_env: None) -> None:
    """
    Run the flow with empty inputs to ensure it executes without raising.

    This is a lightweight guardrail; full data population is covered elsewhere.
    """
    _ = prefect_quiet_env  # ensure harness/quiet logging fixtures are applied
    prev_skip = os.environ.get("CODEINTEL_SKIP_SCIP")
    os.environ["CODEINTEL_SKIP_SCIP"] = "true"
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir(parents=True, exist_ok=True)
    db_path = repo_root / "build" / "db" / "codeintel.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    scip_dir = build_dir / "scip"
    scip_dir.mkdir(parents=True, exist_ok=True)
    (scip_dir / "index.scip").write_text("dummy", encoding="utf8")
    (scip_dir / "index.scip.json").write_text("[]", encoding="utf8")

    export_docs_flow(
        args=ExportArgs(
            repo_root=repo_root,
            repo="demo/repo",
            commit="deadbeef",
            db_path=db_path,
            build_dir=build_dir,
        )
    )
    if prev_skip is None:
        os.environ.pop("CODEINTEL_SKIP_SCIP", None)
    else:
        os.environ["CODEINTEL_SKIP_SCIP"] = prev_skip
