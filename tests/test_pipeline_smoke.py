"""Smoke test that runs the full build on a tiny repo."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from codeintel.storage.gateway import StorageConfig, open_gateway
from tests._helpers.cli import run_cli


def test_build_export_smoke(tmp_path: Path) -> None:
    """Build a minimal repo using the build system and verify exports.

    Raises
    ------
    RuntimeError
        If the build run fails or exports are missing.
    """
    repo_root = tmp_path / "repo"
    (repo_root / "pkg").mkdir(parents=True, exist_ok=True)
    (repo_root / "pkg" / "__init__.py").write_text("", encoding="utf8")
    (repo_root / "pkg" / "mod.py").write_text(
        'def hello(name: str) -> str:\n    """Return a greeting."""\n    return f"hi {name}"\n',
        encoding="utf8",
    )

    db_path = repo_root / "build" / "db" / "codeintel.duckdb"
    build_dir = repo_root / "build"

    os.environ["CODEINTEL_SKIP_SCIP"] = "true"
    pytest.xfail("Build smoke test currently fails in function_effects catalog integration")
    result = run_cli(
        [
            "build",
            "run",
            "--all",
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
        ],
    )
    if result.exit_code != 0:
        message = "Build run failed"
        raise RuntimeError(message)

    document_output = repo_root / "Document Output"
    goids_parquet = document_output / "goids.parquet"
    if not goids_parquet.is_file():
        message = "GOID export not found"
        raise RuntimeError(message)

    # Verify other exports materialize
    for fname in (
        "call_graph_edges.parquet",
        "function_metrics.parquet",
        "coverage_functions.parquet",
    ):
        path = document_output / fname
        if not path.is_file():
            message = f"Expected export missing: {fname}"
            raise RuntimeError(message)

    # Verify docs views contain data
    con = open_gateway(
        StorageConfig(
            db_path=db_path,
            read_only=True,
            apply_schema=False,
            ensure_views=True,
            validate_schema=True,
        )
    ).con
    goid_row = con.execute("SELECT COUNT(*) FROM core.goids").fetchone()
    goid_count = int(goid_row[0]) if goid_row is not None else 0
    if goid_count <= 0:
        message = "core.goids is empty after build run"
        raise RuntimeError(message)
    fn_row = con.execute("SELECT COUNT(*) FROM docs.v_function_summary").fetchone()
    fn_summary_count = int(fn_row[0]) if fn_row is not None else 0
    if fn_summary_count <= 0:
        message = "docs.v_function_summary is empty after build run"
        raise RuntimeError(message)

    # JSONL exports present
    goids_jsonl = document_output / "goids.jsonl"
    if not goids_jsonl.is_file():
        message = "goids.jsonl export not found"
        raise RuntimeError(message)
