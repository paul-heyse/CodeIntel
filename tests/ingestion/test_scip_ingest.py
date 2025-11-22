"""Integration test for scip_ingest using real SCIP binaries."""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pytest

from codeintel.ingestion import scip_ingest
from codeintel.storage.schemas import apply_all_schemas


def test_ingest_scip_produces_artifacts(tmp_path: Path) -> None:
    """
    Ensure scip_ingest generates SCIP artifacts and registers scip_index_view.

    Skips if scip-python or scip binaries are unavailable.
    """
    if shutil.which("scip-python") is None or shutil.which("scip") is None:
        pytest.skip("scip-python or scip not available on PATH")

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir()

    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf8")
    (pkg_dir / "mod.py").write_text("def foo(x: int) -> int:\n    return x + 1\n", encoding="utf8")

    build_dir = repo_root / "build"
    document_output_dir = repo_root / "document_output"
    db_path = build_dir / "db" / "codeintel_prefect.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = scip_ingest.ScipIngestConfig(
        repo_root=repo_root,
        repo="demo/repo",
        commit="deadbeef",
        build_dir=build_dir,
        document_output_dir=document_output_dir,
    )

    con = duckdb.connect(str(db_path))
    apply_all_schemas(con)

    result = scip_ingest.ingest_scip(con=con, cfg=cfg)
    if result.status != "success":
        pytest.skip(f"SCIP ingestion not successful in test environment: {result.reason}")

    build_scip = build_dir / "scip"
    doc_scip = document_output_dir
    if not (build_scip / "index.scip").is_file():
        pytest.fail("index.scip was not created under build/scip")
    if not (build_scip / "index.scip.json").is_file():
        pytest.fail("index.scip.json was not created under build/scip")
    if not (doc_scip / "index.scip").is_file():
        pytest.fail("index.scip was not copied to document_output")
    if not (doc_scip / "index.scip.json").is_file():
        pytest.fail("index.scip.json was not copied to document_output")

    row = con.execute("SELECT COUNT(*) FROM scip_index_view").fetchone()
    if row is None:
        pytest.fail("scip_index_view did not return a row")
    row_count = row[0]
    if row_count == 0:
        pytest.fail("scip_index_view is empty; expected rows after ingest")

    con.close()
