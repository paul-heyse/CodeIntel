"""Unit tests for SCIP incremental ingest operations."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from codeintel.config.models import ScipIngestConfig
from codeintel.ingestion.common import ModuleRecord
from codeintel.ingestion.scip_ingest import ScipIngestOps, ScipRuntime
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService


def test_scip_module_filter_limits_to_src_python_files(tmp_path: Path) -> None:
    """Ensure SCIP module filter only targets source Python files."""
    cfg = ScipIngestConfig(
        repo_root=tmp_path,
        repo="repo",
        commit="deadbeef",
        build_dir=tmp_path,
        document_output_dir=tmp_path,
    )
    con = duckdb.connect(database=":memory:")
    try:
        runtime = ScipRuntime(
            repo_root=tmp_path,
            scip_dir=tmp_path / "scip",
            doc_dir=tmp_path / "docs",
            con=con,
        )
        ops = ScipIngestOps(
            cfg=cfg,
            runtime=runtime,
            service=ToolService(
                runner=ToolRunner(cache_dir=tmp_path / ".tool_cache"),
            ),
        )

        if not ops.module_filter(
            ModuleRecord(
                rel_path="src/pkg/a.py",
                module_name="pkg.a",
                file_path=tmp_path / "src/pkg/a.py",
                index=0,
                total=1,
            )
        ):
            pytest.fail("Expected src Python file to be processed")
        if ops.module_filter(
            ModuleRecord(
                rel_path="tests/a.py",
                module_name="tests.a",
                file_path=tmp_path / "tests/a.py",
                index=0,
                total=1,
            )
        ):
            pytest.fail("Expected test path to be excluded")
        if ops.module_filter(
            ModuleRecord(
                rel_path="src/docs/readme.md",
                module_name="docs.readme",
                file_path=tmp_path / "src/docs/readme.md",
                index=0,
                total=1,
            )
        ):
            pytest.fail("Expected non-Python file to be excluded")
        if ops.module_filter(
            ModuleRecord(
                rel_path="a.py",
                module_name="a",
                file_path=tmp_path / "a.py",
                index=0,
                total=1,
            )
        ):
            pytest.fail("Expected non-src Python file to be excluded")
    finally:
        con.close()
