"""Tests for docs view profiling module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.storage.helpers.profiling import (
    DOCS_VIEWS,
    explain,
    run_profile,
    write_text,
)
from tests._helpers import docs_views_ready_gateway
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_true,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


@pytest.fixture
def docs_profile_db(tmp_path: Path) -> Path:
    """Provision a file-backed docs views database and return its path.

    Returns
    -------
    Path
        Filesystem path to the provisioned docs profiling database.
    """
    db_path = tmp_path / "docs_profile.duckdb"
    ctx = docs_views_ready_gateway(
        tmp_path / "docs_profile_repo",
        file_backed=True,
        db_path=db_path,
    )
    db_file = Path(ctx.gateway.config.db_path)
    ctx.close()
    return db_file


def test_docs_views_is_defined() -> None:
    """Verify DOCS_VIEWS constant is defined and non-empty."""
    expect_is_instance(DOCS_VIEWS, tuple)
    expect_true(len(DOCS_VIEWS) > 0)


def test_docs_views_contains_expected_views() -> None:
    """Verify DOCS_VIEWS contains expected subsystem views."""
    expect_in("docs.v_subsystem_profile", DOCS_VIEWS)
    expect_in("docs.v_subsystem_coverage", DOCS_VIEWS)


def test_write_text_creates_file(tmp_path: Path) -> None:
    """Verify write_text creates file with correct content."""
    file_path = tmp_path / "test.txt"

    write_text(file_path, "Hello, World!")

    expect_true(file_path.exists())
    expect_equal(file_path.read_text(), "Hello, World!")


def test_write_text_creates_nested_directories(tmp_path: Path) -> None:
    """Verify write_text creates parent directories."""
    file_path = tmp_path / "nested" / "dir" / "test.txt"

    write_text(file_path, "Nested content")

    expect_true(file_path.exists())
    expect_equal(file_path.read_text(), "Nested content")


def test_explain_returns_plan_text(docs_views_gateway: StorageGateway) -> None:
    """Verify explain returns EXPLAIN plan as string."""
    con = docs_views_gateway.con

    result = explain(con=con, view="docs.v_subsystem_profile", analyze=False)

    expect_is_instance(result, str)
    expect_true(len(result) > 0)


def test_explain_analyze_returns_plan_text(docs_views_gateway: StorageGateway) -> None:
    """Verify explain with analyze=True returns EXPLAIN ANALYZE plan."""
    con = docs_views_gateway.con

    result = explain(con=con, view="docs.v_subsystem_coverage", analyze=True)

    expect_is_instance(result, str)
    expect_true(len(result) > 0)


def test_run_profile_raises_on_missing_db(tmp_path: Path) -> None:
    """Verify run_profile raises FileNotFoundError for missing DB."""
    db_path = tmp_path / "nonexistent.duckdb"
    output_dir = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="DuckDB not found"):
        run_profile(db_path=db_path, output_dir=output_dir, analyze=False)


def test_run_profile_creates_artifacts(docs_profile_db: Path, tmp_path: Path) -> None:
    """Verify run_profile creates profiling artifacts with EXPLAIN plans."""
    output_dir = tmp_path / "profiling_output"

    run_profile(db_path=docs_profile_db, output_dir=output_dir, analyze=False)

    # Check profile_meta.json is created
    meta_file = output_dir / "profile_meta.json"
    expect_true(meta_file.exists())
    meta = json.loads(meta_file.read_text())
    expect_false(meta["analyze"])
    expect_in("docs.v_subsystem_profile", meta["views"])

    # Check explain files are created
    profile_explain = output_dir / "docs_v_subsystem_profile.explain.txt"
    coverage_explain = output_dir / "docs_v_subsystem_coverage.explain.txt"
    expect_true(profile_explain.exists())
    expect_true(coverage_explain.exists())
    expect_true(len(profile_explain.read_text()) > 0)
    expect_true(len(coverage_explain.read_text()) > 0)


def test_run_profile_analyze_mode_creates_artifacts(docs_profile_db: Path, tmp_path: Path) -> None:
    """Verify run_profile with analyze=True creates EXPLAIN ANALYZE artifacts."""
    output_dir = tmp_path / "profiling_output"

    run_profile(db_path=docs_profile_db, output_dir=output_dir, analyze=True)

    # Check meta shows analyze mode
    meta_file = output_dir / "profile_meta.json"
    meta = json.loads(meta_file.read_text())
    expect_true(meta["analyze"])

    # Check analyze files are created (not explain files)
    profile_analyze = output_dir / "docs_v_subsystem_profile.analyze.txt"
    coverage_analyze = output_dir / "docs_v_subsystem_coverage.analyze.txt"
    expect_true(profile_analyze.exists())
    expect_true(coverage_analyze.exists())


def test_main_with_missing_db_calls_parser_error(tmp_path: Path) -> None:
    """Verify run_profile propagates error for missing database."""
    db_path = tmp_path / "missing.duckdb"
    output_dir = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="DuckDB not found"):
        run_profile(db_path=db_path, output_dir=output_dir, analyze=False)


def test_main_with_valid_db_returns_success(docs_profile_db: Path, tmp_path: Path) -> None:
    """Verify run_profile succeeds on a valid database."""
    output_dir = tmp_path / "output"

    run_profile(db_path=docs_profile_db, output_dir=output_dir, analyze=False)


def test_main_with_analyze_flag(docs_profile_db: Path, tmp_path: Path) -> None:
    """Verify main processes --analyze flag correctly."""
    output_dir = tmp_path / "output"

    run_profile(db_path=docs_profile_db, output_dir=output_dir, analyze=True)

    # Verify analyze files were created
    expect_true((output_dir / "docs_v_subsystem_profile.analyze.txt").exists())
