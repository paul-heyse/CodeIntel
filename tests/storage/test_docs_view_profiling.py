"""Tests for docs view profiling module."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from codeintel.storage.helpers.profiling import (
    DOCS_VIEWS,
    explain,
    run_profile,
    write_text,
)
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.metadata import bootstrap_metadata_datasets
from codeintel.storage.schema import apply_all_schemas
from codeintel.storage.views import create_all_views


def test_docs_views_is_defined() -> None:
    """Verify DOCS_VIEWS constant is defined and non-empty."""
    assert isinstance(DOCS_VIEWS, tuple)
    assert len(DOCS_VIEWS) > 0


def test_docs_views_contains_expected_views() -> None:
    """Verify DOCS_VIEWS contains expected subsystem views."""
    assert "docs.v_subsystem_profile" in DOCS_VIEWS
    assert "docs.v_subsystem_coverage" in DOCS_VIEWS


def test_write_text_creates_file(tmp_path: Path) -> None:
    """Verify write_text creates file with correct content."""
    file_path = tmp_path / "test.txt"

    write_text(file_path, "Hello, World!")

    assert file_path.exists()
    assert file_path.read_text() == "Hello, World!"


def test_write_text_creates_nested_directories(tmp_path: Path) -> None:
    """Verify write_text creates parent directories."""
    file_path = tmp_path / "nested" / "dir" / "test.txt"

    write_text(file_path, "Nested content")

    assert file_path.exists()
    assert file_path.read_text() == "Nested content"


def test_explain_returns_plan_text(fresh_gateway: StorageGateway) -> None:
    """Verify explain returns EXPLAIN plan as string."""
    con = fresh_gateway.con

    result = explain(con=con, view="docs.v_subsystem_profile", analyze=False)

    assert isinstance(result, str)
    assert len(result) > 0


def test_explain_analyze_returns_plan_text(fresh_gateway: StorageGateway) -> None:
    """Verify explain with analyze=True returns EXPLAIN ANALYZE plan."""
    con = fresh_gateway.con

    result = explain(con=con, view="docs.v_subsystem_coverage", analyze=True)

    assert isinstance(result, str)
    assert len(result) > 0


def test_run_profile_raises_on_missing_db(tmp_path: Path) -> None:
    """Verify run_profile raises FileNotFoundError for missing DB."""
    db_path = tmp_path / "nonexistent.duckdb"
    output_dir = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="DuckDB not found"):
        run_profile(db_path=db_path, output_dir=output_dir, analyze=False)


def _create_test_db(db_path: Path) -> None:
    """Create a bootstrapped DuckDB with required schemas and views for profiling tests."""
    # Use ATTACH with STORAGE_VERSION v1.4.0 for typed macro support
    con = duckdb.connect(":memory:")
    try:
        # Attach file database with newer storage version
        con.execute(f"ATTACH DATABASE '{db_path}' AS test_db (STORAGE_VERSION 'v1.4.0')")
        # Switch to the attached database
        con.execute("USE test_db")

        # Apply all schemas to create a properly bootstrapped database
        apply_all_schemas(con)
        create_all_views(con)
        bootstrap_metadata_datasets(con, include_views=True)
    finally:
        con.close()


def test_run_profile_creates_artifacts(tmp_path: Path) -> None:
    """Verify run_profile creates profiling artifacts with EXPLAIN plans."""
    db_path = tmp_path / "test.duckdb"
    output_dir = tmp_path / "profiling_output"

    _create_test_db(db_path)

    run_profile(db_path=db_path, output_dir=output_dir, analyze=False)

    # Check profile_meta.json is created
    meta_file = output_dir / "profile_meta.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text())
    assert meta["analyze"] is False
    assert "docs.v_subsystem_profile" in meta["views"]

    # Check explain files are created
    profile_explain = output_dir / "docs_v_subsystem_profile.explain.txt"
    coverage_explain = output_dir / "docs_v_subsystem_coverage.explain.txt"
    assert profile_explain.exists()
    assert coverage_explain.exists()
    assert len(profile_explain.read_text()) > 0
    assert len(coverage_explain.read_text()) > 0


def test_run_profile_analyze_mode_creates_artifacts(tmp_path: Path) -> None:
    """Verify run_profile with analyze=True creates EXPLAIN ANALYZE artifacts."""
    db_path = tmp_path / "test.duckdb"
    output_dir = tmp_path / "profiling_output"

    _create_test_db(db_path)

    run_profile(db_path=db_path, output_dir=output_dir, analyze=True)

    # Check meta shows analyze mode
    meta_file = output_dir / "profile_meta.json"
    meta = json.loads(meta_file.read_text())
    assert meta["analyze"] is True

    # Check analyze files are created (not explain files)
    profile_analyze = output_dir / "docs_v_subsystem_profile.analyze.txt"
    coverage_analyze = output_dir / "docs_v_subsystem_coverage.analyze.txt"
    assert profile_analyze.exists()
    assert coverage_analyze.exists()


def test_main_with_missing_db_calls_parser_error(tmp_path: Path) -> None:
    """Verify run_profile propagates error for missing database."""
    db_path = tmp_path / "missing.duckdb"
    output_dir = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="DuckDB not found"):
        run_profile(db_path=db_path, output_dir=output_dir, analyze=False)


def test_main_with_valid_db_returns_success(tmp_path: Path) -> None:
    """Verify run_profile succeeds on a valid database."""
    db_path = tmp_path / "test.duckdb"
    output_dir = tmp_path / "output"

    _create_test_db(db_path)

    run_profile(db_path=db_path, output_dir=output_dir, analyze=False)


def test_main_with_analyze_flag(tmp_path: Path) -> None:
    """Verify main processes --analyze flag correctly."""
    db_path = tmp_path / "test.duckdb"
    output_dir = tmp_path / "output"

    _create_test_db(db_path)

    run_profile(db_path=db_path, output_dir=output_dir, analyze=True)

    # Verify analyze files were created
    assert (output_dir / "docs_v_subsystem_profile.analyze.txt").exists()
