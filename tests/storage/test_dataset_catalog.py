"""Tests for dataset catalog generation without DuckDB dependencies."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import duckdb
import pytest

from codeintel.cli.main import run_datasets_catalog
from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.catalog import build_catalog, write_html_catalog, write_markdown_catalog
from codeintel.storage.datasets import Dataset, DatasetRegistry


def _sample_registry() -> DatasetRegistry:
    table_key = "core.ast_nodes"
    schema = TABLE_SCHEMAS[table_key]
    dataset = Dataset(
        table_key=table_key,
        name="ast_nodes",
        schema=schema,
        json_schema_id="ast_nodes",
        jsonl_filename="ast_nodes.jsonl",
        parquet_filename="ast_nodes.parquet",
        owner="team-data",
        freshness_sla="daily",
        retention_policy="90d",
        schema_version="1",
        stable_id="ast_nodes",
        upstream_dependencies=("core.modules",),
        validation_profile="strict",
    )
    return DatasetRegistry(
        by_name={"ast_nodes": dataset},
        by_table_key={table_key: dataset},
        jsonl_datasets={table_key: "ast_nodes.jsonl"},
        parquet_datasets={table_key: "ast_nodes.parquet"},
    )


def test_catalog_generation_writes_files(tmp_path: Path) -> None:
    """Catalog generation writes both Markdown and HTML outputs."""
    registry = _sample_registry()
    entries = build_catalog(registry, con=None, sample_rows=0)
    md_path = write_markdown_catalog(tmp_path, entries)
    html_path = write_html_catalog(tmp_path, entries)

    if not md_path.exists():
        pytest.fail("Markdown catalog was not written")
    if not html_path.exists():
        pytest.fail("HTML catalog was not written")
    content = md_path.read_text(encoding="utf-8")
    if "Dataset Catalog" not in content:
        pytest.fail("Catalog header missing from Markdown output")
    if not any(entry.name in content for entry in entries):
        pytest.fail("Dataset names missing from Markdown output")
    if "- [ast_nodes](#ast_nodes)" not in content:
        pytest.fail("Markdown navigation links missing")
    html_content = html_path.read_text(encoding="utf-8")
    if "<nav><ul>" not in html_content or "#ast_nodes" not in html_content:
        pytest.fail("HTML navigation anchors missing")


def test_catalog_handles_missing_samples(tmp_path: Path) -> None:
    """Catalog includes placeholder text when no samples are available."""
    registry = _sample_registry()
    entries = build_catalog(registry, con=None, sample_rows=0)
    path = write_markdown_catalog(tmp_path, entries)
    data = path.read_text(encoding="utf-8")
    if "_No sample rows available._" not in data:
        pytest.fail("Placeholder for sample rows was not rendered")


def test_catalog_sampling_gracefully_falls_back(tmp_path: Path) -> None:
    """Sampling errors should not crash catalog generation."""
    registry = _sample_registry()
    con = duckdb.connect(database=":memory:")
    warnings: list[str] = []
    entries = build_catalog(
        registry,
        con=con,
        sample_rows=2,
        sample_rows_strict=False,
        warn=warnings.append,
    )
    if entries[0].sample_rows:
        pytest.fail("Sample rows should be empty when sampling returns nothing")
    if not warnings:
        pytest.fail("Sampling fallback should produce a warning")
    path = write_markdown_catalog(tmp_path, entries)
    data = path.read_text(encoding="utf-8")
    if "_No sample rows available._" not in data:
        pytest.fail("Fallback placeholder missing after sampling failure")


def test_catalog_sampling_strict_raises() -> None:
    """Strict sampling should raise when macros are unavailable."""
    registry = _sample_registry()
    con = duckdb.connect(database=":memory:")
    with pytest.raises(RuntimeError):
        build_catalog(
            registry,
            con=con,
            sample_rows=1,
            sample_rows_strict=True,
        )


def test_catalog_missing_db_writes_empty(tmp_path: Path) -> None:
    """When DB is missing and non-strict, write empty catalog with warning."""
    db_path = tmp_path / "missing.duckdb"
    output_dir = tmp_path / "catalog"
    args = Namespace(
        db_path=db_path,
        sample_rows=1,
        sample_rows_strict=False,
        output_dir=output_dir,
        repo_root=tmp_path,
        repo="demo/repo",
        commit="deadbeef",
        build_dir=tmp_path / "build",
        document_output_dir=tmp_path / "Document Output",
    )
    exit_code = run_datasets_catalog(args)
    if exit_code != 0:
        pytest.fail("Catalog should succeed when DB is missing in non-strict mode")
    if not (output_dir / "catalog.md").exists():
        pytest.fail("Catalog markdown not written for missing DB")
    content = (output_dir / "catalog.md").read_text(encoding="utf-8")
    if "Dataset Catalog" not in content:
        pytest.fail("Empty catalog header missing")


def test_catalog_missing_db_strict_fails(tmp_path: Path) -> None:
    """When DB is missing and strict, exit non-zero."""
    db_path = tmp_path / "missing.duckdb"
    args = Namespace(
        db_path=db_path,
        sample_rows=1,
        sample_rows_strict=True,
        output_dir=tmp_path / "catalog",
        repo_root=tmp_path,
        repo="demo/repo",
        commit="deadbeef",
        build_dir=tmp_path / "build",
        document_output_dir=tmp_path / "Document Output",
    )
    exit_code = run_datasets_catalog(args)
    if exit_code == 0:
        pytest.fail("Strict mode should fail when DB is missing")
