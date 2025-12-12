"""Tests for dataset catalog generation without DuckDB dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.config.datasets import DatasetContract
from codeintel.storage.datasets.catalog import (
    SamplingConfig,
    build_catalog,
    write_html_catalog,
    write_markdown_catalog,
)
from codeintel.storage.datasets.registry import DatasetRegistry
from tests._helpers.dataset_factories import sample_dataset_registry

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway

SAMPLE_ROWS_1 = 1
SAMPLE_ROWS_2 = 2


def test_catalog_generation_writes_files(tmp_path: Path) -> None:
    """Catalog generation writes both Markdown and HTML outputs."""
    registry = sample_dataset_registry(tmp_path)
    entries = build_catalog(registry, con=None, sampling=SamplingConfig(sample_rows=0))
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
    registry = sample_dataset_registry(tmp_path)
    entries = build_catalog(registry, con=None, sampling=SamplingConfig(sample_rows=0))
    path = write_markdown_catalog(tmp_path, entries)
    data = path.read_text(encoding="utf-8")
    if "_No sample rows available._" not in data:
        pytest.fail("Placeholder for sample rows was not rendered")


def test_catalog_sampling_gracefully_falls_back(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Sampling errors should not crash catalog generation."""
    registry = sample_dataset_registry(tmp_path)
    warnings: list[str] = []
    entries = build_catalog(
        registry,
        con=fresh_gateway.con,
        sampling=SamplingConfig(sample_rows=SAMPLE_ROWS_2, sample_rows_strict=False),
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


def test_catalog_sampling_strict_raises(fresh_gateway: StorageGateway) -> None:
    """Strict sampling should raise when the target table is missing."""
    contract = DatasetContract(table_key="core.this_table_does_not_exist", name="missing", schema=None)
    registry = DatasetRegistry(
        by_name={"missing": contract},
        by_table_key={contract.table_key: contract},
        jsonl_datasets={},
        parquet_datasets={},
    )
    with pytest.raises(RuntimeError):
        build_catalog(
            registry,
            con=fresh_gateway.con,
            sampling=SamplingConfig(sample_rows=SAMPLE_ROWS_1, sample_rows_strict=True),
        )
