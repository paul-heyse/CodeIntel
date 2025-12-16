"""Tests for docs view profiling via Warehouse explain helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.warehouse import Warehouse
from tests._helpers import docs_views_ready_gateway
from tests._helpers.assertions import (
    expect_false,
    expect_in,
    expect_is_instance,
    expect_true,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


_DOCS_VIEWS = ("docs.v_subsystem_profile", "docs.v_subsystem_coverage")


@pytest.fixture
def docs_profile_db(tmp_path: Path) -> Path:
    """Provision a file-backed docs views database.

    Returns
    -------
    Path
        Path to the DuckDB database file on disk.
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


def test_explain_table_returns_plan_text(docs_views_gateway: StorageGateway) -> None:
    """Verify Warehouse.explain_table returns EXPLAIN plan text."""
    warehouse = Warehouse(docs_views_gateway)
    plan = warehouse.explain_table("docs.v_subsystem_profile", analyze=False)
    expect_is_instance(plan, str)
    expect_true(len(plan) > 0)


def test_explain_table_analyze_returns_plan_text(docs_views_gateway: StorageGateway) -> None:
    """Verify Warehouse.explain_table returns EXPLAIN ANALYZE plan text."""
    warehouse = Warehouse(docs_views_gateway)
    plan = warehouse.explain_table("docs.v_subsystem_coverage", analyze=True)
    expect_is_instance(plan, str)
    expect_true(len(plan) > 0)


def test_profile_views_creates_artifacts(docs_profile_db: Path, tmp_path: Path) -> None:
    """Verify Warehouse.profile_views creates metadata and plan artifacts."""
    output_dir = tmp_path / "profiling_output"
    gateway = open_gateway(StorageConfig.for_readonly(docs_profile_db))
    try:
        warehouse = Warehouse(gateway)
        warehouse.profile_views(
            views=_DOCS_VIEWS,
            output_dir=output_dir,
            analyze=False,
            db_path=docs_profile_db,
        )
    finally:
        gateway.close()

    meta_file = output_dir / "profile_meta.json"
    expect_true(meta_file.exists())
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    expect_false(meta["analyze"])
    expect_in(_DOCS_VIEWS[0], meta["views"])

    profile_explain = output_dir / "docs_v_subsystem_profile.explain.txt"
    coverage_explain = output_dir / "docs_v_subsystem_coverage.explain.txt"
    expect_true(profile_explain.exists())
    expect_true(coverage_explain.exists())
    expect_true(len(profile_explain.read_text(encoding="utf-8")) > 0)
    expect_true(len(coverage_explain.read_text(encoding="utf-8")) > 0)


def test_profile_views_analyze_mode_creates_artifacts(
    docs_profile_db: Path, tmp_path: Path
) -> None:
    """Verify Warehouse.profile_views writes analyze artifacts when enabled."""
    output_dir = tmp_path / "profiling_output"
    gateway = open_gateway(StorageConfig.for_readonly(docs_profile_db))
    try:
        warehouse = Warehouse(gateway)
        warehouse.profile_views(
            views=_DOCS_VIEWS,
            output_dir=output_dir,
            analyze=True,
            db_path=docs_profile_db,
        )
    finally:
        gateway.close()

    meta_file = output_dir / "profile_meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    expect_true(meta["analyze"])

    profile_analyze = output_dir / "docs_v_subsystem_profile.analyze.txt"
    coverage_analyze = output_dir / "docs_v_subsystem_coverage.analyze.txt"
    expect_true(profile_analyze.exists())
    expect_true(coverage_analyze.exists())
