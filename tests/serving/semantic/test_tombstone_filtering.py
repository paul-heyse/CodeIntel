"""Tests for tombstone filtering helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from sqlglot import exp, parse_one

from codeintel.core.config.settings import IcebergSettings
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.semantic import tombstones as tombstones_module
from codeintel.serving.semantic.iceberg_scans import IcebergScanError
from codeintel.serving.semantic.tombstones import (
    TombstoneScanContext,
    apply_tombstone_filter,
    apply_tombstone_filter_lazyframe,
)


def test_apply_tombstone_filter_is_idempotent() -> None:
    """Avoid duplicating tombstone predicates on repeated calls."""
    ast = cast("exp.Select", parse_one('SELECT * FROM "core"."modules"', read="duckdb"))
    filtered = apply_tombstone_filter(
        ast,
        table_key="core.modules",
        primary_key=("module",),
        snapshot_id=42,
    )
    sql = filtered.sql(dialect="duckdb")
    assert "NOT EXISTS" in sql
    assert "snapshot_id <= 42" in sql
    second = apply_tombstone_filter(
        filtered,
        table_key="core.modules",
        primary_key=("module",),
        snapshot_id=42,
    )
    assert second.sql(dialect="duckdb") == sql


def test_apply_tombstone_filter_lazyframe_noop_when_disabled(tmp_path: Path) -> None:
    """Return the original LazyFrame when tombstones are disabled."""
    pl = pytest.importorskip("polars")
    lazyframe = pl.DataFrame({"module": ["m1", "m2"]}).lazy()
    settings = IcebergSettings(tombstones_enabled=False, read_enabled=True)
    pointer = ServingSnapshotPointer(
        snapshot_root=tmp_path / "snap",
        snapshot_manifest_path=tmp_path / "snap" / "snapshot_manifest.json",
        db_path=tmp_path / "snap" / "codeintel.duckdb",
        semantic_registry_path=tmp_path / "snap" / "semantic_registry.json",
        schema_manifest_path=tmp_path / "snap" / "schema_manifest.json",
        buildspec_path=tmp_path / "snap" / "buildspec.json",
        repo="org/repo",
        commit="abc",
        run_id="run-1",
        published_at=datetime.now(tz=UTC),
        semantic_layer_version="v1",
    )
    result = apply_tombstone_filter_lazyframe(
        lazyframe,
        table_key="core.modules",
        primary_key=("module",),
        snapshot_id=42,
        context=TombstoneScanContext(
            pointer=pointer,
            settings=settings,
            batch_size=1000,
        ),
    )
    assert result is lazyframe


def test_apply_tombstone_filter_lazyframe_warns_on_missing_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warn when tombstone scans fail and return original LazyFrame."""
    pl = pytest.importorskip("polars")
    lazyframe = pl.DataFrame({"module": ["m1", "m2"]}).lazy()
    settings = IcebergSettings(tombstones_enabled=True, read_enabled=True)
    pointer = ServingSnapshotPointer(
        snapshot_root=tmp_path / "snap",
        snapshot_manifest_path=tmp_path / "snap" / "snapshot_manifest.json",
        db_path=tmp_path / "snap" / "codeintel.duckdb",
        semantic_registry_path=tmp_path / "snap" / "semantic_registry.json",
        schema_manifest_path=tmp_path / "snap" / "schema_manifest.json",
        buildspec_path=tmp_path / "snap" / "buildspec.json",
        repo="org/repo",
        commit="abc",
        run_id="run-1",
        published_at=datetime.now(tz=UTC),
        semantic_layer_version="v1",
    )

    def _raise_scan(*_: object, **__: object) -> None:
        raise IcebergScanError

    monkeypatch.setattr(tombstones_module, "iceberg_scan_for_query", _raise_scan)
    caplog.set_level("WARNING")

    result = apply_tombstone_filter_lazyframe(
        lazyframe,
        table_key="core.modules",
        primary_key=("module",),
        snapshot_id=42,
        context=TombstoneScanContext(
            pointer=pointer,
            settings=settings,
            batch_size=1000,
        ),
    )

    assert result is lazyframe
    assert any("Tombstone scan failed" in record.message for record in caplog.records)
