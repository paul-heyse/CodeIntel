"""Tests for Iceberg metadata stats helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
import pytest

from codeintel.storage.iceberg.stats import iceberg_stats_for_table

pytestmark = pytest.mark.no_runtime_env

EXPECTED_SNAPSHOT_ID = 10
EXPECTED_SCHEMA_ID = 3
EXPECTED_SNAPSHOT_COUNT = 2
EXPECTED_TOTAL_RECORDS = 5
EXPECTED_DATA_FILE_COUNT = 2
EXPECTED_TOTAL_BYTES = 100
EXPECTED_MANIFEST_COUNT = 4


@dataclass(frozen=True)
class _FakeSummary:
    additional_properties: dict[str, str]


@dataclass(frozen=True)
class _FakeSnapshot:
    snapshot_id: int
    schema_id: int | None
    summary: _FakeSummary | None


@dataclass(frozen=True)
class _FakeInspect:
    manifest_rows: int

    def manifests(self) -> pa.Table:
        return pa.table({"manifest": list(range(self.manifest_rows))})


@dataclass(frozen=True)
class _FakeMetadata:
    snapshots: tuple[object, ...]


@dataclass(frozen=True)
class _FakeTable:
    snapshot: _FakeSnapshot | None
    manifest_rows: int
    snapshots: tuple[object, ...] = ()

    def snapshot_by_id(self, snapshot_id: int) -> _FakeSnapshot | None:
        if self.snapshot is None:
            return None
        if self.snapshot.snapshot_id != snapshot_id:
            return None
        return self.snapshot

    def current_snapshot(self) -> _FakeSnapshot | None:
        return self.snapshot

    @property
    def metadata(self) -> _FakeMetadata:
        return _FakeMetadata(snapshots=self.snapshots)

    @property
    def inspect(self) -> _FakeInspect:
        return _FakeInspect(manifest_rows=self.manifest_rows)


def test_iceberg_stats_for_table_parses_summary() -> None:
    """Return stats from snapshot summary and manifest count."""
    summary = _FakeSummary(
        additional_properties={
            "total-records": "5",
            "total-data-files": "2",
            "total-files-size": "100",
        }
    )
    snapshot = _FakeSnapshot(
        snapshot_id=EXPECTED_SNAPSHOT_ID,
        schema_id=EXPECTED_SCHEMA_ID,
        summary=summary,
    )
    table = _FakeTable(
        snapshot=snapshot,
        manifest_rows=EXPECTED_MANIFEST_COUNT,
        snapshots=(object(), object()),
    )

    stats = iceberg_stats_for_table(table, snapshot_id=EXPECTED_SNAPSHOT_ID)

    assert "snapshot_id" in stats
    assert stats["snapshot_id"] == EXPECTED_SNAPSHOT_ID
    assert "schema_id" in stats
    assert stats["schema_id"] == EXPECTED_SCHEMA_ID
    assert "snapshot_count" in stats
    assert stats["snapshot_count"] == EXPECTED_SNAPSHOT_COUNT
    assert "total_records" in stats
    assert stats["total_records"] == EXPECTED_TOTAL_RECORDS
    assert "data_file_count" in stats
    assert stats["data_file_count"] == EXPECTED_DATA_FILE_COUNT
    assert "total_bytes" in stats
    assert stats["total_bytes"] == EXPECTED_TOTAL_BYTES
    assert "manifest_count" in stats
    assert stats["manifest_count"] == EXPECTED_MANIFEST_COUNT


def test_iceberg_stats_for_table_empty_snapshot() -> None:
    """Return empty stats when no snapshot is available."""
    table = _FakeTable(snapshot=None, manifest_rows=0)
    assert iceberg_stats_for_table(table) == {}
