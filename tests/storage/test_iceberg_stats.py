"""Tests for Iceberg metadata stats helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, cast

import pyarrow as pa
import pytest
from pyiceberg.table.puffin import MAGIC_BYTES
from pyiceberg.table.statistics import StatisticsFile

from codeintel.storage.iceberg.statistics_file import persist_iceberg_statistics
from codeintel.storage.iceberg.stats import iceberg_stats_for_table

if TYPE_CHECKING:
    from pyiceberg.table import Table

pytestmark = pytest.mark.no_runtime_env

EXPECTED_SNAPSHOT_ID = 10
EXPECTED_SCHEMA_ID = 3
EXPECTED_SNAPSHOT_COUNT = 2
EXPECTED_TOTAL_RECORDS = 5
EXPECTED_DATA_FILE_COUNT = 2
EXPECTED_TOTAL_BYTES = 100
EXPECTED_MANIFEST_COUNT = 4


def _read_puffin_properties(path: Path) -> dict[str, str]:
    payload = path.read_bytes()
    magic_len = len(MAGIC_BYTES)
    flags_len = 4
    if not payload.startswith(MAGIC_BYTES) or not payload.endswith(MAGIC_BYTES):
        return {}
    footer_size_start = len(payload) - (magic_len + flags_len + 4)
    footer_size_end = len(payload) - (magic_len + flags_len)
    footer_size = int.from_bytes(payload[footer_size_start:footer_size_end], "little")
    footer_start = magic_len + flags_len
    footer_end = footer_start + footer_size
    footer_payload = payload[footer_start:footer_end]
    footer = json.loads(footer_payload.decode("utf-8"))
    props = footer.get("properties")
    if isinstance(props, dict):
        return {str(key): str(value) for key, value in props.items()}
    return {}


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


class _FakeLocationProvider:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self.last_location: Path | None = None

    def new_metadata_location(self, file_name: str) -> str:
        path = self._base_dir / file_name
        self.last_location = path
        return str(path)


class _FakeOutput:
    def __init__(self, path: Path) -> None:
        self._path = path

    def create(self, *, overwrite: bool) -> IO[bytes]:
        if self._path.exists() and not overwrite:
            msg = f"Output already exists: {self._path}"
            raise OSError(msg)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return self._path.open("wb")


class _FakeIO:
    @staticmethod
    def new_output(location: str) -> _FakeOutput:
        return _FakeOutput(Path(location))


@dataclass
class _FakeUpdateStatistics:
    stats_file: StatisticsFile | None = None
    committed: bool = False

    def set_statistics(self, stats_file: StatisticsFile) -> _FakeUpdateStatistics:
        self.stats_file = stats_file
        return self

    def commit(self) -> None:
        self.committed = True


class _FakeStatsTable:
    def __init__(self, base_dir: Path) -> None:
        self.io = _FakeIO()
        self._location_provider = _FakeLocationProvider(base_dir)
        self.update_ctx = _FakeUpdateStatistics()

    def location_provider(self) -> _FakeLocationProvider:
        return self._location_provider

    def update_statistics(self) -> _FakeUpdateStatistics:
        return self.update_ctx


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


def test_persist_iceberg_statistics_writes_puffin(tmp_path: Path) -> None:
    """Persist derived stats as a puffin statistics file."""
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    table = _FakeStatsTable(stats_dir)
    stats = {"snapshot_id": 42, "schema_id": 7, "total_records": 3}
    snapshot_properties = {"schema_hash": "hash-1"}

    stats_file = persist_iceberg_statistics(
        table=cast("Table", table),
        table_key="core.modules",
        stats=stats,
        snapshot_properties=snapshot_properties,
    )

    assert stats_file is not None
    assert table.update_ctx.committed is True
    location = table.location_provider().last_location
    assert location is not None
    assert location.exists()
    properties = _read_puffin_properties(location)
    assert properties.get("codeintel.table_key") == "core.modules"
    assert properties.get("codeintel.stats.snapshot_id") == "42"
    assert properties.get("codeintel.stats.schema_id") == "7"
    assert properties.get("codeintel.snapshot.schema_hash") == "hash-1"
