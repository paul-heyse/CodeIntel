"""Tests for ManifestStore implementations."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from codeintel.core.plugins.execution.manifest_store import (
    DuckDBManifestStore,
    InMemoryManifestStore,
)
from codeintel.core.plugins.types.result import PluginExecutionRecord
from tests._helpers.assertions import (
    expect_equal,
    expect_is_none,
    expect_is_not_none,
    expect_true,
)


def _record(plugin: str, *, repo: str = "owner/repo", commit: str = "abc") -> PluginExecutionRecord:
    now = datetime.now(tz=UTC)
    return PluginExecutionRecord(
        plugin_name=plugin,
        status="succeeded",
        started_at=now,
        ended_at=now,
        duration_ms=10.0,
        meta={
            "repo": repo,
            "commit": commit,
            "scope_id": None,
            "variant": None,
            "input_hash": "input123",
            "options_hash": "opts123",
        },
    )


def test_inmemory_manifest_store_round_trip() -> None:
    """Verify InMemoryManifestStore stores and retrieves latest record."""
    store = InMemoryManifestStore()
    rec = _record("plugin.a")
    store.append_record(rec)
    loaded = store.load_last_record(
        plugin_name="plugin.a",
        repo="owner/repo",
        commit="abc",
        scope_id=None,
        variant=None,
    )
    loaded_rec = expect_is_not_none(loaded)
    expect_equal(loaded_rec.plugin_name, rec.plugin_name)
    expect_equal(loaded_rec.meta.get("input_hash"), "input123")


@pytest.fixture
def duckdb_store(tmp_path: Path) -> DuckDBManifestStore:
    """Create a DuckDBManifestStore backed by temp file.

    Returns
    -------
    DuckDBManifestStore
        Store instance with schema ensured.
    """
    con = duckdb.connect(database=str(tmp_path / "manifest.duckdb"))
    store = DuckDBManifestStore(con)
    store.ensure_schema()
    return store


def test_duckdb_manifest_store_round_trip(duckdb_store: DuckDBManifestStore) -> None:
    """Verify DuckDBManifestStore persists and reads back records."""
    rec = _record("plugin.duck")
    duckdb_store.append_record(rec)
    loaded = duckdb_store.load_last_record(
        plugin_name="plugin.duck",
        repo="owner/repo",
        commit="abc",
        scope_id=None,
        variant=None,
    )
    loaded_rec = expect_is_not_none(loaded)
    expect_equal(loaded_rec.plugin_name, rec.plugin_name)
    expect_equal(loaded_rec.meta.get("options_hash"), "opts123")
    expect_true(loaded_rec.duration_ms > 0)


def test_duckdb_manifest_store_returns_none_when_missing(
    duckdb_store: DuckDBManifestStore,
) -> None:
    """Verify None is returned when no record matches."""
    result = duckdb_store.load_last_record(
        plugin_name="missing",
        repo="owner/repo",
        commit="abc",
        scope_id=None,
        variant=None,
    )
    expect_is_none(result)
