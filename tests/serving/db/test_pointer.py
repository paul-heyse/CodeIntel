"""Tests for serving snapshot pointer handling."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.serving.db.pointer import ServingSnapshotPointer
from tests._helpers.assertions.expectation_assertions import expect_equal

if TYPE_CHECKING:
    from pathlib import Path


def test_pointer_roundtrip(tmp_path: Path) -> None:
    """Round-trip serialize and load pointer JSON."""
    now = datetime.now(tz=UTC)
    pointer = ServingSnapshotPointer(
        db_path=tmp_path / "codeintel.duckdb",
        semantic_registry_path=tmp_path / "semantic_registry.json",
        schema_manifest_path=tmp_path / "schema_manifest.json",
        buildspec_path=tmp_path / "buildspec.json",
        repo="demo/repo",
        commit="deadbeef",
        run_id="run-1",
        published_at=now,
        semantic_layer_version="v123",
    )

    pointer_path = tmp_path / "current.json"
    pointer_path.write_text(pointer.to_json(), encoding="utf-8")

    loaded = ServingSnapshotPointer.load(pointer_path)
    expect_equal(loaded, pointer)


def test_pointer_load_requires_published_at(tmp_path: Path) -> None:
    """Pointer load fails when published_at is missing."""
    payload = {
        "db_path": str(tmp_path / "codeintel.duckdb"),
        "semantic_registry_path": str(tmp_path / "semantic_registry.json"),
        "schema_manifest_path": str(tmp_path / "schema_manifest.json"),
        "buildspec_path": str(tmp_path / "buildspec.json"),
        "repo": "demo/repo",
        "commit": "deadbeef",
        "run_id": "run-1",
        "semantic_layer_version": "v123",
    }
    pointer_path = tmp_path / "current.json"
    pointer_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KeyError, match="Pointer missing published_at"):
        ServingSnapshotPointer.load(pointer_path)
