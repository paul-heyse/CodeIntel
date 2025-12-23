"""Tests for ResourceStore TTL cleanup behavior."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.serving.errors import ExportNotFoundError
from codeintel.serving.mcp.resource_store import ExportArtifactSpec, ResourceStore
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true

if TYPE_CHECKING:
    from pathlib import Path


def test_resource_store_cleanup_expired_deletes_artifacts(tmp_path: Path) -> None:
    """Delete expired exports based on metadata expires_at timestamps."""
    store = ResourceStore(tmp_path / "exports", ttl_seconds=3600)
    rows: list[dict[str, object]] = [{"id": 1}, {"id": 2}]
    spec = ExportArtifactSpec(view_id="demo.view", format="jsonl")
    token, _artifact, _meta = store.put_with_metadata(rows, spec=spec)

    meta_path = tmp_path / "exports" / f"{token}.meta.json"
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw["expires_at"] = datetime(2000, 1, 1, tzinfo=UTC).isoformat()
    meta_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")

    deleted = store.cleanup_expired()
    expect_equal(deleted, 1)
    expect_true(not meta_path.exists(), message="Expected expired export metadata to be deleted")

    with pytest.raises(ExportNotFoundError):
        store.get(token)
