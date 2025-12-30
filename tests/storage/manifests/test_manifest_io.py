"""Tests for manifest I/O helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.storage.manifests import (
    manifest_hash,
    read_manifest_json,
    validate_manifest_hash,
    write_manifest_json,
)


def test_manifest_hash_validation(tmp_path: Path) -> None:
    """Validate manifest hash round-trip and mismatch behavior."""
    payload = {"dataset": "core.demo", "rows": 3}
    path = tmp_path / "manifest.json"
    write_manifest_json(path, payload)

    expected = manifest_hash(payload)
    loaded = read_manifest_json(path, expected_hash=expected)
    assert loaded == payload

    validate_manifest_hash(payload, expected_hash=expected)
    with pytest.raises(ValueError, match="Manifest hash mismatch"):
        read_manifest_json(path, expected_hash="bad-hash")
