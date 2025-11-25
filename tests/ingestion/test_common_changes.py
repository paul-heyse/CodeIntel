"""Unit tests for change detection and file_state tracking."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.ingestion.common import ChangeRequest, ModuleRecord, compute_changes
from tests._helpers.gateway import open_ingestion_gateway


def test_compute_changes_tracks_add_modify_delete(tmp_path: Path) -> None:
    """Change detection should surface added, modified, and deleted modules."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    file_path = repo_root / "a.py"
    file_path.write_text("x = 1\n", encoding="utf8")

    gateway = open_ingestion_gateway()
    record = ModuleRecord(
        rel_path="a.py",
        module_name="mod",
        file_path=file_path,
        index=1,
        total=1,
    )
    request = ChangeRequest(
        repo="r",
        commit="c",
        repo_root=repo_root,
        modules=[record],
    )

    first = compute_changes(gateway, request)
    if len(first.added) != 1 or first.modified or first.deleted:
        pytest.fail(f"Expected first pass to report one addition only, got {first}")

    unchanged = compute_changes(gateway, request)
    if unchanged.added or unchanged.modified or unchanged.deleted:
        pytest.fail(f"Expected no changes on second pass, got {unchanged}")

    file_path.write_text("x = 2\n", encoding="utf8")
    modified = compute_changes(gateway, request)
    if modified.added or len(modified.modified) != 1 or modified.deleted:
        pytest.fail(f"Expected single modification only, got {modified}")

    file_path.unlink()
    deleted = compute_changes(gateway, request)
    if deleted.added or deleted.modified or len(deleted.deleted) != 1:
        pytest.fail(f"Expected single deletion only, got {deleted}")
