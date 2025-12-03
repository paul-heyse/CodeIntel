"""Unit tests for change detection and file_state tracking."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.ingestion.adapters.duckdb_storage import DuckDBStorageAdapter
from codeintel.ingestion.adapters.hash_change_detection import HashChangeDetectionAdapter
from codeintel.ingestion.ports.change_detection import ChangeRequest
from codeintel.ingestion.ports.discovery import ModuleRecord
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import open_ingestion_gateway_with_macros as open_ingestion_gateway


def compute_changes(gateway: StorageGateway, request: ChangeRequest) -> object:
    """Compute changes using the adapter directly.

    Parameters
    ----------
    gateway
        Storage gateway for database operations.
    request
        Change detection request parameters.

    Returns
    -------
    ChangeSet
        Computed changes (added, modified, deleted modules).
    """
    storage = DuckDBStorageAdapter(gateway)
    adapter = HashChangeDetectionAdapter(storage)
    modules = getattr(request, "modules", []) or []
    return adapter.compute_changes(request, modules)


def test_compute_changes_tracks_add_modify_delete(tmp_path: Path) -> None:
    """Change detection should surface added, modified, and deleted modules.

    This test verifies realistic change detection behavior:
    - Added: file exists in current scan but not in previous state
    - Modified: file exists with different content/hash
    - Deleted: file was in previous state but not in current scan

    The key insight is that deletions are detected by ABSENCE from the current
    module list, not by passing a module record for a non-existent file.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    file_path = repo_root / "a.py"
    file_path.write_text("x = 1\n", encoding="utf8")

    gateway = open_ingestion_gateway()

    def make_record() -> ModuleRecord:
        return ModuleRecord(
            rel_path="a.py",
            module_name="mod",
            file_path=file_path,
            index=1,
            total=1,
        )

    def make_request(modules: list[ModuleRecord]) -> ChangeRequest:
        return ChangeRequest(
            repo="r",
            commit="c",
            repo_root=repo_root,
            modules=modules,
        )

    # First pass: file is new → should report as added
    first = compute_changes(gateway, make_request([make_record()]))
    if len(first.added) != 1 or first.modified or first.deleted:
        pytest.fail(f"Expected first pass to report one addition only, got {first}")

    # Second pass with same file → no changes
    unchanged = compute_changes(gateway, make_request([make_record()]))
    if unchanged.added or unchanged.modified or unchanged.deleted:
        pytest.fail(f"Expected no changes on second pass, got {unchanged}")

    # Third pass: modify file content → should detect modification
    file_path.write_text("x = 2\n", encoding="utf8")
    modified = compute_changes(gateway, make_request([make_record()]))
    if modified.added or len(modified.modified) != 1 or modified.deleted:
        pytest.fail(f"Expected single modification only, got {modified}")

    # Fourth pass: delete file and pass EMPTY module list
    # (simulating that the scanner no longer finds the file)
    file_path.unlink()
    deleted = compute_changes(gateway, make_request([]))  # Empty list = file not found
    if deleted.added or deleted.modified or len(deleted.deleted) != 1:
        pytest.fail(f"Expected single deletion only, got {deleted}")
