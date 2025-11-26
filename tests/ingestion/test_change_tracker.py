"""Unit tests for ChangeTracker dataset views."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.ingestion.change_tracker import ChangeTracker, IncrementalIngestPolicy
from codeintel.ingestion.common import ChangeRequest, ChangeSet, ModuleRecord
from codeintel.storage.gateway import StorageGateway


def _module(rel_path: str) -> ModuleRecord:
    return ModuleRecord(
        rel_path=rel_path,
        module_name=rel_path.replace("/", ".").removesuffix(".py"),
        file_path=Path(rel_path),
        index=0,
        total=0,
    )


def test_view_for_dataset_incremental(fresh_gateway: StorageGateway) -> None:
    """Use incremental mode when change ratios remain below thresholds."""
    modules = [_module("a.py"), _module("b.py"), _module("c.py")]
    tracker = ChangeTracker(
        gateway=fresh_gateway,
        change_request=ChangeRequest(
            repo="repo",
            commit="deadbeef",
            repo_root=Path("repo"),
            modules=modules,
        ),
        modules=modules,
        change_set=ChangeSet(added=[], modified=[modules[1]], deleted=[]),
        policy=IncrementalIngestPolicy(min_total_modules_for_ratio=1),
    )

    view = tracker.view_for_dataset(dataset_name="test", module_filter=None)

    if view.use_full_rebuild:
        pytest.fail("Expected incremental mode")
    if view.to_reparse != [modules[1]]:
        pytest.fail("Unexpected modules selected for reparse")
    if view.deleted_paths:
        pytest.fail("Expected no deleted paths")


def test_view_for_dataset_full_rebuild_when_changed_ratio_exceeds_policy(
    fresh_gateway: StorageGateway,
) -> None:
    """Trigger full rebuild when change ratio exceeds policy limits."""
    modules = [_module("a.py"), _module("b.py"), _module("c.py")]
    tracker = ChangeTracker(
        gateway=fresh_gateway,
        change_request=ChangeRequest(
            repo="repo",
            commit="deadbeef",
            repo_root=Path("repo"),
            modules=modules,
        ),
        modules=modules,
        change_set=ChangeSet(added=[modules[0]], modified=[modules[1]], deleted=[]),
        policy=IncrementalIngestPolicy(max_changed_ratio=0.5, min_total_modules_for_ratio=1),
    )

    view = tracker.view_for_dataset(dataset_name="test", module_filter=None)

    if not view.use_full_rebuild:
        pytest.fail("Expected full rebuild")
    if view.to_reparse != modules:
        pytest.fail("Expected all modules to reparse on full rebuild")
    expected_deleted = [module.rel_path for module in modules]
    if view.deleted_paths != expected_deleted:
        pytest.fail("Deleted paths did not match expected full rebuild set")


def test_view_for_dataset_respects_module_filter_and_deleted_paths(
    fresh_gateway: StorageGateway,
) -> None:
    """Apply module filter and ignore deletions outside the filtered set."""
    modules = [_module("src/a.py"), _module("src/b.txt"), _module("tests/c.py")]
    tracker = ChangeTracker(
        gateway=fresh_gateway,
        change_request=ChangeRequest(
            repo="repo",
            commit="deadbeef",
            repo_root=Path("repo"),
            modules=modules,
        ),
        modules=modules,
        change_set=ChangeSet(added=[modules[0]], modified=[], deleted=[modules[2]]),
        policy=IncrementalIngestPolicy(min_total_modules_for_ratio=10),
    )

    view = tracker.view_for_dataset(
        dataset_name="test",
        module_filter=lambda module: module.rel_path.endswith(".py")
        and module.rel_path.startswith("src/"),
    )

    if view.use_full_rebuild:
        pytest.fail("Expected incremental mode under filter")
    if view.to_reparse != [modules[0]]:
        pytest.fail("Filter should only select src Python modules")
    if view.deleted_paths:
        pytest.fail("Deleted paths outside filter should be ignored")


def test_view_for_dataset_full_rebuild_flag_forces_rebuild(
    fresh_gateway: StorageGateway,
) -> None:
    """Force full rebuild when change request flag is set."""
    modules = [_module("a.py"), _module("b.py")]
    tracker = ChangeTracker(
        gateway=fresh_gateway,
        change_request=ChangeRequest(
            repo="repo",
            commit="deadbeef",
            repo_root=Path("repo"),
            modules=modules,
            full_rebuild=True,
        ),
        modules=modules,
        change_set=ChangeSet(added=[], modified=[modules[0]], deleted=[]),
        policy=IncrementalIngestPolicy(),
    )

    view = tracker.view_for_dataset(dataset_name="test", module_filter=None)

    if not view.use_full_rebuild:
        pytest.fail("Expected full rebuild via flag")
    if view.to_reparse != modules:
        pytest.fail("Full rebuild should reparse all modules")
    expected_deleted = [module.rel_path for module in modules]
    if view.deleted_paths != expected_deleted:
        pytest.fail("Full rebuild should delete all module paths")
