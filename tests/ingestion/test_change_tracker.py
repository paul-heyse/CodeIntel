from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from codeintel.ingestion.change_tracker import ChangeTracker, IncrementalIngestPolicy
from codeintel.ingestion.common import ChangeRequest, ChangeSet, ModuleRecord


def _module(rel_path: str) -> ModuleRecord:
    return ModuleRecord(
        rel_path=rel_path,
        module_name=rel_path.replace("/", ".").removesuffix(".py"),
        file_path=Path(rel_path),
        index=0,
        total=0,
    )


def test_view_for_dataset_incremental() -> None:
    modules = [_module("a.py"), _module("b.py"), _module("c.py")]
    tracker = ChangeTracker(
        gateway=Mock(),
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

    assert view.use_full_rebuild is False
    assert view.to_reparse == [modules[1]]
    assert view.deleted_paths == []


def test_view_for_dataset_full_rebuild_when_changed_ratio_exceeds_policy() -> None:
    modules = [_module("a.py"), _module("b.py"), _module("c.py")]
    tracker = ChangeTracker(
        gateway=Mock(),
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

    assert view.use_full_rebuild is True
    assert view.to_reparse == modules
    assert view.deleted_paths == [module.rel_path for module in modules]
