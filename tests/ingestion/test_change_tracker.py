"""Unit tests for ChangeTracker dataset views."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.ingestion import (
    DocstringsExtractStep,
    DuckDBStorageAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.ports.change_detection import ChangeRequest, ChangeSet
from codeintel.ingestion.ports.discovery import ModuleRecord
from codeintel.ingestion.tracker import ChangeTracker, IncrementalIngestPolicy
from codeintel.storage.gateway import StorageGateway
from tests._helpers.factories import make_snapshot
from tests._helpers.ingestion import (
    ScanSetupOptions,
    make_scan_setup,
    module_records_for_paths,
    seed_inventory_from_paths,
)


def _modules(paths: list[str], *, repo_root: Path) -> list[ModuleRecord]:
    """Build ModuleRecord instances for tests.

    Returns
    -------
    list[ModuleRecord]
    Module records derived from the provided relative paths.
    """
    return module_records_for_paths(paths, repo_root)


def _seed_inventory(
    gateway: StorageGateway,
    repo_root: Path,
    repo: str,
    commit: str,
    paths: list[str],
) -> None:
    """Seed core.modules and repo_map for consistency with module lists."""
    seed_inventory_from_paths(
        repo_root=repo_root,
        gateway=gateway,
        repo=repo,
        commit=commit,
        paths=paths,
    )


def _compute_changes(gateway: StorageGateway, request: ChangeRequest) -> ChangeSet:
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


def test_view_for_dataset_incremental(tmp_path: Path) -> None:
    """Use incremental mode when change ratios remain below thresholds."""
    setup = make_scan_setup(
        tmp_path,
        options=ScanSetupOptions(
            repo_structure={
                "a.py": "x = 1\n",
                "b.py": "y = 2\n",
                "c.py": "z = 3\n",
            }
        ),
    )
    modules = _modules(["a.py", "b.py", "c.py"], repo_root=setup.repo_root)
    _seed_inventory(
        setup.gateway,
        setup.repo_root,
        repo="repo",
        commit="deadbeef",
        paths=[module.rel_path for module in modules],
    )
    tracker = ChangeTracker(
        gateway=setup.gateway,
        change_request=ChangeRequest(
            repo="repo",
            commit="deadbeef",
            repo_root=setup.repo_root,
            modules=modules,
        ),
        modules=modules,
        change_set=ChangeSet(added=[], modified=[modules[1]], deleted=[]),
        policy=IncrementalIngestPolicy(min_total_modules_for_ratio=1),
    )

    try:
        view = tracker.view_for_dataset(dataset_name="test", module_filter=None)

        if view.use_full_rebuild:
            pytest.fail("Expected incremental mode")
        if view.to_reparse != [modules[1]]:
            pytest.fail("Unexpected modules selected for reparse")
        if view.deleted_paths:
            pytest.fail("Expected no deleted paths")
    finally:
        setup.gateway.close()


def test_view_for_dataset_full_rebuild_when_changed_ratio_exceeds_policy(
    tmp_path: Path,
) -> None:
    """Trigger full rebuild when change ratio exceeds policy limits."""
    setup = make_scan_setup(
        tmp_path,
        options=ScanSetupOptions(
            repo_structure={
                "a.py": "x = 1\n",
                "b.py": "y = 2\n",
                "c.py": "z = 3\n",
            }
        ),
    )
    modules = _modules(["a.py", "b.py", "c.py"], repo_root=setup.repo_root)
    _seed_inventory(
        setup.gateway,
        setup.repo_root,
        repo="repo",
        commit="deadbeef",
        paths=[module.rel_path for module in modules],
    )
    tracker = ChangeTracker(
        gateway=setup.gateway,
        change_request=ChangeRequest(
            repo="repo",
            commit="deadbeef",
            repo_root=setup.repo_root,
            modules=modules,
        ),
        modules=modules,
        change_set=ChangeSet(added=[modules[0]], modified=[modules[1]], deleted=[]),
        policy=IncrementalIngestPolicy(max_changed_ratio=0.5, min_total_modules_for_ratio=1),
    )

    try:
        view = tracker.view_for_dataset(dataset_name="test", module_filter=None)

        if not view.use_full_rebuild:
            pytest.fail("Expected full rebuild")
        if view.to_reparse != modules:
            pytest.fail("Expected all modules to reparse on full rebuild")
        expected_deleted = [module.rel_path for module in modules]
        if view.deleted_paths != expected_deleted:
            pytest.fail("Deleted paths did not match expected full rebuild set")
    finally:
        setup.gateway.close()


def test_view_for_dataset_respects_module_filter_and_deleted_paths(
    tmp_path: Path,
) -> None:
    """Apply module filter and ignore deletions outside the filtered set."""
    setup = make_scan_setup(
        tmp_path,
        options=ScanSetupOptions(
            repo_structure={
                "src/a.py": "x = 1\n",
                "src/b.txt": "y = 2\n",
                "tests/c.py": "z = 3\n",
            }
        ),
    )
    modules = _modules(["src/a.py", "src/b.txt", "tests/c.py"], repo_root=setup.repo_root)
    _seed_inventory(
        setup.gateway,
        setup.repo_root,
        repo="repo",
        commit="deadbeef",
        paths=[module.rel_path for module in modules],
    )
    tracker = ChangeTracker(
        gateway=setup.gateway,
        change_request=ChangeRequest(
            repo="repo",
            commit="deadbeef",
            repo_root=setup.repo_root,
            modules=modules,
        ),
        modules=modules,
        change_set=ChangeSet(added=[modules[0]], modified=[], deleted=[modules[2]]),
        policy=IncrementalIngestPolicy(min_total_modules_for_ratio=10),
    )

    try:
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
    finally:
        setup.gateway.close()


def test_view_for_dataset_full_rebuild_flag_forces_rebuild(
    tmp_path: Path,
) -> None:
    """Force full rebuild when change request flag is set."""
    setup = make_scan_setup(
        tmp_path,
        options=ScanSetupOptions(
            repo_structure={
                "a.py": "x = 1\n",
                "b.py": "y = 2\n",
            }
        ),
    )
    modules = _modules(["a.py", "b.py"], repo_root=setup.repo_root)
    _seed_inventory(
        setup.gateway,
        setup.repo_root,
        repo="repo",
        commit="deadbeef",
        paths=[module.rel_path for module in modules],
    )
    tracker = ChangeTracker(
        gateway=setup.gateway,
        change_request=ChangeRequest(
            repo="repo",
            commit="deadbeef",
            repo_root=setup.repo_root,
            modules=modules,
            full_rebuild=True,
        ),
        modules=modules,
        change_set=ChangeSet(added=[], modified=[modules[0]], deleted=[]),
        policy=IncrementalIngestPolicy(),
    )

    try:
        view = tracker.view_for_dataset(dataset_name="test", module_filter=None)

        if not view.use_full_rebuild:
            pytest.fail("Expected full rebuild via flag")
        if view.to_reparse != modules:
            pytest.fail("Full rebuild should reparse all modules")
        expected_deleted = [module.rel_path for module in modules]
        if view.deleted_paths != expected_deleted:
            pytest.fail("Full rebuild should delete all module paths")
    finally:
        setup.gateway.close()


def _docstrings_by_path(gateway: StorageGateway) -> dict[str, set[str]]:
    rows = gateway.con.table("core.docstrings").select("rel_path", "raw_docstring").fetchall()
    grouped: dict[str, set[str]] = {}
    for rel_path, raw_docstring in rows:
        grouped.setdefault(rel_path, set()).add(raw_docstring)
    return grouped


def test_incremental_ingest_ops_reparse_changed_modules(tmp_path: Path) -> None:
    """Ensure incremental typing ingest only processes modules flagged as changed.

    This test verifies that:
    1. Baseline typing metrics are established via initial full ingest
    2. When a file is modified, only that file's metrics change
    3. Unchanged files retain their original metrics
    """
    setup = make_scan_setup(
        tmp_path,
        options=ScanSetupOptions(
            repo_structure={
                "a.py": '"""Module A."""\n\ndef foo(x: int) -> int:\n    """Doc A."""\n    return x + 1',
                "b.py": '"""Module B."""\n\ndef bar(y):\n    """Doc B."""\n    return y',
            }
        ),
    )
    snapshot = make_snapshot(repo_root=setup.repo_root)
    doc_step = DocstringsExtractStep(storage=setup.storage, discovery=setup.discovery)

    try:
        _, modules, _ = setup.scan_step.execute(
            repo=snapshot.repo,
            commit=snapshot.commit,
            repo_root=setup.repo_root,
            profile=setup.profile,
        )

        doc_step.execute(list(modules), repo=snapshot.repo, commit=snapshot.commit)
        baseline_docstrings = _docstrings_by_path(setup.gateway)

        file_b = setup.repo_root / "b.py"
        file_b.write_text(
            '"""Module B updated."""\n\ndef bar(y: int) -> int:\n'
            '    """Doc B updated."""\n    return y + 2',
            encoding="utf8",
        )

        doc_step.execute(list(modules), repo=snapshot.repo, commit=snapshot.commit)
        updated_docstrings = _docstrings_by_path(setup.gateway)

        if updated_docstrings.get("a.py") != baseline_docstrings.get("a.py"):
            baseline_a = baseline_docstrings.get("a.py")
            updated_a = updated_docstrings.get("a.py")
            pytest.fail(
                "Unchanged module docstrings should remain stable. "
                f"Baseline: {baseline_a}, Updated: {updated_a}"
            )
        if updated_docstrings.get("b.py") == baseline_docstrings.get("b.py"):
            pytest.fail("Changed module docstrings should be updated")
        if "Module B updated." not in updated_docstrings.get("b.py", ""):
            pytest.fail("Updated docstring content was not ingested")
    finally:
        setup.gateway.close()


def test_compute_changes_tracks_add_modify_delete(tmp_path: Path) -> None:
    """Change detection should surface added, modified, and deleted modules.

    This test verifies realistic change detection behavior:
    - Added: file exists in current scan but not in previous state
    - Modified: file exists with different content/hash
    - Deleted: file was in previous state but not in current scan

    The key insight is that deletions are detected by ABSENCE from the current
    module list, not by passing a module record for a non-existent file.
    """
    setup = make_scan_setup(
        tmp_path,
        options=ScanSetupOptions(
            repo_structure={"a.py": "x = 1\n"},
        ),
    )
    repo_root = setup.repo_root
    file_path = repo_root / "a.py"
    gateway = setup.gateway

    def make_record() -> ModuleRecord:
        return _modules(["a.py"], repo_root=repo_root)[0]

    def make_request(modules: list[ModuleRecord]) -> ChangeRequest:
        return ChangeRequest(
            repo="r",
            commit="c",
            repo_root=repo_root,
            modules=modules,
        )

    try:
        # First pass: file is new → should report as added
        first = _compute_changes(gateway, make_request([make_record()]))
        if len(first.added) != 1 or first.modified or first.deleted:
            pytest.fail(f"Expected first pass to report one addition only, got {first}")

        # Second pass with same file → no changes
        unchanged = _compute_changes(gateway, make_request([make_record()]))
        if unchanged.added or unchanged.modified or unchanged.deleted:
            pytest.fail(f"Expected no changes on second pass, got {unchanged}")

        # Third pass: modify file content → should detect modification
        file_path.write_text("x = 2\n", encoding="utf8")
        modified = _compute_changes(gateway, make_request([make_record()]))
        if modified.added or len(modified.modified) != 1 or modified.deleted:
            pytest.fail(f"Expected single modification only, got {modified}")

        # Fourth pass: delete file and pass EMPTY module list
        # (simulating that the scanner no longer finds the file)
        file_path.unlink()
        deleted = _compute_changes(gateway, make_request([]))  # Empty list = file not found
        if deleted.added or deleted.modified or len(deleted.deleted) != 1:
            pytest.fail(f"Expected single deletion only, got {deleted}")
    finally:
        gateway.close()
