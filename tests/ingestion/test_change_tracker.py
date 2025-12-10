"""Unit tests for ChangeTracker dataset views."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.ingestion import DocstringsExtractStep, RepoScanStep
from codeintel.ingestion.infrastructure.scanning import ScanProfile
from codeintel.ingestion.ports.change_detection import ChangeRequest, ChangeSet
from codeintel.ingestion.ports.discovery import ModuleRecord
from codeintel.ingestion.tracker import (
    ChangeTracker,
    IncrementalIngestPolicy,
)
from codeintel.storage.gateway import StorageGateway
from tests._helpers.factories import make_snapshot
from tests._helpers.gateway import GatewayFactory
from tests._helpers.ingestion import (
    TargetContextConfig,
    build_ingestion_adapters,
    build_target_context_for_plugin,
)
from codeintel.ingestion.plugins.repo_scan import RepoScanPlugin


def _module(rel_path: str) -> ModuleRecord:
    return ModuleRecord(
        rel_path=rel_path,
        module_name=rel_path.replace("/", ".").removesuffix(".py"),
        file_path=Path(rel_path),
        index=0,
        total=0,
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


def _docstrings_by_path(gateway: StorageGateway) -> dict[str, set[str]]:
    rows = gateway.con.table("core.docstrings").select("rel_path", "raw_docstring").fetchall()
    grouped: dict[str, set[str]] = {}
    for rel_path, raw_docstring in rows:
        grouped.setdefault(rel_path, set()).add(raw_docstring)
    return grouped


def _setup_test_files(repo_root: Path) -> tuple[Path, Path]:
    """Create test files a.py and b.py with initial content.

    Returns
    -------
    tuple[Path, Path]
        Paths to file_a and file_b.
    """
    repo_root.mkdir()
    file_a = repo_root / "a.py"
    file_b = repo_root / "b.py"
    file_a.write_text(
        '"""Module A."""\n\ndef foo(x: int) -> int:\n    """Doc A."""\n    return x + 1',
        encoding="utf8",
    )
    file_b.write_text(
        '"""Module B."""\n\ndef bar(y):\n    """Doc B."""\n    return y',
        encoding="utf8",
    )
    return file_a, file_b


def _create_scan_infrastructure(
    gateway: StorageGateway,
    repo_root: Path,
    tmp_path: Path,
) -> tuple[RepoScanStep, DocstringsExtractStep, ScanProfile]:
    """Create steps and profile for repo scanning.

    Returns
    -------
    tuple[RepoScanStep, DocstringsExtractStep, ScanProfile]
        Scan step, docstrings step, and scan profile.
    """
    ctx = build_target_context_for_plugin(
        RepoScanPlugin(),
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, gateway=gateway),
    )
    storage, discovery, change_detection, _ = build_ingestion_adapters(ctx)
    scan_profile = ScanProfile(
        repo_root=repo_root,
        source_roots=(repo_root,),
        include_globs=("*.py",),
        ignore_dirs=(),
    )
    scan_step = RepoScanStep(
        storage=storage,
        discovery=discovery,
        change_detection=change_detection,
    )
    doc_step = DocstringsExtractStep(storage=storage, discovery=discovery)
    return scan_step, doc_step, scan_profile


def test_incremental_ingest_ops_reparse_changed_modules(tmp_path: Path) -> None:
    """Ensure incremental typing ingest only processes modules flagged as changed.

    This test verifies that:
    1. Baseline typing metrics are established via initial full ingest
    2. When a file is modified, only that file's metrics change
    3. Unchanged files retain their original metrics
    """
    repo_root = tmp_path / "repo"
    _, file_b = _setup_test_files(repo_root)
    snapshot = make_snapshot(repo_root=repo_root)

    gateway = GatewayFactory().open()
    try:
        scan_step, doc_step, scan_profile = _create_scan_infrastructure(gateway, repo_root, tmp_path)

        # Step 1: Populate core.modules via repo scan
        _, modules, _ = scan_step.execute(
            repo=snapshot.repo, commit=snapshot.commit, repo_root=repo_root, profile=scan_profile
        )

        # Step 2: Ingest baseline docstrings
        doc_step.execute(list(modules), repo=snapshot.repo, commit=snapshot.commit)
        baseline_docstrings = _docstrings_by_path(gateway)

        # Step 3: Modify file_b to add type annotations
        file_b.write_text(
            '"""Module B updated."""\n\ndef bar(y: int) -> int:\n'
            '    """Doc B updated."""\n    return y + 2',
            encoding="utf8",
        )

        # Step 4: Re-ingest to pick up changes
        doc_step.execute(list(modules), repo=snapshot.repo, commit=snapshot.commit)
        updated_docstrings = _docstrings_by_path(gateway)

        # Verify: b.py should be updated, a.py unchanged
        if updated_docstrings.get("a.py") != baseline_docstrings.get("a.py"):
            pytest.fail(
                f"Unchanged module docstrings should remain stable. "
                f"Baseline: {baseline_docstrings.get('a.py')}, Updated: {updated_docstrings.get('a.py')}"
            )
        if updated_docstrings.get("b.py") == baseline_docstrings.get("b.py"):
            pytest.fail("Changed module docstrings should be updated")
        if "Module B updated." not in updated_docstrings.get("b.py", ""):
            pytest.fail("Updated docstring content was not ingested")
    finally:
        gateway.close()


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

    gateway = GatewayFactory().open()

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
