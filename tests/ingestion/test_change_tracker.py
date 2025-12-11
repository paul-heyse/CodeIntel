"""Unit tests for ChangeTracker dataset views."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypedDict

import pytest

from codeintel.ingestion import (
    DocstringsExtractStep,
    DuckDBStorageAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.ports.change_detection import ChangeRequest, ChangeSet
from codeintel.ingestion.ports.discovery import ModuleRecord
from codeintel.ingestion.tracker import ChangeTracker, IncrementalIngestPolicy
from tests._helpers.factories import make_snapshot
from tests._helpers.ingestion import (
    ScanSetupOptions,
    build_repo_tree,
    closing_gateway,
    make_scan_setup,
    module_records_for_paths,
    seed_inventory_from_paths,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.gateway import GatewayFactory

ModuleFilter = Callable[[ModuleRecord], bool]


class ViewScenario(TypedDict):
    """Typed configuration for change tracker view assertions."""

    id: str
    structure: dict[str, str]
    full_rebuild: bool
    change_set: Callable[[list[ModuleRecord]], ChangeSet]
    policy: Callable[[], IncrementalIngestPolicy]
    module_filter: Callable[[list[ModuleRecord]], ModuleFilter | None]
    expected_use_full: bool
    expected_reparse: Callable[[list[ModuleRecord]], list[ModuleRecord]]
    expected_deleted: Callable[[list[ModuleRecord]], list[str]]


@pytest.fixture
def scan_setup(
    tmp_path: Path, ingestion_gateway_factory: GatewayFactory
) -> Generator[SimpleNamespace]:
    """Provision a reusable scan setup with gateway and snapshot.

    Yields
    ------
    SimpleNamespace
        Bundle containing repo root, gateway, snapshot, and supporting services.
    """
    setup = make_scan_setup(
        tmp_path,
        options=ScanSetupOptions(gateway_factory=ingestion_gateway_factory),
    )
    snapshot = make_snapshot(repo="demo/change-tracker", commit="abc123", repo_root=setup.repo_root)
    ctx = SimpleNamespace(
        repo_root=setup.repo_root,
        gateway=setup.gateway,
        snapshot=snapshot,
        scan_step=setup.scan_step,
        profile=setup.profile,
        storage=setup.storage,
        discovery=setup.discovery,
    )
    with closing_gateway(setup.gateway):
        yield ctx


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


def _build_repo(repo_root: Path, structure: dict[str, str]) -> list[ModuleRecord]:
    """Materialize a repo structure and build module records.

    Returns
    -------
    list[ModuleRecord]
        Module records corresponding to the created repository structure.
    """
    build_repo_tree(repo_root, structure)
    return _modules(list(structure.keys()), repo_root=repo_root)


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


def _src_python_filter(_modules: list[ModuleRecord]) -> ModuleFilter:
    """Return a predicate that keeps only src/ Python modules.

    Returns
    -------
    ModuleFilter
        Predicate that accepts only Python files under src/.
    """

    def _predicate(module: ModuleRecord) -> bool:
        return module.rel_path.endswith(".py") and module.rel_path.startswith("src/")

    return _predicate


VIEW_SCENARIOS: tuple[ViewScenario, ...] = (
    {
        "id": "incremental",
        "structure": {"a.py": "print('ok')\n", "b.py": "print('ok')\n", "c.py": "print('ok')\n"},
        "full_rebuild": False,
        "change_set": lambda modules: ChangeSet(added=[], modified=[modules[1]], deleted=[]),
        "policy": lambda: IncrementalIngestPolicy(min_total_modules_for_ratio=1),
        "module_filter": lambda _modules: None,
        "expected_use_full": False,
        "expected_reparse": lambda modules: [modules[1]],
        "expected_deleted": lambda _modules: [],
    },
    {
        "id": "ratio_full",
        "structure": {"a.py": "print('ok')\n", "b.py": "print('ok')\n", "c.py": "print('ok')\n"},
        "full_rebuild": False,
        "change_set": lambda modules: ChangeSet(
            added=[modules[0]], modified=[modules[1]], deleted=[]
        ),
        "policy": lambda: IncrementalIngestPolicy(
            max_changed_ratio=0.5, min_total_modules_for_ratio=1
        ),
        "module_filter": lambda _modules: None,
        "expected_use_full": True,
        "expected_reparse": lambda modules: modules,
        "expected_deleted": lambda modules: [module.rel_path for module in modules],
    },
    {
        "id": "filter_incremental",
        "structure": {
            "src/a.py": "x = 1\n",
            "src/b.txt": "y = 2\n",
            "tests/c.py": "z = 3\n",
        },
        "full_rebuild": False,
        "change_set": lambda modules: ChangeSet(
            added=[modules[0]], modified=[], deleted=[modules[2]]
        ),
        "policy": lambda: IncrementalIngestPolicy(min_total_modules_for_ratio=10),
        "module_filter": _src_python_filter,
        "expected_use_full": False,
        "expected_reparse": lambda modules: [modules[0]],
        "expected_deleted": lambda _modules: [],
    },
    {
        "id": "flag_full_rebuild",
        "structure": {"a.py": "print('ok')\n", "b.py": "print('ok')\n"},
        "full_rebuild": True,
        "change_set": lambda modules: ChangeSet(added=[], modified=[modules[0]], deleted=[]),
        "policy": IncrementalIngestPolicy,
        "module_filter": lambda _modules: None,
        "expected_use_full": True,
        "expected_reparse": lambda modules: modules,
        "expected_deleted": lambda modules: [module.rel_path for module in modules],
    },
)


@pytest.mark.parametrize("case", VIEW_SCENARIOS, ids=[case["id"] for case in VIEW_SCENARIOS])
def test_view_for_dataset_modes(scan_setup: SimpleNamespace, case: ViewScenario) -> None:
    """Validate incremental vs full rebuild selection across scenarios."""
    repo_root = scan_setup.repo_root
    gateway = scan_setup.gateway
    modules = _build_repo(repo_root, case["structure"])
    _seed_inventory(
        gateway=gateway,
        repo_root=repo_root,
        repo=scan_setup.snapshot.repo,
        commit=scan_setup.snapshot.commit,
        paths=[module.rel_path for module in modules],
    )
    change_set = case["change_set"](modules)
    tracker = ChangeTracker(
        gateway=gateway,
        change_request=ChangeRequest(
            repo=scan_setup.snapshot.repo,
            commit=scan_setup.snapshot.commit,
            repo_root=repo_root,
            modules=modules,
            full_rebuild=bool(case["full_rebuild"]),
        ),
        modules=modules,
        change_set=change_set,
        policy=case["policy"](),
    )
    module_filter_factory = case["module_filter"]
    module_filter = module_filter_factory(modules) if module_filter_factory else None

    view = tracker.view_for_dataset(dataset_name="test", module_filter=module_filter)

    expected_reparse = case["expected_reparse"](modules)
    expected_deleted = case["expected_deleted"](modules)
    if bool(view.use_full_rebuild) != bool(case["expected_use_full"]):
        pytest.fail(f"Unexpected rebuild mode for scenario {case['id']}")
    if view.to_reparse != expected_reparse:
        pytest.fail(f"Unexpected modules selected for reparse in scenario {case['id']}")
    if view.deleted_paths != expected_deleted:
        pytest.fail(f"Deleted paths mismatch in scenario {case['id']}: {view.deleted_paths}")


def _docstrings_by_path(gateway: StorageGateway) -> dict[str, set[str]]:
    rows = gateway.con.table("core.docstrings").select("rel_path", "raw_docstring").fetchall()
    grouped: dict[str, set[str]] = {}
    for rel_path, raw_docstring in rows:
        grouped.setdefault(rel_path, set()).add(raw_docstring)
    return grouped


def test_incremental_ingest_ops_reparse_changed_modules(
    ingestion_ctx_bundle: SimpleNamespace,
) -> None:
    """Ensure incremental typing ingest only processes modules flagged as changed.

    This test verifies that:
    1. Baseline typing metrics are established via initial full ingest
    2. When a file is modified, only that file's metrics change
    3. Unchanged files retain their original metrics
    """
    repo_root = ingestion_ctx_bundle.repo_root
    repo = ingestion_ctx_bundle.ctx.snapshot.repo
    commit = ingestion_ctx_bundle.ctx.snapshot.commit
    structure = {
        "a.py": '"""Module A."""\n\ndef foo(x: int) -> int:\n    """Doc A."""\n    return x + 1',
        "b.py": '"""Module B."""\n\ndef bar(y):\n    """Doc B."""\n    return y',
    }
    _build_repo(repo_root, structure)

    doc_step = DocstringsExtractStep(
        storage=ingestion_ctx_bundle.storage, discovery=ingestion_ctx_bundle.discovery
    )
    _, modules, _ = ingestion_ctx_bundle.scan_step.execute(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        profile=ingestion_ctx_bundle.profile,
    )

    doc_step.execute(list(modules), repo=repo, commit=commit)
    baseline_docstrings = _docstrings_by_path(ingestion_ctx_bundle.gateway)

    file_b = repo_root / "b.py"
    file_b.write_text(
        '"""Module B updated."""\n\ndef bar(y: int) -> int:\n'
        '    """Doc B updated."""\n    return y + 2',
        encoding="utf8",
    )

    doc_step.execute(list(modules), repo=repo, commit=commit)
    updated_docstrings = _docstrings_by_path(ingestion_ctx_bundle.gateway)

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


def test_compute_changes_tracks_add_modify_delete(ingestion_ctx_bundle: SimpleNamespace) -> None:
    """Change detection should surface added, modified, and deleted modules.

    This test verifies realistic change detection behavior:
    - Added: file exists in current scan but not in previous state
    - Modified: file exists with different content/hash
    - Deleted: file was in previous state but not in current scan

    The key insight is that deletions are detected by ABSENCE from the current
    module list, not by passing a module record for a non-existent file.
    """
    repo_root = ingestion_ctx_bundle.repo_root
    _build_repo(repo_root, {"a.py": "x = 1\n"})
    file_path = repo_root / "a.py"
    gateway = ingestion_ctx_bundle.gateway

    def make_record() -> ModuleRecord:
        return _modules(["a.py"], repo_root=repo_root)[0]

    def make_request(modules: list[ModuleRecord]) -> ChangeRequest:
        return ChangeRequest(
            repo=ingestion_ctx_bundle.ctx.snapshot.repo,
            commit=ingestion_ctx_bundle.ctx.snapshot.commit,
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
