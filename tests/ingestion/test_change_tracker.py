"""Unit tests for ChangeTracker dataset views."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.config import ConfigBuilder
from codeintel.ingestion.change_tracker import (
    ChangeTracker,
    IncrementalIngestPolicy,
    run_incremental_ingest,
)
from codeintel.ingestion.common import ChangeRequest, ChangeSet, ModuleRecord
from codeintel.ingestion.docstrings_ingest import DocstringIngestOps
from codeintel.ingestion.typing_ingest import ingest_typing_signals
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import open_ingestion_gateway

if TYPE_CHECKING:
    from codeintel.ingestion.tool_service import ToolService


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


class _FakeToolService:
    """Minimal async tool service returning empty diagnostics."""

    @staticmethod
    async def run_pyrefly(repo_root: Path) -> dict[str, int]:
        del repo_root
        return {}

    @staticmethod
    async def run_pyright(repo_root: Path) -> dict[str, int]:
        del repo_root
        return {}

    @staticmethod
    async def run_ruff(repo_root: Path) -> dict[str, int]:
        del repo_root
        return {}


def _docstrings_by_path(gateway: StorageGateway) -> dict[str, set[str]]:
    rows = gateway.con.execute(
        "SELECT rel_path, raw_docstring FROM core.docstrings",
    ).fetchall()
    grouped: dict[str, set[str]] = {}
    for rel_path, raw_docstring in rows:
        grouped.setdefault(rel_path, set()).add(raw_docstring)
    return grouped


def _typedness_metrics_by_path(
    gateway: StorageGateway,
) -> dict[str, set[tuple[float, float, int]]]:
    rows = gateway.con.execute(
        "SELECT path, annotation_ratio, untyped_defs FROM analytics.typedness",
    ).fetchall()
    grouped: dict[str, set[tuple[float, float, int]]] = {}
    for path, ratio, untyped_defs in rows:
        if isinstance(ratio, str):
            try:
                ratio = json.loads(ratio)
            except json.JSONDecodeError:
                ratio = {"params": 0.0, "returns": 0.0}
        grouped.setdefault(path, set()).add(
            (ratio.get("params", 0.0), ratio.get("returns", 0.0), untyped_defs)
        )
    return grouped


def test_incremental_ingest_ops_reparse_changed_modules(tmp_path: Path) -> None:
    """Ensure incremental ops only append rows for modules flagged as changed."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    file_a = repo_root / "a.py"
    file_b = repo_root / "b.py"
    file_a.write_text(
        "\n".join(
            [
                '"""Module A."""',
                "",
                "def foo(x: int) -> int:",
                '    """Doc A."""',
                "    return x + 1",
            ]
        ),
        encoding="utf8",
    )
    file_b.write_text(
        "\n".join(
            [
                '"""Module B."""',
                "",
                "def bar(y):",
                '    """Doc B."""',
                "    return y",
            ]
        ),
        encoding="utf8",
    )

    modules = [
        ModuleRecord(rel_path="a.py", module_name="a", file_path=file_a, index=0, total=2),
        ModuleRecord(rel_path="b.py", module_name="b", file_path=file_b, index=1, total=2),
    ]
    gateway = open_ingestion_gateway()
    try:
        builder = ConfigBuilder.from_snapshot(
            repo="demo/repo",
            commit="deadbeef",
            repo_root=repo_root,
            build_dir=repo_root / "build",
        )
        doc_cfg = builder.docstring()
        typing_cfg = builder.typing_ingest(tool_runner=None)
        policy = IncrementalIngestPolicy(min_total_modules_for_ratio=1)

        tracker = ChangeTracker(
            gateway=gateway,
            change_request=ChangeRequest(
                repo=doc_cfg.repo,
                commit=doc_cfg.commit,
                repo_root=repo_root,
                modules=modules,
            ),
            modules=modules,
            change_set=ChangeSet(added=modules, modified=[], deleted=[]),
            policy=policy,
        )

        fake_service = cast("ToolService", _FakeToolService())

        run_incremental_ingest(
            tracker,
            DocstringIngestOps(cfg=doc_cfg, created_at=datetime(2024, 1, 1, tzinfo=UTC)),
        )
        ingest_typing_signals(
            gateway,
            cfg=typing_cfg,
            tool_service=fake_service,
            tracker=tracker,
        )
        baseline_docstrings = _docstrings_by_path(gateway)
        baseline_metrics = _typedness_metrics_by_path(gateway)

        file_b.write_text(
            "\n".join(
                [
                    '"""Module B updated."""',
                    "",
                    "def bar(y: int) -> int:",
                    '    """Doc B updated."""',
                    "    return y + 2",
                ]
            ),
            encoding="utf8",
        )

        tracker = ChangeTracker(
            gateway=gateway,
            change_request=ChangeRequest(
                repo=doc_cfg.repo,
                commit=doc_cfg.commit,
                repo_root=repo_root,
                modules=modules,
            ),
            modules=modules,
            change_set=ChangeSet(added=[], modified=[modules[1]], deleted=[]),
            policy=policy,
        )

        run_incremental_ingest(
            tracker,
            DocstringIngestOps(cfg=doc_cfg, created_at=datetime(2024, 1, 2, tzinfo=UTC)),
        )
        ingest_typing_signals(
            gateway,
            cfg=typing_cfg,
            tool_service=fake_service,
            tracker=tracker,
        )

        updated_docstrings = _docstrings_by_path(gateway)
        updated_metrics = _typedness_metrics_by_path(gateway)

        if updated_docstrings["a.py"] != baseline_docstrings["a.py"]:
            pytest.fail("Unchanged module docstrings should remain stable")
        if updated_docstrings["b.py"] == baseline_docstrings["b.py"]:
            pytest.fail("Changed module docstrings should be updated")
        if "Module B updated." not in updated_docstrings["b.py"]:
            pytest.fail("Updated docstring content was not ingested")

        if updated_metrics["a.py"] != baseline_metrics["a.py"]:
            pytest.fail("Unchanged module metrics should remain stable")
        if updated_metrics["b.py"] == baseline_metrics["b.py"]:
            pytest.fail("Updated metrics for changed module should differ")
    finally:
        gateway.close()
