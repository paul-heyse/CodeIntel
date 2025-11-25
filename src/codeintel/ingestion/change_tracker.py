"""Shared change tracking and incremental ingest harness."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from typing import NamedTuple, Protocol, TypeVar, runtime_checkable

from codeintel.ingestion.common import ChangeRequest, ChangeSet, ModuleRecord, compute_changes
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

ModuleFilter = Callable[[ModuleRecord], bool]
RowT = TypeVar("RowT")
ExecutorFactory = Callable[[], Executor]


@dataclass(frozen=True)
class IncrementalIngestPolicy:
    """Tuning knobs for incremental ingestion."""

    max_changed_ratio: float = 0.7
    max_deleted_ratio: float = 0.7
    min_total_modules_for_ratio: int = 20
    log_every: int = 100
    flush_every: int = 500


class ChangeTrackerDatasetView(NamedTuple):
    """Per-dataset view of modules to reparse and rows to delete."""

    to_reparse: list[ModuleRecord]
    deleted_paths: list[str]
    total_modules_considered: int
    changed_modules_count: int
    deleted_modules_count: int
    use_full_rebuild: bool


@dataclass
class ChangeTracker:
    """Single source of truth for change detection across ingest steps."""

    gateway: StorageGateway
    change_request: ChangeRequest
    modules: Sequence[ModuleRecord]
    change_set: ChangeSet
    policy: IncrementalIngestPolicy

    @classmethod
    def create(
        cls,
        gateway: StorageGateway,
        change_request: ChangeRequest,
        modules: Sequence[ModuleRecord],
        policy: IncrementalIngestPolicy | None = None,
    ) -> ChangeTracker:
        effective_policy = policy or IncrementalIngestPolicy()
        request_modules = change_request.modules or tuple(modules)
        request = replace(change_request, modules=request_modules)
        change_set = compute_changes(gateway, request)
        return cls(
            gateway=gateway,
            change_request=request,
            modules=tuple(modules),
            change_set=change_set,
            policy=effective_policy,
        )

    def view_for_dataset(
        self,
        *,
        dataset_name: str,
        module_filter: ModuleFilter | None = None,
    ) -> ChangeTrackerDatasetView:
        """Compute dataset-scoped changes with full rebuild policy applied."""
        relevant_modules = (
            [module for module in self.modules if module_filter(module)]
            if module_filter is not None
            else list(self.modules)
        )
        rel_paths = {module.rel_path for module in relevant_modules}

        added = [module for module in self.change_set.added if module.rel_path in rel_paths]
        modified = [module for module in self.change_set.modified if module.rel_path in rel_paths]
        deleted = [module for module in self.change_set.deleted if module.rel_path in rel_paths]

        to_reparse = added + modified
        deleted_paths = [module.rel_path for module in deleted]

        total = len(relevant_modules)
        changed_count = len(to_reparse)
        deleted_count = len(deleted)

        use_full = False
        if total >= self.policy.min_total_modules_for_ratio and total > 0:
            changed_ratio = (changed_count + deleted_count) / total
            deleted_ratio = deleted_count / total
            if (
                changed_ratio >= self.policy.max_changed_ratio
                or deleted_ratio >= self.policy.max_deleted_ratio
            ):
                use_full = True

        if use_full:
            to_reparse = list(relevant_modules)
            deleted_paths = [module.rel_path for module in relevant_modules]

        log.info(
            "Dataset view computed for %s (total=%d changed=%d deleted=%d full=%s)",
            dataset_name,
            total,
            changed_count,
            deleted_count,
            use_full,
        )

        return ChangeTrackerDatasetView(
            to_reparse=to_reparse,
            deleted_paths=deleted_paths,
            total_modules_considered=total,
            changed_modules_count=changed_count,
            deleted_modules_count=deleted_count,
            use_full_rebuild=use_full,
        )


@runtime_checkable
class IncrementalIngestOps(Protocol[RowT]):
    """Operations required to incrementally ingest a dataset."""

    dataset_name: str

    def module_filter(self, module: ModuleRecord) -> bool:
        """Return True when a module should be considered for this dataset."""

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """Remove rows corresponding to the provided relative paths."""

    def process_module(self, module: ModuleRecord) -> Iterable[RowT]:
        """Generate rows for a single module."""

    def insert_rows(self, gateway: StorageGateway, rows: Sequence[RowT]) -> None:
        """Persist generated rows to the target dataset."""


@runtime_checkable
class SupportsFullRebuild(Protocol[RowT]):
    """Optional hook for datasets that need a specialized full rebuild path."""

    def run_full_rebuild(self, tracker: ChangeTracker) -> bool:
        """Return True when the full rebuild was handled and no further work is needed."""


def run_incremental_ingest(
    tracker: ChangeTracker,
    ops: IncrementalIngestOps[RowT],
    *,
    executor_factory: ExecutorFactory | None = None,
) -> None:
    """Shared driver for per-module ingestion using a precomputed change tracker."""
    view = tracker.view_for_dataset(dataset_name=ops.dataset_name, module_filter=ops.module_filter)

    if view.use_full_rebuild and isinstance(ops, SupportsFullRebuild):
        handled = ops.run_full_rebuild(tracker)
        if handled:
            return

    if not view.to_reparse and not view.deleted_paths:
        log.info(
            "No changes for dataset %s (total=%d)",
            ops.dataset_name,
            view.total_modules_considered,
        )
        return

    if view.use_full_rebuild:
        log.info(
            "Dataset %s: full rebuild (changed=%d deleted=%d total=%d)",
            ops.dataset_name,
            view.changed_modules_count,
            view.deleted_modules_count,
            view.total_modules_considered,
        )
    else:
        log.info(
            "Dataset %s: incremental ingest (reparse=%d delete=%d total=%d)",
            ops.dataset_name,
            len(view.to_reparse),
            len(view.deleted_paths),
            view.total_modules_considered,
        )

    if view.deleted_paths:
        ops.delete_rows(tracker.gateway, view.deleted_paths)

    rows: list[RowT] = []
    if executor_factory is None:
        for module in view.to_reparse:
            rows.extend(ops.process_module(module))
    else:
        with executor_factory() as executor:
            for result in executor.map(ops.process_module, view.to_reparse):
                rows.extend(result)

    if not rows:
        log.info("Dataset %s: no rows to insert after processing", ops.dataset_name)
        return

    ops.insert_rows(tracker.gateway, rows)
