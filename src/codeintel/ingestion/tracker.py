"""Domain service for change tracking and incremental ingestion.

This module provides unified change tracking for incremental ingestion,
analogous to graphs/catalog.py for function catalog services.

Key Components
--------------
- ChangeTracker: Single source of truth for change detection
- ChangeTrackerDatasetView: Per-dataset view of changes
- IncrementalIngestOps: Protocol for dataset-specific operations
- run_incremental_ingest: Main entry point for incremental ingestion
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, NamedTuple, Protocol, TypeVar, runtime_checkable

from codeintel.ingestion.adapters.duckdb_storage import DuckDBStorageAdapter
from codeintel.ingestion.adapters.hash_change_detection import HashChangeDetectionAdapter
from codeintel.ingestion.ports.change_detection import ChangeRequest, ChangeSet
from codeintel.ingestion.ports.discovery import ModuleRecord

if TYPE_CHECKING:
    from codeintel.ingestion.ports.change_detection import ChangeDetectionPort
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

ModuleFilter = Callable[[ModuleRecord], bool]
RowT = TypeVar("RowT")
ExecutorFactory = Callable[[], Executor]
IncrementalIngestObserver = Callable[
    [str, "ChangeTrackerDatasetView"],
    None,
]


@dataclass(frozen=True)
class IncrementalIngestPolicy:
    """Tuning knobs for incremental ingestion.

    Attributes
    ----------
    max_changed_ratio
        Maximum ratio of changed modules before triggering full rebuild.
    max_deleted_ratio
        Maximum ratio of deleted modules before triggering full rebuild.
    min_total_modules_for_ratio
        Minimum module count before applying ratio thresholds.
    log_every
        Progress logging interval.
    flush_every
        Batch flush interval.
    """

    max_changed_ratio: float = 0.7
    max_deleted_ratio: float = 0.7
    min_total_modules_for_ratio: int = 20
    log_every: int = 100
    flush_every: int = 500


class ChangeTrackerDatasetView(NamedTuple):
    """Per-dataset view of modules to reparse and rows to delete.

    Attributes
    ----------
    to_reparse
        Modules that need processing.
    deleted_paths
        Paths of deleted modules.
    total_modules_considered
        Total modules in scope.
    changed_modules_count
        Number of changed modules.
    deleted_modules_count
        Number of deleted modules.
    use_full_rebuild
        Whether full rebuild mode is active.
    """

    to_reparse: list[ModuleRecord]
    deleted_paths: list[str]
    total_modules_considered: int
    changed_modules_count: int
    deleted_modules_count: int
    use_full_rebuild: bool


@dataclass
class ChangeTracker:
    """Single source of truth for change detection across ingest steps.

    This class uses the port-adapter architecture for change detection,
    supporting both explicit port injection and automatic adapter creation.

    Attributes
    ----------
    gateway
        Storage gateway for database access.
    change_request
        The request parameters used for change detection.
    modules
        All modules considered for this snapshot.
    change_set
        Computed changes (added, modified, deleted).
    policy
        Policy tuning for incremental behavior.
    """

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
        *,
        change_detection: ChangeDetectionPort | None = None,
    ) -> ChangeTracker:
        """Build a change tracker with a computed change set.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        change_request
            Change detection request parameters.
        modules
            Modules to track changes for.
        policy
            Optional policy for incremental behavior.
        change_detection
            Optional change detection port (creates one if not provided).

        Returns
        -------
        ChangeTracker
            Tracker containing the computed change set and policy.
        """
        effective_policy = policy or IncrementalIngestPolicy()
        request_modules = change_request.modules or tuple(modules)
        request = replace(change_request, modules=request_modules)

        # Use provided port or create default adapter
        if change_detection is None:
            storage = DuckDBStorageAdapter(gateway)
            change_detection = HashChangeDetectionAdapter(storage)

        change_set = change_detection.compute_changes(request, list(modules))

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
        """Compute dataset-scoped changes with full rebuild policy applied.

        Parameters
        ----------
        dataset_name
            Name of the dataset for logging.
        module_filter
            Optional filter for relevant modules.

        Returns
        -------
        ChangeTrackerDatasetView
            Resolved reparse list, delete list, and rebuild mode for the dataset.
        """
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

        use_full = self.change_request.full_rebuild
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

        reason = "flag" if self.change_request.full_rebuild else "policy"
        log.info(
            "Dataset view computed for %s (total=%d changed=%d deleted=%d full=%s reason=%s)",
            dataset_name,
            total,
            changed_count,
            deleted_count,
            use_full,
            reason if use_full else "incremental",
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
    """Operations required to incrementally ingest a dataset.

    Attributes
    ----------
    dataset_name
        Name of the dataset for logging.
    """

    dataset_name: ClassVar[str]

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool:
        """Return True when a module should be considered for this dataset.

        Parameters
        ----------
        module
            Module to evaluate.

        Returns
        -------
        bool
            True if module is relevant.
        """
        ...

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """Remove rows corresponding to the provided relative paths.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        rel_paths
            Paths to delete.
        """
        ...

    @staticmethod
    def process_module(module: ModuleRecord) -> Iterable[RowT]:
        """Generate rows for a single module.

        Parameters
        ----------
        module
            Module to process.

        Returns
        -------
        Iterable[RowT]
            Generated rows.
        """
        ...

    def insert_rows(self, gateway: StorageGateway, rows: Sequence[RowT]) -> None:
        """Persist generated rows to the target dataset.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        rows
            Rows to insert.
        """
        ...


@runtime_checkable
class SupportsFullRebuild(Protocol):
    """Optional hook for datasets that need a specialized full rebuild path."""

    def run_full_rebuild(self, tracker: ChangeTracker) -> bool:
        """Execute a full rebuild of the dataset.

        Parameters
        ----------
        tracker
            Change tracker with context.

        Returns
        -------
        bool
            True when the full rebuild was handled and no further work is needed.
        """
        ...


def _notify_observer[RowT](
    observer: IncrementalIngestObserver | None,
    ops: IncrementalIngestOps[RowT],
    view: ChangeTrackerDatasetView,
) -> None:
    """Invoke observer callback when provided, swallowing exceptions."""
    if observer is None:
        return
    try:
        observer(ops.dataset_name, view)
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
    ):  # pragma: no cover - observer errors are non-fatal
        log.exception("Incremental ingest observer failed for dataset %s", ops.dataset_name)


def _handle_full_rebuild[RowT](
    view: ChangeTrackerDatasetView,
    tracker: ChangeTracker,
    ops: IncrementalIngestOps[RowT],
) -> bool:
    """Handle optional full rebuild hook and indicate whether processing should stop.

    Returns
    -------
    bool
        True when a full rebuild was handled and no further work is needed.
    """
    if view.use_full_rebuild and isinstance(ops, SupportsFullRebuild):
        handled = ops.run_full_rebuild(tracker)
        if handled:
            return True
    return False


def _log_view_mode(view: ChangeTrackerDatasetView, dataset_name: str) -> None:
    """Log whether a dataset is processed incrementally or via full rebuild."""
    if view.use_full_rebuild:
        log.info(
            "Dataset %s: full rebuild (changed=%d deleted=%d total=%d)",
            dataset_name,
            view.changed_modules_count,
            view.deleted_modules_count,
            view.total_modules_considered,
        )
        return
    log.info(
        "Dataset %s: incremental ingest (reparse=%d delete=%d total=%d)",
        dataset_name,
        len(view.to_reparse),
        len(view.deleted_paths),
        view.total_modules_considered,
    )


def _process_rows[RowT](
    view: ChangeTrackerDatasetView,
    ops: IncrementalIngestOps[RowT],
    executor_factory: ExecutorFactory | None,
) -> list[RowT]:
    """Generate rows for modules marked for reparse.

    Returns
    -------
    list[RowT]
        Processed rows ready for insertion.
    """
    rows: list[RowT] = []
    if executor_factory is None:
        for module in view.to_reparse:
            rows.extend(ops.process_module(module))
        return rows
    with executor_factory() as executor:
        for result in executor.map(ops.process_module, view.to_reparse):
            rows.extend(result)
    return rows


def run_incremental_ingest[RowT](
    tracker: ChangeTracker,
    ops: IncrementalIngestOps[RowT],
    *,
    executor_factory: ExecutorFactory | None = None,
    observer: IncrementalIngestObserver | None = None,
) -> None:
    """Execute incremental ingestion using a precomputed change tracker.

    Parameters
    ----------
    tracker
        ChangeTracker containing the precomputed ChangeSet and modules.
    ops
        Dataset-specific operations for delete/process/insert.
    executor_factory
        Optional factory yielding an Executor for parallel processing.
    observer
        Optional callback invoked with (dataset_name, view) before any rows are
        deleted or inserted. This is ideal for recording metrics.
    """
    view = tracker.view_for_dataset(dataset_name=ops.dataset_name, module_filter=ops.module_filter)

    _notify_observer(observer, ops, view)

    if _handle_full_rebuild(view, tracker, ops):
        return

    if not view.to_reparse and not view.deleted_paths:
        log.info(
            "No changes for dataset %s (total=%d)",
            ops.dataset_name,
            view.total_modules_considered,
        )
        return

    _log_view_mode(view, ops.dataset_name)

    if view.deleted_paths:
        ops.delete_rows(tracker.gateway, view.deleted_paths)

    rows = _process_rows(view, ops, executor_factory)

    if not rows:
        log.info("Dataset %s: no rows to insert after processing", ops.dataset_name)
        return

    ops.insert_rows(tracker.gateway, rows)


__all__ = [
    "ChangeTracker",
    "ChangeTrackerDatasetView",
    "IncrementalIngestOps",
    "IncrementalIngestPolicy",
    "SupportsFullRebuild",
    "run_incremental_ingest",
]
