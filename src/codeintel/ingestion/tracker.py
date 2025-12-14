"""Domain service for change tracking.

This module provides unified change tracking for ingestion,
analogous to graphs/catalog.py for function catalog services.

Key Components
--------------
- ChangeTracker: Single source of truth for change detection
- ChangeTrackerDatasetView: Per-dataset view of changes
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

from codeintel.ingestion.adapters.duckdb_storage import DuckDBStorageAdapter
from codeintel.ingestion.adapters.hash_change_detection import HashChangeDetectionAdapter
from codeintel.ingestion.ports.discovery import ModuleRecord

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.ingestion.ports.change_detection import (
        ChangeDetectionPort,
        ChangeRequest,
        ChangeSet,
    )
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

ModuleFilter = Callable[[ModuleRecord], bool]


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


__all__ = [
    "ChangeTracker",
    "ChangeTrackerDatasetView",
]
