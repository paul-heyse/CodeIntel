"""Unified pipeline abstraction for ingestion.

This module provides the IngestPipeline protocol and PipelineExecutor that
unify incremental and full ingestion modes. Ingest modules implement the
pipeline protocol; the executor handles tracker detection, worker pools,
batching, and progress logging.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.ingestion.common import ModuleRecord
from codeintel.ingestion.workers import (
    WorkerConfig,
    executor_factory,
    resolve_worker_count,
)

if TYPE_CHECKING:
    from codeintel.ingestion.change_tracker import (
        ChangeTracker,
        ChangeTrackerDatasetView,
    )
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

# Default batch sizes and logging intervals
DEFAULT_FLUSH_EVERY = 500
DEFAULT_LOG_EVERY = 100


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for pipeline execution.

    Attributes
    ----------
    flush_every
        Batch size for row persistence.
    log_every
        Module count interval for progress logging.
    worker_config
        Optional worker pool configuration.
    max_workers
        Explicit worker count override.
    executor_kind
        Executor type override ("thread" or "process").
    """

    flush_every: int = DEFAULT_FLUSH_EVERY
    log_every: int = DEFAULT_LOG_EVERY
    worker_config: WorkerConfig | None = None
    max_workers: int | None = None
    executor_kind: str | None = None


@dataclass
class PipelineResult:
    """Result of pipeline execution.

    Attributes
    ----------
    dataset_name
        Name of the dataset processed.
    modules_processed
        Number of modules processed.
    rows_persisted
        Total rows persisted to database.
    duration_s
        Total execution duration in seconds.
    errors
        List of error messages encountered.
    used_full_rebuild
        Whether full rebuild mode was used.
    """

    dataset_name: str
    modules_processed: int = 0
    rows_persisted: int = 0
    duration_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    used_full_rebuild: bool = False


@runtime_checkable
class IngestPipeline[RowT](Protocol):
    """Protocol for unified ingestion pipelines.

    Implementations provide dataset-specific logic for module filtering,
    processing, and persistence. The PipelineExecutor handles the common
    orchestration logic including tracker detection and worker pools.

    Attributes
    ----------
    dataset_name
        Unique identifier for the dataset (e.g., "core.ast_nodes").
    """

    @property
    def dataset_name(self) -> str:
        """Return the dataset name for this pipeline."""
        ...

    def module_filter(self, module: ModuleRecord) -> bool:
        """
        Determine whether a module should be processed.

        Parameters
        ----------
        module
            Module metadata to evaluate.

        Returns
        -------
        bool
            True if the module should be processed.
        """
        ...

    def process_module(self, module: ModuleRecord) -> Iterable[RowT]:
        """
        Process a single module and yield rows.

        Parameters
        ----------
        module
            Module to process.

        Returns
        -------
        Iterable[RowT]
            Rows generated from processing.
        """
        ...

    def persist_rows(self, gateway: StorageGateway, rows: Sequence[RowT]) -> int:
        """
        Persist rows to the database.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        rows
            Rows to persist.

        Returns
        -------
        int
            Number of rows persisted.
        """
        ...

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """
        Delete rows for specified module paths.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        rel_paths
            Relative paths of modules to delete.
        """
        ...


@runtime_checkable
class SupportsFullRebuild(Protocol):
    """Protocol for pipelines that support specialized full rebuild logic.

    When a pipeline implements this protocol, the executor will call
    run_full_rebuild instead of the normal incremental flow when
    full rebuild mode is detected.
    """

    def run_full_rebuild(
        self,
        gateway: StorageGateway,
        modules: Sequence[ModuleRecord],
        config: PipelineConfig,
    ) -> PipelineResult:
        """
        Execute a full rebuild of the dataset.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        modules
            All modules to process.
        config
            Pipeline configuration.

        Returns
        -------
        PipelineResult
            Result of the full rebuild.
        """
        ...


class PipelineExecutor[RowT]:
    """Execute ingestion pipelines with unified incremental/full mode handling.

    The executor handles:
    - Tracker presence detection (incremental vs full mode)
    - Worker pool lifecycle management
    - Batched row persistence with progress logging
    - Error aggregation and reporting

    Parameters
    ----------
    pipeline
        The pipeline implementation to execute.
    gateway
        Storage gateway for database access.
    config
        Pipeline execution configuration.
    """

    def __init__(
        self,
        pipeline: IngestPipeline[RowT],
        gateway: StorageGateway,
        config: PipelineConfig | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._gateway = gateway
        self._config = config or PipelineConfig()

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        tracker: ChangeTracker | None = None,
    ) -> PipelineResult:
        """
        Execute the pipeline against the provided modules.

        When tracker is provided, uses change detection to process only
        changed modules. When tracker is None, processes all modules.

        Parameters
        ----------
        modules
            Available modules for processing.
        tracker
            Optional change tracker for incremental mode.

        Returns
        -------
        PipelineResult
            Execution result with counts and timing.
        """
        start_time = time.perf_counter()
        result = PipelineResult(dataset_name=self._pipeline.dataset_name)

        # Determine what to process based on tracker
        if tracker is not None:
            view = tracker.view_for_dataset(
                dataset_name=self._pipeline.dataset_name,
                module_filter=self._pipeline.module_filter,
            )
            result.used_full_rebuild = view.use_full_rebuild

            # Check for full rebuild protocol
            if view.use_full_rebuild and isinstance(self._pipeline, SupportsFullRebuild):
                rebuild_result = self._pipeline.run_full_rebuild(
                    self._gateway,
                    view.to_reparse,
                    self._config,
                )
                result.modules_processed = rebuild_result.modules_processed
                result.rows_persisted = rebuild_result.rows_persisted
                result.errors = rebuild_result.errors
                result.duration_s = time.perf_counter() - start_time
                return result

            result = self._execute_incremental(view, result)
        else:
            # No tracker - process all modules that pass filter
            filtered = [m for m in modules if self._pipeline.module_filter(m)]
            result = self._execute_full(filtered, result)

        result.duration_s = time.perf_counter() - start_time

        log.info(
            "Pipeline %s completed: modules=%d rows=%d duration=%.2fs mode=%s",
            self._pipeline.dataset_name,
            result.modules_processed,
            result.rows_persisted,
            result.duration_s,
            "incremental" if tracker and not result.used_full_rebuild else "full",
        )

        return result

    def _execute_incremental(
        self,
        view: ChangeTrackerDatasetView,
        result: PipelineResult,
    ) -> PipelineResult:
        """
        Execute incremental processing based on change view.

        Returns
        -------
        PipelineResult
            Result updated after incremental processing.
        """
        # Delete rows for removed modules
        if view.deleted_paths:
            self._pipeline.delete_rows(self._gateway, view.deleted_paths)
            log.debug(
                "Deleted rows for %d paths in %s",
                len(view.deleted_paths),
                self._pipeline.dataset_name,
            )

        if not view.to_reparse:
            log.info(
                "No modules to reparse for %s (total=%d)",
                self._pipeline.dataset_name,
                view.total_modules_considered,
            )
            return result

        return self._process_modules(view.to_reparse, result)

    def _execute_full(
        self,
        modules: Sequence[ModuleRecord],
        result: PipelineResult,
    ) -> PipelineResult:
        """
        Execute full processing of all modules.

        Returns
        -------
        PipelineResult
            Result updated after full processing.
        """
        if not modules:
            log.info("No modules to process for %s", self._pipeline.dataset_name)
            return result

        return self._process_modules(modules, result)

    def _process_modules(
        self,
        modules: Sequence[ModuleRecord],
        result: PipelineResult,
    ) -> PipelineResult:
        """
        Process modules with optional parallelization.

        Returns
        -------
        PipelineResult
            Result updated with persisted row counts and totals.
        """
        # Resolve worker configuration
        worker_config = self._config.worker_config
        if worker_config is not None:
            workers = resolve_worker_count(
                worker_config.env_var,
                explicit_count=self._config.max_workers,
                default_max=worker_config.default_max,
                default_min=worker_config.default_min,
            )
            kind = self._config.executor_kind or worker_config.executor_kind
        else:
            workers = self._config.max_workers or 1
            kind = self._config.executor_kind or "thread"

        # Collect and persist rows
        rows: list[RowT] = []
        processed = 0

        if workers > 1:
            factory = executor_factory(kind, workers)
            rows, processed = self._process_parallel(modules, factory)
        else:
            rows, processed = self._process_sequential(modules)

        # Persist collected rows
        if rows:
            persisted = self._pipeline.persist_rows(self._gateway, rows)
            result.rows_persisted = persisted

        result.modules_processed = processed
        return result

    def _process_sequential(
        self,
        modules: Sequence[ModuleRecord],
    ) -> tuple[list[RowT], int]:
        """
        Process modules sequentially.

        Returns
        -------
        tuple[list[RowT], int]
            Collected rows and number of processed modules.
        """
        rows: list[RowT] = []
        processed = 0

        for idx, module in enumerate(modules, start=1):
            try:
                module_rows = list(self._pipeline.process_module(module))
                rows.extend(module_rows)
                processed += 1

                if idx % self._config.log_every == 0:
                    log.debug(
                        "Processed %d/%d modules for %s",
                        idx,
                        len(modules),
                        self._pipeline.dataset_name,
                    )

            except (OSError, ValueError, RuntimeError) as exc:
                log.warning(
                    "Error processing %s for %s: %s",
                    module.rel_path,
                    self._pipeline.dataset_name,
                    exc,
                )

        return rows, processed

    def _process_parallel(
        self,
        modules: Sequence[ModuleRecord],
        factory: Callable[[], Executor],
    ) -> tuple[list[RowT], int]:
        """
        Process modules in parallel using an executor.

        Returns
        -------
        tuple[list[RowT], int]
            Collected rows and number of processed modules.
        """
        rows: list[RowT] = []
        processed = 0

        with factory() as executor:
            futures_map = {
                executor.submit(self._pipeline.process_module, module): module
                for module in modules
            }

            for future, module in futures_map.items():
                try:
                    module_rows = list(future.result())
                    rows.extend(module_rows)
                    processed += 1
                except (OSError, ValueError, RuntimeError) as exc:
                    log.warning(
                        "Error processing %s for %s: %s",
                        module.rel_path,
                        self._pipeline.dataset_name,
                        exc,
                    )

        return rows, processed


def execute_pipeline[RowT](
    pipeline: IngestPipeline[RowT],
    gateway: StorageGateway,
    modules: Sequence[ModuleRecord],
    *,
    tracker: ChangeTracker | None = None,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """
    Execute an ingestion pipeline.

    This is the main entry point for running pipelines. It creates an
    executor and runs the pipeline against the provided modules.

    Parameters
    ----------
    pipeline
        Pipeline implementation to execute.
    gateway
        Storage gateway for database access.
    modules
        Available modules for processing.
    tracker
        Optional change tracker for incremental mode.
    config
        Optional pipeline configuration.

    Returns
    -------
    PipelineResult
        Execution result with counts and timing.
    """
    executor = PipelineExecutor(pipeline, gateway, config)
    return executor.execute(modules, tracker=tracker)


__all__ = [
    "DEFAULT_FLUSH_EVERY",
    "DEFAULT_LOG_EVERY",
    "IngestPipeline",
    "PipelineConfig",
    "PipelineExecutor",
    "PipelineResult",
    "SupportsFullRebuild",
    "execute_pipeline",
]
