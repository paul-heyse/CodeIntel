"""Repository scanning step with port injection.

This module provides a pure domain logic implementation for scanning
repository modules and building change tracker state, using ports
for all I/O operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.columnar.dedupe_ops import stable_dedupe_with_ties
from codeintel.core.columnar.iter import iter_rows
from codeintel.core.columnar.rows import (
    ColumnarRows,
    columnar_buffer_for_table_key,
    empty_table_for_table,
    reader_for_columnar_rows,
)
from codeintel.ingestion.context import (
    IngestionContext,
    resolve_repo_commit,
    resolve_repo_root,
    resolve_scan_profile,
)
from codeintel.ingestion.ports.change_detection import ChangeRequest
from codeintel.ingestion.ports.discovery import ModuleRecord

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from codeintel.config.primitives import SnapshotRef
    from codeintel.ingestion.infrastructure.scanning import ScanProfile
    from codeintel.ingestion.ports.change_detection import ChangeDetectionPort, ChangeSet
    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort

log = logging.getLogger(__name__)
MODULES_TABLE_KEY = "core.modules"
FILE_STATE_TABLE_KEY = "core.file_state"
REPO_MAP_TABLE_KEY = "core.repo_map"
REPO_SCAN_TARGET_NAME = "repo_scan"


@dataclass(frozen=True)
class RepoScanResult:
    """Result from repository scanning.

    Attributes
    ----------
    modules
        Discovered module records.
    change_set
        Change set describing added/modified/deleted modules.
    module_rows
        Columnar rows for core.modules.
    file_state_rows
        Columnar rows for core.file_state.
    repo_map_rows
        Columnar rows for core.repo_map.
    """

    modules: tuple[ModuleRecord, ...]
    change_set: ChangeSet
    module_rows: ColumnarRows
    file_state_rows: ColumnarRows
    repo_map_rows: ColumnarRows
    module_rows_reader: pa.RecordBatchReader = field(
        default_factory=lambda: empty_table_for_table(MODULES_TABLE_KEY).to_reader()
    )
    file_state_rows_reader: pa.RecordBatchReader = field(
        default_factory=lambda: empty_table_for_table(FILE_STATE_TABLE_KEY).to_reader()
    )
    repo_map_rows_reader: pa.RecordBatchReader = field(
        default_factory=lambda: empty_table_for_table(REPO_MAP_TABLE_KEY).to_reader()
    )


@dataclass(frozen=True, slots=True)
class _RepoScanContext:
    repo: str
    commit: str
    root: Path
    profile: ScanProfile
    change_request: ChangeRequest


@dataclass(frozen=True, slots=True)
class _RepoScanTables:
    module_rows: ColumnarRows
    file_state_rows: ColumnarRows
    repo_map_rows: ColumnarRows
    module_rows_reader: pa.RecordBatchReader
    file_state_rows_reader: pa.RecordBatchReader
    repo_map_rows_reader: pa.RecordBatchReader

    def as_mapping(self) -> dict[str, pa.RecordBatchReader]:
        return {
            MODULES_TABLE_KEY: self.module_rows_reader,
            FILE_STATE_TABLE_KEY: self.file_state_rows_reader,
            REPO_MAP_TABLE_KEY: self.repo_map_rows_reader,
        }


def _resolve_repo_scan_context(
    *,
    snapshot: SnapshotRef | None,
    repo_root: Path | None,
    profile: ScanProfile | None,
    full_rebuild: bool,
    context: IngestionContext | None,
) -> _RepoScanContext:
    resolved_repo, resolved_commit = resolve_repo_commit(
        context=context,
        repo=snapshot.repo if snapshot is not None else None,
        commit=snapshot.commit if snapshot is not None else None,
    )
    resolved_root = resolve_repo_root(
        context=context,
        repo_root=snapshot.repo_root if snapshot is not None else repo_root,
    )
    resolved_profile = resolve_scan_profile(context=context, scan_profile=profile)
    if context is not None:
        change_request = ChangeRequest.from_context(
            context=context,
            language="python",
            full_rebuild=full_rebuild,
            scan_profile=resolved_profile,
        )
    else:
        change_request = ChangeRequest(
            repo=resolved_repo,
            commit=resolved_commit,
            repo_root=resolved_root,
            language="python",
            full_rebuild=full_rebuild,
            scan_profile=resolved_profile,
        )
    return _RepoScanContext(
        repo=resolved_repo,
        commit=resolved_commit,
        root=resolved_root,
        profile=resolved_profile,
        change_request=change_request,
    )


class RepoScanStep:
    """Repository scanning step with port injection.

    This step scans repository modules and builds change tracker state,
    using ports for all I/O operations.

    Parameters
    ----------
    discovery
        Discovery port for finding modules.
    change_detection
        Change detection port for computing changes.
    """

    def __init__(
        self,
        discovery: ModuleDiscoveryPort,
        change_detection: ChangeDetectionPort,
        module_filter: Callable[[Sequence[ModuleRecord]], Sequence[ModuleRecord]] | None = None,
    ) -> None:
        """Initialize the step.

        Parameters
        ----------
        discovery
            Discovery port for finding modules.
        change_detection
            Change detection port for computing changes.
        module_filter
            Optional filter applied to discovered modules before persistence.
        """
        self._discovery = discovery
        self._change_detection = change_detection
        self._module_filter = module_filter

    def execute(
        self,
        *,
        snapshot: SnapshotRef | None = None,
        repo_root: Path | None = None,
        profile: ScanProfile | None = None,
        full_rebuild: bool = False,
        context: IngestionContext | None = None,
    ) -> RepoScanResult:
        """Execute repository scanning.

        Parameters
        ----------
        snapshot
            Optional snapshot reference (provides repo/commit/root when set).
        repo_root
            Repository root path.
        profile
            Scan profile for module discovery.
        full_rebuild
            Whether to force a full rebuild.
        context
            Optional ingestion context for repo/commit/root resolution.

        Returns
        -------
        RepoScanResult
            Discovered modules, change set, and row tuples.
        """
        resolved = _resolve_repo_scan_context(
            snapshot=snapshot,
            repo_root=repo_root,
            profile=profile,
            full_rebuild=full_rebuild,
            context=context,
        )
        modules = list(self._discovery.discover_modules(resolved.root, resolved.profile))
        if self._module_filter is not None:
            modules = list(self._module_filter(modules))
        modules = _dedupe_modules(modules)
        log.info("Discovered %d modules in %s", len(modules), resolved.root)
        change_set = self._change_detection.compute_changes(resolved.change_request, modules)
        tables = self._build_repo_scan_tables(
            modules,
            change_set,
            repo=resolved.repo,
            commit=resolved.commit,
        )

        log.info(
            "Repo scan: repo=%s commit=%s modules=%d added=%d modified=%d deleted=%d",
            resolved.repo,
            resolved.commit,
            len(modules),
            len(change_set.added),
            len(change_set.modified),
            len(change_set.deleted),
        )

        return RepoScanResult(
            modules=tuple(modules),
            change_set=change_set,
            module_rows=tables.module_rows,
            file_state_rows=tables.file_state_rows,
            repo_map_rows=tables.repo_map_rows,
            module_rows_reader=tables.module_rows_reader,
            file_state_rows_reader=tables.file_state_rows_reader,
            repo_map_rows_reader=tables.repo_map_rows_reader,
        )

    def _build_repo_scan_tables(
        self,
        modules: Sequence[ModuleRecord],
        change_set: ChangeSet,
        *,
        repo: str,
        commit: str,
    ) -> _RepoScanTables:
        module_buffer = columnar_buffer_for_table_key(MODULES_TABLE_KEY)
        for module in modules:
            module_buffer.append(
                {
                    "module": module.module_name,
                    "path": module.rel_path,
                    "repo": repo,
                    "commit": commit,
                    "language": "python",
                    "tags": [],
                    "owners": [],
                    "row_hash": None,
                }
            )
        repo_map_rows = self._build_repo_map_rows(
            repo=repo,
            commit=commit,
            modules=modules,
        )
        module_rows_reader, _ = reader_for_columnar_rows(
            MODULES_TABLE_KEY,
            module_buffer.data,
        )
        file_state_rows_reader, _ = reader_for_columnar_rows(
            FILE_STATE_TABLE_KEY,
            change_set.state_rows,
        )
        repo_map_rows_reader, _ = reader_for_columnar_rows(
            REPO_MAP_TABLE_KEY,
            repo_map_rows,
        )
        return _RepoScanTables(
            module_rows=module_buffer.data,
            file_state_rows=change_set.state_rows,
            repo_map_rows=repo_map_rows,
            module_rows_reader=module_rows_reader,
            file_state_rows_reader=file_state_rows_reader,
            repo_map_rows_reader=repo_map_rows_reader,
        )

    @staticmethod
    def _build_repo_map_rows(
        *,
        repo: str,
        commit: str,
        modules: Sequence[ModuleRecord],
    ) -> ColumnarRows:
        if not modules:
            return {}
        module_entries: dict[str, str] = {}
        for module in modules:
            module_entries[str(module.module_name)] = str(module.rel_path)
        buffer = columnar_buffer_for_table_key(REPO_MAP_TABLE_KEY)
        buffer.append(
            {
                "repo": repo,
                "commit": commit,
                "modules": module_entries,
                "overlays": {},
                "generated_at": datetime.now(tz=UTC),
            }
        )
        return buffer.data


__all__ = ["RepoScanResult", "RepoScanStep"]


def _dedupe_modules(modules: Sequence[ModuleRecord]) -> list[ModuleRecord]:
    if not modules:
        return []
    rows = {
        "module_name": [module.module_name for module in modules],
        "rel_path": [module.rel_path for module in modules],
        "file_path": [str(module.file_path) for module in modules],
        "row_index": list(range(len(modules))),
    }
    table = pa.table(rows)
    deduped = stable_dedupe_with_ties(
        table,
        key_columns=("module_name", "rel_path"),
        order_by=(("row_index", "ascending"),),
    )
    ordered = (
        deduped.sort_by([("row_index", "ascending")])
        if "row_index" in deduped.column_names
        else deduped
    )
    total = ordered.num_rows
    records: list[ModuleRecord] = []
    for index, row in enumerate(
        iter_rows(ordered, columns=("rel_path", "module_name", "file_path")),
        start=1,
    ):
        rel_path = row.get("rel_path")
        module_name = row.get("module_name")
        file_path = row.get("file_path")
        records.append(
            ModuleRecord(
                rel_path=rel_path if isinstance(rel_path, str) else "",
                module_name=module_name if isinstance(module_name, str) else "",
                file_path=Path(file_path) if isinstance(file_path, str) else Path(),
                index=index,
                total=total,
            )
        )
    return records
