"""Hash-based change detection adapter implementing ChangeDetectionPort.

This adapter detects file changes using Blake2b content hashing and can
optionally compare against parquet-backed file_state snapshots.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, SupportsInt, cast

import pyarrow as pa

from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.columnar.rows import ColumnarRows, columnar_buffer_for_table_key
from codeintel.core.datasets.scanning import (
    ParquetScanOptions,
    ParquetScanTelemetry,
    scan_parquet_dataset_with_telemetry,
)
from codeintel.core.hashing import sha256_short
from codeintel.core.paths import normalize_path
from codeintel.ingestion.context import IngestionContext
from codeintel.ingestion.ports.change_detection import ChangeRequest, ChangeSet, FileDigest
from codeintel.ingestion.ports.discovery import ModuleRecord

if TYPE_CHECKING:
    from codeintel.ingestion.infrastructure.scanning import ScanProfile

log = logging.getLogger(__name__)
FILE_STATE_TABLE_KEY = "core.file_state"
_REQUIRED_STATE_COLUMNS: tuple[str, ...] = ("rel_path", "size_bytes", "mtime_ns", "content_hash")
_READ_EXCEPTIONS = (
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    pa.ArrowInvalid,
)


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, SupportsInt):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _scan_telemetry_payload(telemetry: ParquetScanTelemetry) -> dict[str, int | None]:
    return {
        "fragment_count": telemetry.fragment_count,
        "estimated_rows": telemetry.row_count,
    }


def _scan_telemetry_mapping(
    telemetry: ParquetScanTelemetry | None,
) -> dict[str, dict[str, int | None]]:
    if telemetry is None:
        return {}
    return {FILE_STATE_TABLE_KEY: _scan_telemetry_payload(telemetry)}


class HashChangeDetectionAdapter:
    """Hash-based change detection adapter implementing ChangeDetectionPort.

    This adapter detects file changes by computing Blake2b content hashes
    and comparing against previously stored file_state snapshots when
    a dataset root is configured.

    Parameters
    ----------
    dataset_root
        Optional dataset root directory for reading prior file_state snapshots.
    snapshot_id
        Optional snapshot identifier used for parquet dataset lookups.
    execution_ctx
        Optional execution context for scan defaults and thread pools.
    """

    def __init__(
        self,
        *,
        dataset_root: Path | None = None,
        snapshot_id: str | None = None,
        execution_ctx: ExecutionContext | None = None,
    ) -> None:
        self._dataset_root = dataset_root
        self._snapshot_id = snapshot_id
        self._execution_ctx = execution_ctx

    def compute_changes(
        self,
        request: ChangeRequest,
        current_modules: Sequence[ModuleRecord],
    ) -> ChangeSet:
        """Compute changes between previous and current state.

        Parameters
        ----------
        request
            Change detection request parameters.
        current_modules
            Current modules discovered in the repository.

        Returns
        -------
        ChangeSet
            Detected changes (added, modified, deleted).
        """
        current_state = self._build_current_state(current_modules)
        state_hash = self.compute_state_hash(current_state)

        previous_state: Mapping[str, FileDigest]
        scan_telemetry: dict[str, dict[str, int | None]] = {}
        if request.full_rebuild:
            previous_state = {}
        else:
            previous_state, scan_telemetry = self._safe_previous_state_with_telemetry(
                request.repo,
                request.language,
            )

        added: list[ModuleRecord] = []
        modified: list[ModuleRecord] = []
        current_paths = set()

        for module in current_modules:
            normalized_path = normalize_path(module.rel_path)
            current_paths.add(normalized_path)

            current_digest = current_state.get(normalized_path)
            if current_digest is None:
                continue

            previous_digest = previous_state.get(normalized_path)

            if previous_digest is None:
                added.append(module)
            elif current_digest.content_hash != previous_digest.content_hash:
                modified.append(module)

        deleted = [
            ModuleRecord(
                rel_path=rel_path,
                module_name="<deleted>",
                file_path=request.repo_root / rel_path,
                index=0,
                total=0,
            )
            for rel_path in previous_state
            if rel_path not in current_paths
        ]

        state_rows = self._build_state_rows(
            repo=request.repo,
            commit=request.commit,
            language=request.language,
            state=current_state,
        )

        log.info(
            "Change detection: repo=%s added=%d modified=%d deleted=%d",
            request.repo,
            len(added),
            len(modified),
            len(deleted),
        )

        return ChangeSet(
            added=added,
            modified=modified,
            deleted=deleted,
            state_hash=state_hash,
            state_rows=state_rows,
            scan_telemetry=scan_telemetry,
        )

    def compute_changes_for_context(
        self,
        *,
        context: IngestionContext,
        current_modules: Sequence[ModuleRecord],
        language: str = "python",
        full_rebuild: bool = False,
        scan_profile: ScanProfile | None = None,
    ) -> ChangeSet:
        """Compute changes using ingestion context defaults.

        Returns
        -------
        ChangeSet
            Detected changes using the context-derived request.
        """
        request = ChangeRequest.from_context(
            context=context,
            language=language,
            full_rebuild=full_rebuild,
            scan_profile=scan_profile,
        )
        return self.compute_changes(request, current_modules)

    def load_previous_state(
        self,
        repo: str,
        language: str,
    ) -> Mapping[str, FileDigest]:
        """Load the previous file state from parquet snapshots.

        Parameters
        ----------
        repo
            Repository identifier.
        language
            Source language.

        Returns
        -------
        Mapping[str, FileDigest]
            Mapping from relative path to file digest.
        """
        state, _ = self._load_previous_state_with_telemetry(repo, language)
        return state

    def _load_previous_state_with_telemetry(
        self,
        repo: str,
        language: str,
    ) -> tuple[Mapping[str, FileDigest], dict[str, dict[str, int | None]]]:
        rows_iter, scan_telemetry = self._rows_iter_for_previous_state(repo, language)
        if rows_iter is None:
            return {}, scan_telemetry

        state, completed = self._build_state_from_rows(rows_iter, repo=repo, language=language)
        if completed and not state:
            log.info("No previous file_state rows found for %s", repo)

        return state, scan_telemetry

    @staticmethod
    def save_current_state(
        repo: str,
        commit: str,
        language: str,
        state: Mapping[str, FileDigest],
    ) -> None:
        """Save the current file state.

        Persistence is handled by the ingestion materializers; this adapter
        does not write state directly.
        """
        _ = (repo, commit, language, state)

    @staticmethod
    def compute_file_digest(path: Path) -> FileDigest | None:
        """Compute the digest for a single file.

        Parameters
        ----------
        path
            Path to the file.

        Returns
        -------
        FileDigest | None
            File digest if readable, None otherwise.
        """
        try:
            stat_result = path.stat()
            content = path.read_bytes()
            digest = hashlib.blake2b(content, digest_size=16).hexdigest()
            return FileDigest(
                size_bytes=stat_result.st_size,
                mtime_ns=stat_result.st_mtime_ns,
                content_hash=digest,
            )
        except OSError as exc:
            log.warning("Failed to stat %s: %s", path, exc)
            return None

    @staticmethod
    def compute_state_hash(state: Mapping[str, FileDigest]) -> str:
        """Compute a stable hash for the current file state.

        Parameters
        ----------
        state
            Mapping from relative path to file digest.

        Returns
        -------
        str
            Stable hash derived from path + content hashes.
        """
        parts = [f"{rel_path}:{digest.content_hash}" for rel_path, digest in sorted(state.items())]
        payload = "|".join(parts)
        if payload:
            payload = f"{payload}|"
        return sha256_short(payload, length=16, used_for_security=False)

    def _safe_previous_state_with_telemetry(
        self,
        repo: str,
        language: str,
    ) -> tuple[Mapping[str, FileDigest], dict[str, dict[str, int | None]]]:
        try:
            return self._load_previous_state_with_telemetry(repo, language)
        except _READ_EXCEPTIONS as exc:
            log.warning(
                "file_state previous state load failed for repo=%s language=%s: %s",
                repo,
                language,
                exc,
            )
            return {}, {}

    def _rows_iter_for_previous_state(
        self,
        repo: str,
        language: str,
    ) -> tuple[
        Iterable[tuple[object, object, object, object]] | None,
        dict[str, dict[str, int | None]],
    ]:
        dataset_root = self._dataset_root
        snapshot_id = self._snapshot_id
        if dataset_root is None or snapshot_id is None:
            return None, {}

        options = ParquetScanOptions(
            repo=repo,
            commit=snapshot_id,
            execution_ctx=self._execution_ctx,
        )
        try:
            reader, telemetry = scan_parquet_dataset_with_telemetry(
                dataset_root=dataset_root,
                table_key=FILE_STATE_TABLE_KEY,
                snapshot_id=snapshot_id,
                options=options,
            )
        except _READ_EXCEPTIONS as exc:
            log.warning(
                "file_state dataset scan failed for repo=%s snapshot=%s: %s",
                repo,
                snapshot_id,
                exc,
            )
            return None, {}
        scan_telemetry = _scan_telemetry_mapping(telemetry)
        if reader is None:
            return None, scan_telemetry

        available = set(reader.schema.names)
        missing = [name for name in _REQUIRED_STATE_COLUMNS if name not in available]
        if missing:
            log.warning("file_state dataset missing columns: %s", missing)
            return None, scan_telemetry

        columns = list(_REQUIRED_STATE_COLUMNS)
        include_language = "language" in available
        if include_language:
            columns.append("language")
        rows = iter_tuples(reader, columns=columns)
        if not include_language:
            return cast("Iterable[tuple[object, object, object, object]]", rows), scan_telemetry
        rows_with_language = cast("Iterable[tuple[object, object, object, object, object]]", rows)
        return self._filter_language(rows_with_language, language=language), scan_telemetry

    @staticmethod
    def _filter_language(
        rows: Iterable[tuple[object, object, object, object, object]],
        *,
        language: str,
    ) -> Iterable[tuple[object, object, object, object]]:
        for rel_path, size_bytes, mtime_ns, content_hash, row_language in rows:
            if row_language is None:
                continue
            if str(row_language) != language:
                continue
            yield rel_path, size_bytes, mtime_ns, content_hash

    @staticmethod
    def _build_state_from_rows(
        rows_iter: Iterable[tuple[object, object, object, object]],
        *,
        repo: str,
        language: str,
    ) -> tuple[dict[str, FileDigest], bool]:
        state: dict[str, FileDigest] = {}
        try:
            for rel_path, size_bytes, mtime_ns, content_hash in rows_iter:
                if rel_path is None or content_hash is None:
                    continue
                size_value = _coerce_int(size_bytes)
                mtime_value = _coerce_int(mtime_ns)
                if size_value is None or mtime_value is None:
                    continue
                normalized = normalize_path(str(rel_path))
                digest = FileDigest(
                    size_bytes=size_value,
                    mtime_ns=mtime_value,
                    content_hash=str(content_hash),
                )
                existing = state.get(normalized)
                if existing is None or digest.mtime_ns >= existing.mtime_ns:
                    state[normalized] = digest
        except _READ_EXCEPTIONS as exc:
            log.warning(
                "file_state iteration failed for repo=%s language=%s: %s",
                repo,
                language,
                exc,
            )
            return {}, False
        return state, True

    def _build_current_state(
        self,
        modules: Sequence[ModuleRecord],
    ) -> dict[str, FileDigest]:
        """Build current file state from modules.

        Parameters
        ----------
        modules
            Current modules.

        Returns
        -------
        dict[str, FileDigest]
            Mapping from normalized path to digest.
        """
        state: dict[str, FileDigest] = {}
        for module in modules:
            digest = self.compute_file_digest(module.file_path)
            if digest is not None:
                normalized = normalize_path(module.rel_path)
                state[normalized] = digest
        return state

    @staticmethod
    def _build_state_rows(
        *,
        repo: str,
        commit: str,
        language: str,
        state: Mapping[str, FileDigest],
    ) -> ColumnarRows:
        if not state:
            return {}
        buffer = columnar_buffer_for_table_key(FILE_STATE_TABLE_KEY)
        for rel_path, digest in sorted(state.items()):
            buffer.append(
                {
                    "repo": repo,
                    "commit": commit,
                    "rel_path": rel_path,
                    "language": language,
                    "size_bytes": digest.size_bytes,
                    "mtime_ns": digest.mtime_ns,
                    "content_hash": digest.content_hash,
                }
            )
        return buffer.data


__all__ = ["HashChangeDetectionAdapter"]
