"""Hash-based change detection adapter implementing ChangeDetectionPort.

This adapter detects file changes using Blake2b content hashing and
persists state to DuckDB.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, SupportsInt, cast

from codeintel.core.columnar.rows import ColumnarRows, columnar_buffer_for_table_key
from codeintel.core.hashing import sha256_short
from codeintel.core.paths import normalize_path
from codeintel.core.schemas.row_serialization import row_serializer_for_table_key
from codeintel.ingestion.context import IngestionContext
from codeintel.ingestion.ports.change_detection import (
    ChangeRequest,
    ChangeSet,
    FileDigest,
)
from codeintel.ingestion.ports.discovery import ModuleRecord
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.duckdb_types import (
    ColumnExpression,
    ConstantExpression,
    DuckDBError,
    DuckDBRelation,
)
from codeintel.storage.query_results import (
    iter_tuples_from_arrow_reader,
    iter_tuples_from_relation,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from codeintel.ingestion.infrastructure.scanning import ScanProfile
    from codeintel.ingestion.ports.storage import IngestStoragePort
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)
FILE_STATE_TABLE_KEY = "core.file_state"
READ_EXCEPTIONS = (
    DuckDBError,
    FileNotFoundError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
RELATION_EXCEPTIONS = (DuckDBError, RuntimeError, TypeError, ValueError)


class HashChangeDetectionAdapter:
    """Hash-based change detection adapter implementing ChangeDetectionPort.

    This adapter detects file changes by computing Blake2b content hashes
    and comparing against previously stored state.

    Parameters
    ----------
    storage
        Storage port for persisting file state.
    """

    def __init__(self, storage: IngestStoragePort) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        storage
            Storage port for persisting file state.
        """
        self._storage = storage

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

        try:
            previous_state = self.load_previous_state(request.repo, request.language)
        except (
            DuckDBError,
            FileNotFoundError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            log.warning(
                "file_state previous state load failed for repo=%s language=%s: %s",
                request.repo,
                request.language,
                exc,
            )
            previous_state = {}

        added: list[ModuleRecord] = []
        modified: list[ModuleRecord] = []
        current_paths = set()

        for module in current_modules:
            normalized_path = normalize_path(module.rel_path)
            current_paths.add(normalized_path)

            if normalized_path not in current_state:
                continue

            current_digest = current_state[normalized_path]
            previous_digest = previous_state.get(normalized_path)

            if previous_digest is None:
                added.append(module)
            elif current_digest.content_hash != previous_digest.content_hash:
                modified.append(module)

        deleted: list[ModuleRecord] = []
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
        """Load the previous file state from storage.

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
        rows_iter = self._rows_iter_for_previous_state(repo, language)
        if rows_iter is None:
            return {}

        state, completed = self._build_state_from_rows(rows_iter, repo=repo, language=language)
        if completed and not state:
            log.info("No previous file_state rows found for %s", repo)

        return state

    def _rows_iter_for_previous_state(
        self,
        repo: str,
        language: str,
    ) -> Iterable[tuple[object, object, object, object]] | None:
        gateway = getattr(self._storage, "_gateway", None)
        if gateway is None:
            return self._rows_iter_from_storage(repo, language)
        return self._rows_iter_from_gateway(cast("StorageGateway", gateway), repo, language)

    def _rows_iter_from_gateway(
        self,
        gateway: StorageGateway,
        repo: str,
        language: str,
    ) -> Iterable[tuple[object, object, object, object]] | None:
        relation = self._filtered_relation_from_gateway(gateway, repo, language)
        if relation is None:
            return None
        try:
            return cast(
                "Iterable[tuple[object, object, object, object]]",
                iter_tuples_from_relation(relation),
            )
        except READ_EXCEPTIONS as exc:
            log.warning(
                "file_state relation load failed for repo=%s language=%s: %s",
                repo,
                language,
                exc,
            )
            return None

    def _filtered_relation_from_gateway(
        self,
        gateway: StorageGateway,
        repo: str,
        language: str,
    ) -> DuckDBRelation | None:
        base_relation = self._base_relation_from_gateway(gateway, repo, language)
        if base_relation is None:
            return None
        predicate = (ColumnExpression("repo") == ConstantExpression(repo)) & (
            ColumnExpression("language") == ConstantExpression(language)
        )
        try:
            return base_relation.filter(predicate).select(
                "rel_path",
                "size_bytes",
                "mtime_ns",
                "content_hash",
            )
        except RELATION_EXCEPTIONS as exc:
            log.warning(
                "file_state load failed for repo=%s language=%s: %s",
                repo,
                language,
                exc,
            )
            return None

    def _base_relation_from_gateway(
        self,
        gateway: StorageGateway,
        repo: str,
        language: str,
    ) -> DuckDBRelation | None:
        try:
            return gateway.relation_from_table_key(FILE_STATE_TABLE_KEY)
        except RELATION_EXCEPTIONS as exc:
            if self._should_skip_missing_dataset(gateway, repo, language, exc):
                return None
            return self._fallback_relation(gateway, repo, language, exc)

    def _should_skip_missing_dataset(
        self,
        gateway: StorageGateway,
        repo: str,
        language: str,
        exc: Exception,
    ) -> bool:
        dataset = self._dataset_for_table_key(gateway, FILE_STATE_TABLE_KEY)
        config = getattr(gateway, "config", None)
        dataset_root_dir = getattr(config, "dataset_root_dir", None)
        snapshot_id = getattr(config, "commit", None)
        is_view = bool(getattr(dataset, "is_view", False)) if dataset is not None else False
        if dataset is None or is_view or dataset_root_dir is None or snapshot_id is None:
            return False
        log.info(
            "file_state dataset missing for repo=%s language=%s snapshot=%s; "
            "skipping previous state: %s",
            repo,
            language,
            snapshot_id,
            exc,
        )
        return True

    @staticmethod
    def _dataset_for_table_key(
        gateway: StorageGateway,
        table_key: str,
    ) -> object | None:
        datasets = getattr(gateway, "datasets", None)
        by_table_key = getattr(datasets, "by_table_key", None)
        if isinstance(by_table_key, Mapping):
            return by_table_key.get(table_key)
        return None

    @staticmethod
    def _fallback_relation(
        gateway: StorageGateway,
        repo: str,
        language: str,
        exc: Exception,
    ) -> DuckDBRelation | None:
        try:
            gateway.policy.ensure_table(FILE_STATE_TABLE_KEY)
            base_relation = gateway.table(FILE_STATE_TABLE_KEY)
        except RELATION_EXCEPTIONS as table_exc:
            log.warning(
                "file_state fallback failed for repo=%s language=%s: %s",
                repo,
                language,
                table_exc,
            )
            return None
        log.info(
            "file_state manifest missing for repo=%s language=%s; falling back to DuckDB table: %s",
            repo,
            language,
            exc,
        )
        return base_relation

    def _rows_iter_from_storage(
        self,
        repo: str,
        language: str,
    ) -> Iterable[tuple[object, object, object, object]] | None:
        self._storage.ensure_schema(FILE_STATE_TABLE_KEY)
        try:
            reader = self._storage.fetch_arrow_reader(
                """
                WITH ranked AS (
                    SELECT
                        rel_path,
                        size_bytes,
                        mtime_ns,
                        content_hash,
                        ROW_NUMBER() OVER (PARTITION BY rel_path ORDER BY mtime_ns DESC) AS rn
                    FROM core.file_state
                    WHERE repo = ? AND language = ?
                )
                SELECT rel_path, size_bytes, mtime_ns, content_hash
                FROM ranked
                WHERE rn = 1
                """,
                [repo, language],
            )
        except READ_EXCEPTIONS as exc:
            log.warning(
                "file_state reader load failed for repo=%s language=%s: %s",
                repo,
                language,
                exc,
            )
            return None
        try:
            return cast(
                "Iterable[tuple[object, object, object, object]]",
                iter_tuples_from_arrow_reader(reader),
            )
        except READ_EXCEPTIONS as exc:
            log.warning(
                "file_state reader load failed for repo=%s language=%s: %s",
                repo,
                language,
                exc,
            )
            return None

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
                normalized = normalize_path(str(rel_path))
                digest = FileDigest(
                    size_bytes=int(cast("SupportsInt", size_bytes)),
                    mtime_ns=int(cast("SupportsInt", mtime_ns)),
                    content_hash=str(content_hash),
                )
                existing = state.get(normalized)
                if existing is None or digest.mtime_ns >= existing.mtime_ns:
                    state[normalized] = digest
        except READ_EXCEPTIONS as exc:
            log.warning(
                "file_state iteration failed for repo=%s language=%s: %s",
                repo,
                language,
                exc,
            )
            return {}, False
        return state, True

    def save_current_state(
        self,
        repo: str,
        commit: str,
        language: str,
        state: Mapping[str, FileDigest],
    ) -> None:
        """Save the current file state to storage.

        Parameters
        ----------
        repo
            Repository identifier.
        commit
            Commit identifier.
        language
            Source language.
        state
            Mapping from relative path to file digest.
        """
        if not state:
            return

        gateway = getattr(self._storage, "_gateway", None)
        serializer = row_serializer_for_table_key(FILE_STATE_TABLE_KEY)
        rows = [
            serializer(
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
            for rel_path, digest in sorted(state.items())
        ]
        if gateway is not None:
            backend = DuckDBPolicyBackend(cast("StorageGateway", gateway))
            backend.delete_for_snapshot(FILE_STATE_TABLE_KEY, repo=repo, commit=commit)
            backend.bulk_insert(FILE_STATE_TABLE_KEY, rows)
            return

        self._storage.ensure_schema(FILE_STATE_TABLE_KEY)
        for rel_path in state:
            self._storage.execute_query(
                "DELETE FROM core.file_state WHERE repo = ? AND rel_path = ? AND language = ?",
                [repo, rel_path, language],
            )
        self._storage.write_batch(FILE_STATE_TABLE_KEY, rows)

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
