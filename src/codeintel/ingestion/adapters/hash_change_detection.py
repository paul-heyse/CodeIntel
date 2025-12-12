"""Hash-based change detection adapter implementing ChangeDetectionPort.

This adapter detects file changes using Blake2b content hashing and
persists state to DuckDB.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, cast

import ibis

from codeintel.config.datasets import load_columns_by_table
from codeintel.ingestion.infrastructure.paths import normalize_rel_path
from codeintel.ingestion.ports.change_detection import (
    ChangeSet,
    FileDigest,
)
from codeintel.ingestion.ports.discovery import ModuleRecord
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.ibis_types import filter_by, ibis_bool, window_over

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from codeintel.ingestion.ports.change_detection import (
        ChangeRequest,
    )
    from codeintel.ingestion.ports.storage import IngestStoragePort
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


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

        previous_state = self.load_previous_state(request.repo, request.language)

        added: list[ModuleRecord] = []
        modified: list[ModuleRecord] = []
        current_paths = set()

        for module in current_modules:
            normalized_path = normalize_rel_path(module.rel_path)
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

        self.save_current_state(request.repo, request.commit, request.language, current_state)

        log.info(
            "Change detection: repo=%s added=%d modified=%d deleted=%d",
            request.repo,
            len(added),
            len(modified),
            len(deleted),
        )

        return ChangeSet(added=added, modified=modified, deleted=deleted)

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
        gateway = getattr(self._storage, "_gateway", None)
        if gateway is not None:
            file_state = gateway.ibis.table("core.file_state")
            window = window_over(
                partition_by=[file_state.rel_path],
                order_by=[file_state.mtime_ns.desc()],
            )
            rn_expr = ibis.row_number().over(window)
            ranked = (
                filter_by(file_state, file_state.repo == repo, file_state.language == language)
                .mutate(rn=rn_expr)
                .filter(ibis_bool(rn_expr == 0))
                .select("rel_path", "size_bytes", "mtime_ns", "content_hash")
            )
            df = ranked.execute()
            rows = df.to_dict(orient="records")
        else:
            self._storage.ensure_schema("core.file_state")
            result = self._storage.execute_query(
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
            rows = [
                {
                    "rel_path": rel_path,
                    "size_bytes": size_bytes,
                    "mtime_ns": mtime_ns,
                    "content_hash": content_hash,
                }
                for rel_path, size_bytes, mtime_ns, content_hash in result.rows
            ]

        state: dict[str, FileDigest] = {}
        for row in rows:
            normalized = normalize_rel_path(str(row["rel_path"]))
            state[normalized] = FileDigest(
                size_bytes=int(row["size_bytes"]),
                mtime_ns=int(row["mtime_ns"]),
                content_hash=str(row["content_hash"]),
            )

        if not state:
            log.info("No previous file_state rows found for %s", repo)

        return state

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
        rows = [
            (
                repo,
                commit,
                rel_path,
                language,
                digest.size_bytes,
                digest.mtime_ns,
                digest.content_hash,
            )
            for rel_path, digest in sorted(state.items())
        ]
        if gateway is not None:
            backend = DuckDBPolicyBackend(cast("StorageGateway", gateway))
            columns = load_columns_by_table().get("core.file_state", [])
            backend.delete_for_snapshot("core.file_state", repo=repo, commit=commit)
            backend.bulk_insert("core.file_state", rows, columns=columns)
            return

        self._storage.ensure_schema("core.file_state")
        for rel_path in state:
            self._storage.execute_query(
                "DELETE FROM core.file_state WHERE repo = ? AND rel_path = ? AND language = ?",
                [repo, rel_path, language],
            )
        self._storage.write_batch("core.file_state", rows)

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
                normalized = normalize_rel_path(module.rel_path)
                state[normalized] = digest
        return state


__all__ = ["HashChangeDetectionAdapter"]
