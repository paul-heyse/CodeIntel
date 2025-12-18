"""Build run persistence utilities.

This module centralizes persistence for the Hamilton build executor:

- start/complete build run records
- persist per-target run records
- persist node-level telemetry
- emit Phase 4 asset catalog records

All persistence operations are best-effort: storage failures are logged and execution continues.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import duckdb
import ibis
import pyarrow as pa
import sqlglot

from codeintel.build.assets.emitter import persist_asset_catalog_for_run
from codeintel.core.build_manifest import BuildRunRecord
from codeintel.storage.exceptions import StorageError
from codeintel.storage.tracking.asset_tracking import RunEnvironmentRecord

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.targets import TargetGraph
    from codeintel.hamilton.records import NodeExecutionRecord, TargetRunRecord
    from codeintel.storage.gateway.protocol import StorageGateway

log = logging.getLogger(__name__)


def _sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tool_versions() -> dict[str, str]:
    return {
        "duckdb": str(getattr(duckdb, "__version__", "unknown")),
        "ibis": str(getattr(ibis, "__version__", "unknown")),
        "pyarrow": str(getattr(pa, "__version__", "unknown")),
        "sqlglot": str(getattr(sqlglot, "__version__", "unknown")),
    }


def _config_hash(env: BuildEnv) -> str | None:
    try:
        raw = getattr(env.config, "_raw", None)
        if not isinstance(raw, dict):
            return None
        payload = json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return _sha256_text(payload)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class BuildRunWriter:
    """Persist build run lifecycle data to storage.

    Parameters
    ----------
    gateway
        Storage gateway used for persistence.
    """

    gateway: StorageGateway

    def start_run(
        self,
        *,
        env: BuildEnv,
        run_id: str,
        requested_targets: Sequence[str],
        started_at: datetime,
    ) -> None:
        """Record the start of a build run.

        Parameters
        ----------
        env
            Build environment containing repo/commit identifiers.
        run_id
            Run identifier.
        requested_targets
            Requested targets for the run.
        started_at
            Run start timestamp.
        """
        try:
            record = BuildRunRecord(
                run_id=run_id,
                repo=env.repo,
                commit=env.commit,
                requested_targets=tuple(requested_targets),
                computed_targets=(),
                skipped_targets=(),
                started_at=started_at,
                status="running",
            )
            self.gateway.build.start_run(record)
        except StorageError as exc:
            log.warning("build.hamilton.writer.start_run_failed run_id=%s error=%s", run_id, exc)

        try:
            self.gateway.assets.record_run_environment(
                RunEnvironmentRecord(
                    run_id=run_id,
                    python_version=platform.python_version(),
                    os_name=platform.system(),
                    os_version=platform.release(),
                    tool_versions=_tool_versions(),
                    config_hash=_config_hash(env),
                    git_dirty=False,
                    captured_at=started_at,
                )
            )
        except StorageError as exc:
            log.warning(
                "build.hamilton.writer.run_environment_failed run_id=%s error=%s",
                run_id,
                exc,
            )

    def complete_run(
        self,
        *,
        run_id: str,
        success: bool,
        computed_targets: Sequence[str],
        skipped_targets: Sequence[str],
        error_summary: str | None,
    ) -> None:
        """Complete the build run record.

        Parameters
        ----------
        run_id
            Run identifier.
        success
            Whether the run succeeded.
        computed_targets
            Targets that were computed.
        skipped_targets
            Targets that were skipped.
        error_summary
            Optional error summary if failed.
        """
        try:
            status = "succeeded" if success else "failed"
            self.gateway.build.complete_run(
                run_id=run_id,
                status=status,
                computed_targets=tuple(computed_targets),
                skipped_targets=tuple(skipped_targets),
                error_summary=error_summary,
            )
        except StorageError as exc:
            log.warning("build.hamilton.writer.complete_run_failed run_id=%s error=%s", run_id, exc)

    def save_run_targets(
        self,
        *,
        env: BuildEnv,
        run_id: str,
        records: Sequence[TargetRunRecord],
    ) -> None:
        """Persist per-target execution records for a run.

        Parameters
        ----------
        env
            Build environment containing repo/commit identifiers.
        run_id
            Run identifier.
        records
            Target run records to persist.
        """
        if not records:
            return
        try:
            sorted_records = sorted(records, key=lambda record: record.target)
            self.gateway.build.save_run_targets(
                run_id=run_id,
                repo=env.repo,
                commit=env.commit,
                records=sorted_records,
            )
        except StorageError as exc:
            log.warning("build.hamilton.writer.run_targets_failed run_id=%s error=%s", run_id, exc)

    def save_run_nodes(
        self,
        run_id: str,
        records: Sequence[NodeExecutionRecord],
    ) -> int:
        """Persist node-level execution telemetry for a run.

        Parameters
        ----------
        run_id
            Run identifier.
        records
            Node execution records to persist.

        Returns
        -------
        int
            Number of records persisted.
        """
        try:
            return self.gateway.build.save_run_nodes(run_id, records)
        except StorageError as exc:
            log.warning("build.hamilton.writer.run_nodes_failed run_id=%s error=%s", run_id, exc)
            return 0

    def persist_asset_catalog(
        self,
        *,
        env: BuildEnv,
        run_id: str,
        graph: TargetGraph,
        records: Sequence[TargetRunRecord],
    ) -> None:
        """Emit Phase 4 asset catalog records for a run.

        Parameters
        ----------
        env
            Build environment containing gateway access and snapshot metadata.
        run_id
            Run identifier.
        graph
            Target graph for resolving contracts/dependencies.
        records
            Target run records to emit as assets.
        """
        if not records:
            return
        try:
            sorted_records = sorted(records, key=lambda record: record.target)
            env_with_gateway = replace(env, gateway=self.gateway)
            persist_asset_catalog_for_run(
                env=env_with_gateway,
                run_id=run_id,
                graph=graph,
                records=sorted_records,
            )
        except StorageError as exc:
            log.warning("build.hamilton.writer.asset_catalog_failed run_id=%s error=%s", run_id, exc)
