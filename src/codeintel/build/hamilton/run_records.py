"""Run record and manifest utilities for Hamilton build execution.

This module is the single source of truth for:

- TargetRunRecord construction (succeeded/skipped/failed).
- Manifest persistence for succeeded native targets.

It intentionally consolidates logic that previously lived across:

- ``codeintel.build.hamilton.hooks.manifest_hook``
- ``codeintel.build.hamilton.native.runner``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Self

from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.native.outputs import expected_artifacts, expected_datasets
from codeintel.build.hashing import compute_target_options_hash
from codeintel.core.build_manifest import OutputManifest
from codeintel.core.errors.storage import StorageError
from codeintel.core.hamilton.records import TargetRunRecord
from codeintel.storage.duckdb_types import DuckDBError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.build.hamilton.dag_catalog import DagCatalog, TargetDescriptor
    from codeintel.build.hamilton.env import BuildEnv

log = logging.getLogger(__name__)


def options_hash_for_target(env: BuildEnv, target_name: str) -> str | None:
    """Compute the current configuration options hash for a target.

    Returns
    -------
    str | None
        Hash string for target options, or None when the target has no parameters.
    """
    params = env.config.parameters_for(target_name)
    return compute_target_options_hash(params)


def _load_drift_summaries(
    env: BuildEnv,
    datasets: Sequence[DatasetRef],
) -> dict[str, Mapping[str, object]]:
    if not datasets:
        return {}
    summaries: dict[str, Mapping[str, object]] = {}
    for dataset in datasets:
        try:
            observation = env.gateway.schemas.load_latest_schema_observation(
                table_key=dataset.table_key,
            )
        except (DuckDBError, StorageError, RuntimeError, TypeError, ValueError) as exc:
            log.warning(
                "build.hamilton.run_record.drift_summary_failed table_key=%s error=%s",
                dataset.table_key,
                exc,
            )
            continue
        if observation is None or not observation.drift_summary:
            continue
        summaries[dataset.table_key] = dict(observation.drift_summary)
    return summaries


@dataclass(frozen=True)
class NativeRunInfo:
    """Execution metadata used to create a TargetRunRecord.

    The input_hash value is the cache key for the target inputs.
    """

    input_hash: str | None
    options_hash: str | None
    duration_ms: float
    row_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class RunRecordInputs:
    """Inputs required to build a TargetRunRecord."""

    env: BuildEnv | None = None
    run: NativeRunInfo | None = None
    error: Exception | None = None
    catalog: DagCatalog | None = None


@dataclass
class RunRecordBuilder:
    """Builder for TargetRunRecord instances."""

    target: TargetDescriptor
    status: Literal["succeeded", "skipped", "failed"]
    input_hash: str
    _env: BuildEnv | None = None
    _run: NativeRunInfo | None = None
    _error: Exception | None = None
    _catalog: DagCatalog | None = None

    @classmethod
    def for_success(cls, target: TargetDescriptor, input_hash: str) -> Self:
        """Create a builder for a successful run.

        Returns
        -------
        Self
            Builder instance configured for status="succeeded".
        """
        return cls(target=target, status="succeeded", input_hash=input_hash)

    @classmethod
    def for_skipped(cls, target: TargetDescriptor, input_hash: str) -> Self:
        """Create a builder for a skipped run.

        Returns
        -------
        Self
            Builder instance configured for status="skipped".
        """
        return cls(target=target, status="skipped", input_hash=input_hash)

    @classmethod
    def for_failure(cls, target: TargetDescriptor, input_hash: str) -> Self:
        """Create a builder for a failed run.

        Returns
        -------
        Self
            Builder instance configured for status="failed".
        """
        return cls(target=target, status="failed", input_hash=input_hash)

    def with_env(self, env: BuildEnv) -> Self:
        """Set the build environment (required for succeeded/skipped).

        Returns
        -------
        Self
            This builder instance for chaining.
        """
        self._env = env
        return self

    def with_run(self, run: NativeRunInfo) -> Self:
        """Set the run metadata (required for succeeded/skipped).

        Returns
        -------
        Self
            This builder instance for chaining.
        """
        self._run = run
        return self

    def with_error(self, error: Exception) -> Self:
        """Set the exception that caused failure (required for failed).

        Returns
        -------
        Self
            This builder instance for chaining.
        """
        self._error = error
        return self

    def with_catalog(self, catalog: DagCatalog) -> Self:
        """Set the DAG catalog (required for succeeded/skipped).

        Returns
        -------
        Self
            This builder instance for chaining.
        """
        self._catalog = catalog
        return self

    def build(self) -> TargetRunRecord:
        """Build the TargetRunRecord.

        Returns
        -------
        TargetRunRecord
            Run record for the configured status.

        Raises
        ------
        ValueError
            If required inputs for the configured status are missing.
        """
        if self.status == "failed":
            if self._error is None:
                msg = "error is required for status 'failed'"
                raise ValueError(msg)
        elif self._env is None or self._run is None or self._catalog is None:
            msg = f"env, run, and catalog are required for status '{self.status}'"
            raise ValueError(msg)

        return create_run_record(
            self.target,
            self.status,
            self.input_hash,
            inputs=RunRecordInputs(
                env=self._env,
                run=self._run,
                error=self._error,
                catalog=self._catalog,
            ),
        )


def create_run_record(
    target: TargetDescriptor,
    status: Literal["succeeded", "skipped", "failed"],
    input_hash: str | None,
    *,
    inputs: RunRecordInputs | None = None,
) -> TargetRunRecord:
    """Create a TargetRunRecord for any completion status.

    Parameters
    ----------
    target
        Target that was executed.
    status
        Completion status: succeeded, skipped, or failed.
    input_hash
        Cache key for this execution inputs.
    inputs
        Inputs required for record construction.

    Returns
    -------
    TargetRunRecord
        Record with appropriate datasets/artifacts based on status.

    Raises
    ------
    ValueError
        If required parameters for the given status are not provided.
    """
    resolved_inputs = inputs or RunRecordInputs()
    env = resolved_inputs.env
    run = resolved_inputs.run
    error = resolved_inputs.error
    impl_kind = "native"

    if status == "failed":
        return TargetRunRecord(
            target=target.name,
            impl_kind=impl_kind,
            status="failed",
            input_hash=input_hash,
            options_hash=run.options_hash if run else None,
            duration_ms=run.duration_ms if run else 0.0,
            row_counts={},
            error=str(error) if error else None,
            datasets=(),
            artifacts=(),
        )

    if env is None or run is None:
        msg = f"env and run are required for status '{status}'"
        raise ValueError(msg)

    catalog = resolved_inputs.catalog
    if catalog is None:
        msg = f"catalog is required for status '{status}'"
        raise ValueError(msg)

    datasets = expected_datasets(target, env.snapshot, outputs=catalog)
    artifacts = expected_artifacts(
        target,
        env.snapshot,
        outputs=catalog,
        path_formatter={
            "build_dir": str(env.paths.build_dir),
            "scip_dir": str(env.paths.scip_dir),
            "export_dir": str(env.paths.document_output_dir),
            "repo_root": str(env.snapshot.repo_root),
        },
    )

    if status == "succeeded" and run.row_counts:
        updated_datasets: list[DatasetRef] = []
        for ds in datasets:
            row_count = run.row_counts.get(ds.table_key, ds.row_count)
            updated_datasets.append(
                DatasetRef(
                    table_key=ds.table_key,
                    repo=ds.repo,
                    commit=ds.commit,
                    row_count=row_count,
                )
            )
        datasets = tuple(updated_datasets)

    drift_summaries = _load_drift_summaries(env, datasets)

    return TargetRunRecord(
        target=target.name,
        impl_kind=impl_kind,
        status=status,
        input_hash=input_hash,
        options_hash=run.options_hash,
        duration_ms=run.duration_ms,
        row_counts=run.row_counts or {},
        error=None,
        datasets=datasets,
        artifacts=artifacts,
        drift_summaries=drift_summaries,
    )


def save_manifest(
    env: BuildEnv,
    record: TargetRunRecord,
    *,
    change_delta: Mapping[str, object] | None = None,
) -> None:
    """Persist an audit-only OutputManifest for a completed native target execution.

    Parameters
    ----------
    env
        Build environment with gateway access.
    record
        Target run record to persist as manifest. Input hash is the cache key.
    change_delta
        Optional change-detection delta payload for auditability.
    """
    manifest = OutputManifest(
        target=record.target,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        impl_kind=record.impl_kind,
        computed_at=datetime.now(tz=UTC),
        duration_ms=record.duration_ms,
        input_hash=record.input_hash or "",
        row_count=sum(record.row_counts.values()) if record.row_counts else None,
        options_hash=record.options_hash,
        change_delta=dict(change_delta) if change_delta is not None else None,
    )

    env.gateway.build.save_manifest(manifest)
    log.debug(
        "build.hamilton.manifest.saved target=%s input_hash=%s", record.target, record.input_hash
    )


__all__ = [
    "NativeRunInfo",
    "RunRecordBuilder",
    "RunRecordInputs",
    "TargetRunRecord",
    "compute_target_options_hash",
    "create_run_record",
    "options_hash_for_target",
    "save_manifest",
]
