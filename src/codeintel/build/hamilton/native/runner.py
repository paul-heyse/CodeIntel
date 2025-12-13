"""Native target runner for Hamilton Phase 3.

This module provides execution utilities for native Hamilton targets,
including skip checks, manifest persistence, and TargetRunRecord creation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.manifest import OutputManifest
from codeintel.build.hamilton.dataset_ref import DatasetRef
from codeintel.build.hamilton.manifest_hook import (
    SkipCheckRequest,
    TargetRunRecord,
    should_skip,
)
from codeintel.build.hamilton.native.outputs import expected_artifacts, expected_datasets

if TYPE_CHECKING:
    from codeintel.build.env import BuildEnv
    from codeintel.build.targets import OutputTarget


@dataclass(frozen=True)
class NativeRunInfo:
    """Execution metadata used to create a TargetRunRecord.

    Attributes
    ----------
    input_hash
        Computed input hash for this target execution.
    options_hash
        Optional options/config hash for this target.
    duration_ms
        Execution duration in milliseconds.
    row_counts
        Optional row counts per produced table key.
    """

    input_hash: str
    options_hash: str | None
    duration_ms: float
    row_counts: dict[str, int] | None = None


def should_skip_native_target(
    env: BuildEnv,
    target: OutputTarget,
    input_hash: str,
) -> bool:
    """Check if a native target can be skipped based on manifest.

    Parameters
    ----------
    env
        Build environment with gateway and manifest index.
    target
        Target to check for skip eligibility.
    input_hash
        Current computed input hash for the target.

    Returns
    -------
    bool
        True if target can be skipped (manifest matches), False otherwise.

    Examples
    --------
    >>> # Assume env and target are set up
    >>> can_skip = should_skip_native_target(env, target, "abc123")
    >>> can_skip
    False
    """
    # Check forced targets
    if target.name in env.force_targets:
        return False

    # Use skip check with manifest index
    request = SkipCheckRequest(
        gateway=env.gateway,
        target=target.name,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        input_hash=input_hash,
        manifest_index=env.manifest_index,
    )

    return should_skip(request)


def create_success_record(
    target: OutputTarget,
    env: BuildEnv,
    run: NativeRunInfo,
) -> TargetRunRecord:
    """Create a successful TargetRunRecord for a native target.

    Parameters
    ----------
    target
        Target that was executed.
    env
        Build environment with snapshot info.
    run
        Run metadata (hashes, duration, row counts).

    Returns
    -------
    TargetRunRecord
        Record with status="succeeded" and populated datasets/artifacts.

    Examples
    --------
    >>> record = create_success_record(
    ...     target=my_target,
    ...     env=env,
    ...     run=NativeRunInfo(
    ...         input_hash="abc123",
    ...         options_hash="def456",
    ...         duration_ms=1500.0,
    ...         row_counts={"analytics.my_table": 100},
    ...     ),
    ... )
    >>> record.status
    'succeeded'
    """
    # Generate expected refs from contract
    datasets = expected_datasets(target, env.snapshot)

    # Update row counts if available
    if run.row_counts:
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

    artifacts = expected_artifacts(
        target,
        env.snapshot,
        path_formatter={
            "build_dir": str(env.paths.build_dir),
            "scip_dir": str(env.paths.scip_dir),
            "export_dir": str(env.paths.document_output_dir),
            "repo_root": str(env.snapshot.repo_root),
        },
    )

    return TargetRunRecord(
        target=target.name,
        plugin_name=f"native:{target.name}",
        status="succeeded",
        input_hash=run.input_hash,
        options_hash=run.options_hash,
        duration_ms=run.duration_ms,
        row_counts=run.row_counts or {},
        error=None,
        datasets=datasets,
        artifacts=artifacts,
    )


def create_skipped_record(
    target: OutputTarget,
    env: BuildEnv,
    run: NativeRunInfo,
) -> TargetRunRecord:
    """Create a skipped TargetRunRecord for a native target.

    Parameters
    ----------
    target
        Target that was skipped.
    env
        Build environment with snapshot info.
    run
        Run metadata (hashes, duration).

    Returns
    -------
    TargetRunRecord
        Record with status="skipped" and expected datasets/artifacts.

    Examples
    --------
    >>> record = create_skipped_record(
    ...     target=my_target,
    ...     env=env,
    ...     run=NativeRunInfo(input_hash="abc123", options_hash="def456", duration_ms=0.0),
    ... )
    >>> record.status
    'skipped'
    """
    # Generate expected refs (row counts unknown for skipped targets)
    datasets = expected_datasets(target, env.snapshot)
    artifacts = expected_artifacts(
        target,
        env.snapshot,
        path_formatter={
            "build_dir": str(env.paths.build_dir),
            "scip_dir": str(env.paths.scip_dir),
            "export_dir": str(env.paths.document_output_dir),
            "repo_root": str(env.snapshot.repo_root),
        },
    )

    return TargetRunRecord(
        target=target.name,
        plugin_name=f"native:{target.name}",
        status="skipped",
        input_hash=run.input_hash,
        options_hash=run.options_hash,
        duration_ms=run.duration_ms,
        row_counts={},
        error=None,
        datasets=datasets,
        artifacts=artifacts,
    )


def create_failed_record(
    target: OutputTarget,
    input_hash: str,
    options_hash: str | None,
    duration_ms: float,
    error: Exception,
) -> TargetRunRecord:
    """Create a failed TargetRunRecord for a native target.

    Parameters
    ----------
    target
        Target that failed.
    input_hash
        Input hash that was used for this execution.
    options_hash
        Options hash from configuration.
    duration_ms
        Execution duration before failure.
    error
        Exception that caused the failure.

    Returns
    -------
    TargetRunRecord
        Record with status="failed" and error message.

    Examples
    --------
    >>> try:
    ...     raise ValueError("Something went wrong")
    ... except Exception as e:
    ...     record = create_failed_record(
    ...         target=my_target,
    ...         input_hash="abc123",
    ...         options_hash="def456",
    ...         duration_ms=500.0,
    ...         error=e,
    ...     )
    >>> record.status
    'failed'
    """
    return TargetRunRecord(
        target=target.name,
        plugin_name=f"native:{target.name}",
        status="failed",
        input_hash=input_hash,
        options_hash=options_hash,
        duration_ms=duration_ms,
        row_counts={},
        error=str(error),
        datasets=(),
        artifacts=(),
    )


def save_manifest(
    env: BuildEnv,
    record: TargetRunRecord,
) -> None:
    """Persist a manifest for a completed native target execution.

    Parameters
    ----------
    env
        Build environment with gateway access.
    record
        Target run record to persist as manifest.

    Examples
    --------
    >>> save_manifest(env, success_record)
    """
    manifest = OutputManifest(
        target=record.target,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        plugin=record.plugin_name,
        computed_at=datetime.now(tz=UTC),
        duration_ms=record.duration_ms,
        input_hash=record.input_hash or "",
        row_count=sum(record.row_counts.values()) if record.row_counts else None,
        options_hash=record.options_hash,
    )

    env.gateway.build.save_manifest(manifest)


__all__ = [
    "NativeRunInfo",
    "create_failed_record",
    "create_skipped_record",
    "create_success_record",
    "save_manifest",
    "should_skip_native_target",
]
