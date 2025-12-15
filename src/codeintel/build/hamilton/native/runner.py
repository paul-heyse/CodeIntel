"""Native target runner for Hamilton Phase 3.

This module provides execution utilities for native Hamilton targets,
including skip checks, manifest persistence, and TargetRunRecord creation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Self

from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.manifest_hook import (
    SkipCheckRequest,
    TargetRunRecord,
    should_skip,
)
from codeintel.build.hamilton.native.outputs import expected_artifacts, expected_datasets
from codeintel.build.manifest import OutputManifest

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv
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


@dataclass
class RunRecordBuilder:
    """Builder for TargetRunRecord instances.

    Provides a fluent interface for constructing run records, avoiding the need
    for functions with many parameters. Use the status-specific class methods
    to start building.

    Attributes
    ----------
    target
        Target that was executed.
    status
        Completion status: succeeded, skipped, or failed.
    input_hash
        Input hash for this execution.

    Examples
    --------
    >>> record = (
    ...     RunRecordBuilder.for_success(target, input_hash).with_env(env).with_run(run).build()
    ... )
    >>> record.status
    'succeeded'

    >>> record = RunRecordBuilder.for_failure(target, input_hash).with_error(exc).build()
    >>> record.status
    'failed'
    """

    target: OutputTarget
    status: Literal["succeeded", "skipped", "failed"]
    input_hash: str
    _env: BuildEnv | None = None
    _run: NativeRunInfo | None = None
    _error: Exception | None = None

    @classmethod
    def for_success(cls, target: OutputTarget, input_hash: str) -> Self:
        """Create a builder for a successful run.

        Parameters
        ----------
        target
            Target that was executed.
        input_hash
            Input hash for this execution.

        Returns
        -------
        Self
            Builder instance configured for success status.
        """
        return cls(target=target, status="succeeded", input_hash=input_hash)

    @classmethod
    def for_skipped(cls, target: OutputTarget, input_hash: str) -> Self:
        """Create a builder for a skipped run.

        Parameters
        ----------
        target
            Target that was checked but skipped.
        input_hash
            Input hash for this execution.

        Returns
        -------
        Self
            Builder instance configured for skipped status.
        """
        return cls(target=target, status="skipped", input_hash=input_hash)

    @classmethod
    def for_failure(cls, target: OutputTarget, input_hash: str) -> Self:
        """Create a builder for a failed run.

        Parameters
        ----------
        target
            Target that failed.
        input_hash
            Input hash for this execution.

        Returns
        -------
        Self
            Builder instance configured for failed status.
        """
        return cls(target=target, status="failed", input_hash=input_hash)

    def with_env(self, env: BuildEnv) -> Self:
        """Set the build environment.

        Required for succeeded and skipped statuses.

        Parameters
        ----------
        env
            Build environment with gateway and paths.

        Returns
        -------
        Self
            This builder instance for chaining.
        """
        self._env = env
        return self

    def with_run(self, run: NativeRunInfo) -> Self:
        """Set the run metadata.

        Required for succeeded and skipped statuses.

        Parameters
        ----------
        run
            Run metadata with timing and hashes.

        Returns
        -------
        Self
            This builder instance for chaining.
        """
        self._run = run
        return self

    def with_error(self, error: Exception) -> Self:
        """Set the exception that caused failure.

        Required for failed status.

        Parameters
        ----------
        error
            The exception that caused the failure.

        Returns
        -------
        Self
            This builder instance for chaining.
        """
        self._error = error
        return self

    def build(self) -> TargetRunRecord:
        """Build the TargetRunRecord.

        Returns
        -------
        TargetRunRecord
            The constructed run record.

        Raises
        ------
        ValueError
            If required fields for the given status are missing.
        """
        return create_run_record(
            self.target,
            self.status,
            self.input_hash,
            env=self._env,
            run=self._run,
            error=self._error,
        )


def create_run_record(
    target: OutputTarget,
    status: Literal["succeeded", "skipped", "failed"],
    input_hash: str,
    *,
    env: BuildEnv | None = None,
    run: NativeRunInfo | None = None,
    error: Exception | None = None,
) -> TargetRunRecord:
    """Create a TargetRunRecord for any completion status.

    This is the unified factory function for creating run records. Use this
    instead of the status-specific functions (create_success_record,
    create_skipped_record, create_failed_record) for new code.

    Parameters
    ----------
    target
        Target that was executed.
    status
        Completion status: succeeded, skipped, or failed.
    input_hash
        Input hash for this execution.
    env
        Build environment (required for succeeded/skipped).
    run
        Run metadata (required for succeeded/skipped).
    error
        Exception that caused failure (required for failed).

    Returns
    -------
    TargetRunRecord
        Record with appropriate datasets/artifacts based on status.

    Raises
    ------
    ValueError
        If required parameters for the given status are not provided.

    Examples
    --------
    >>> # Success record
    >>> record = create_run_record(target, "succeeded", run.input_hash, env=env, run=run)
    >>> record.status
    'succeeded'

    >>> # Skipped record
    >>> record = create_run_record(target, "skipped", run.input_hash, env=env, run=run)
    >>> record.status
    'skipped'

    >>> # Failed record
    >>> record = create_run_record(target, "failed", input_hash, error=error)
    >>> record.status
    'failed'
    """
    plugin_name = f"native:{target.name}"

    if status == "failed":
        return TargetRunRecord(
            target=target.name,
            plugin_name=plugin_name,
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

    # Generate expected refs from contract
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

    # Update row counts for success
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

    return TargetRunRecord(
        target=target.name,
        plugin_name=plugin_name,
        status=status,
        input_hash=input_hash,
        options_hash=run.options_hash,
        duration_ms=run.duration_ms,
        row_counts=run.row_counts or {},
        error=None,
        datasets=datasets,
        artifacts=artifacts,
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
    "RunRecordBuilder",
    "create_run_record",
    "save_manifest",
    "should_skip_native_target",
]
