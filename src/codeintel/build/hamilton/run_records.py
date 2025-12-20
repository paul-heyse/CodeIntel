"""Run record, skip, and manifest utilities for Hamilton build execution.

This module is the single source of truth for:

- Target input hashing helpers used by planning and execution.
- Manifest-based skip evaluation.
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
from codeintel.build.hash_evaluator import evaluate_hash_state
from codeintel.build.hashing import (
    InputHashOptions,
    compute_input_hash,
    compute_input_hash_with_deps,
    compute_target_options_hash,
)
from codeintel.core.build_manifest import OutputManifest
from codeintel.core.hamilton.records import TargetRunRecord

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.targets import OutputTarget
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def _validate_strict_row_counts(
    *,
    target: OutputTarget,
    row_counts: dict[str, int] | None,
) -> None:
    if not target.table_keys:
        if row_counts:
            msg = (
                "Strict contracts require empty row_counts for artifact-only targets: "
                f"target={target.name} row_count_keys={sorted(row_counts)}"
            )
            raise ValueError(msg)
        return

    if row_counts is None:
        msg = (
            "Strict contracts require row_counts for table-producing targets: "
            f"target={target.name} table_keys={target.table_keys}"
        )
        raise ValueError(msg)

    expected_keys = set(target.table_keys)
    actual_keys = set(row_counts)
    if expected_keys != actual_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        msg = (
            "Strict contracts require row_counts keys to exactly match contract table_keys: "
            f"target={target.name} missing={missing} extra={extra}"
        )
        raise ValueError(msg)

    for table_key, count in row_counts.items():
        if count < 0:
            msg = (
                "Strict contracts require non-negative row counts: "
                f"target={target.name} table_key={table_key} row_count={count}"
            )
            raise ValueError(msg)


def compute_target_input_hash(
    *,
    target: OutputTarget,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
    options_hash: str | None = None,
    manifests: Mapping[str, OutputManifest] | None = None,
) -> str:
    """Compute input hash for a target using the build hashing infrastructure.

    Parameters
    ----------
    target
        Target to compute hash for.
    snapshot
        Repository snapshot reference.
    gateway
        Storage gateway for loading dependency manifests.
    options_hash
        Optional hash of plugin options.
    manifests
        Optional pre-loaded manifest index to avoid per-dependency DB calls.

    Returns
    -------
    str
        16-character hex hash string.
    """
    options = InputHashOptions(options_hash=options_hash, manifests=manifests)
    return compute_input_hash(target, snapshot, gateway, options=options)


def compute_target_input_hash_with_deps(
    *,
    target: OutputTarget,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
    options_hash: str | None = None,
    manifests: Mapping[str, OutputManifest] | None = None,
) -> tuple[str, dict[str, str]]:
    """Compute input hash and dependency hash mapping.

    Parameters
    ----------
    target
        Target to compute hash for.
    snapshot
        Repository snapshot reference.
    gateway
        Storage gateway for loading dependency manifests.
    options_hash
        Optional hash of plugin options.
    manifests
        Optional pre-loaded manifest index to avoid per-dependency DB calls.

    Returns
    -------
    tuple[str, dict[str, str]]
        Tuple of (input_hash, dep_hashes) where dep_hashes maps dependency names
        to their input hashes (or "MISSING" sentinel).
    """
    options = InputHashOptions(options_hash=options_hash, manifests=manifests)
    return compute_input_hash_with_deps(target, snapshot, gateway, options=options)


def options_hash_for_target(env: BuildEnv, target_name: str) -> str | None:
    """Compute the current configuration options hash for a target.

    Parameters
    ----------
    env
        Build environment with configuration.
    target_name
        Target name to compute the options hash for.

    Returns
    -------
    str | None
        16-character options hash, or None when the target has no configuration parameters.
    """
    params = env.config.parameters_for(target_name)
    return compute_target_options_hash(params)


@dataclass(frozen=True)
class SkipCheckRequest:
    """Input parameters for manifest skip evaluation."""

    gateway: StorageGateway
    target: str
    repo: str
    commit: str
    input_hash: str
    options_hash: str | None = None
    manifest_index: Mapping[str, OutputManifest] | None = None


def _resolve_manifest(request: SkipCheckRequest) -> OutputManifest | None:
    """Resolve a manifest from request inputs.

    Parameters
    ----------
    request
        Skip check inputs including optional manifest index.

    Returns
    -------
    OutputManifest | None
        Resolved manifest when available, otherwise None.
    """
    if request.manifest_index is not None:
        manifest = request.manifest_index.get(request.target)
        if manifest is not None:
            return manifest
    return request.gateway.build.load_manifest(
        target=request.target,
        repo=request.repo,
        commit=request.commit,
    )


def should_skip(request: SkipCheckRequest) -> bool:
    """Return True if the target can be skipped based on an existing manifest.

    Parameters
    ----------
    request
        SkipCheckRequest containing gateway, target, repo, commit, input hash, and optional manifest
        index.

    Returns
    -------
    bool
        True if the target can be skipped (output is still valid).
    """
    manifest = _resolve_manifest(request)
    evaluation = evaluate_hash_state(
        manifest=manifest,
        input_hash=request.input_hash,
        options_hash=request.options_hash,
    )
    return evaluation.status == "current"


def should_skip_native_target(
    env: BuildEnv,
    target: OutputTarget,
    input_hash: str,
    options_hash: str | None = None,
) -> bool:
    """Return True if a native target can be skipped based on manifest.

    Parameters
    ----------
    env
        Build environment with gateway and manifest index.
    target
        Target to check for skip eligibility.
    input_hash
        Current computed input hash for the target.
    options_hash
        Optional configuration options hash.

    Returns
    -------
    bool
        True if target can be skipped (manifest matches), False otherwise.
    """
    if target.name in env.force_targets:
        return False
    resolved_options_hash = options_hash
    if resolved_options_hash is None:
        resolved_options_hash = options_hash_for_target(env, target.name)

    request = SkipCheckRequest(
        gateway=env.gateway,
        target=target.name,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        input_hash=input_hash,
        options_hash=resolved_options_hash,
        manifest_index=env.manifest_index,
    )
    return should_skip(request)


@dataclass(frozen=True)
class NativeRunInfo:
    """Execution metadata used to create a TargetRunRecord."""

    input_hash: str
    options_hash: str | None
    duration_ms: float
    row_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class RunRecordInputs:
    """Inputs required to build a TargetRunRecord."""

    env: BuildEnv | None = None
    run: NativeRunInfo | None = None
    error: Exception | None = None


@dataclass
class RunRecordBuilder:
    """Builder for TargetRunRecord instances."""

    target: OutputTarget
    status: Literal["succeeded", "skipped", "failed"]
    input_hash: str
    _env: BuildEnv | None = None
    _run: NativeRunInfo | None = None
    _error: Exception | None = None

    @classmethod
    def for_success(cls, target: OutputTarget, input_hash: str) -> Self:
        """Create a builder for a successful run.

        Returns
        -------
        Self
            Builder instance configured for status="succeeded".
        """
        return cls(target=target, status="succeeded", input_hash=input_hash)

    @classmethod
    def for_skipped(cls, target: OutputTarget, input_hash: str) -> Self:
        """Create a builder for a skipped run.

        Returns
        -------
        Self
            Builder instance configured for status="skipped".
        """
        return cls(target=target, status="skipped", input_hash=input_hash)

    @classmethod
    def for_failure(cls, target: OutputTarget, input_hash: str) -> Self:
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
        elif self._env is None or self._run is None:
            msg = f"env and run are required for status '{self.status}'"
            raise ValueError(msg)

        return create_run_record(
            self.target,
            self.status,
            self.input_hash,
            inputs=RunRecordInputs(env=self._env, run=self._run, error=self._error),
        )


def create_run_record(
    target: OutputTarget,
    status: Literal["succeeded", "skipped", "failed"],
    input_hash: str,
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
        Input hash for this execution.
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

    if env.strict_contracts and status == "succeeded":
        _validate_strict_row_counts(target=target, row_counts=run.row_counts)

    datasets = expected_datasets(target, env.snapshot, output_inventory=env.output_inventory)
    artifacts = expected_artifacts(
        target,
        env.snapshot,
        output_inventory=env.output_inventory,
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
    """Persist an OutputManifest for a completed native target execution.

    Parameters
    ----------
    env
        Build environment with gateway access.
    record
        Target run record to persist as manifest.
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
    log.debug(
        "build.hamilton.manifest.saved target=%s input_hash=%s", record.target, record.input_hash
    )


__all__ = [
    "NativeRunInfo",
    "RunRecordBuilder",
    "RunRecordInputs",
    "SkipCheckRequest",
    "TargetRunRecord",
    "compute_target_input_hash",
    "compute_target_input_hash_with_deps",
    "compute_target_options_hash",
    "create_run_record",
    "options_hash_for_target",
    "save_manifest",
    "should_skip",
    "should_skip_native_target",
]
