"""Manifest and skip logic for Hamilton node execution.

This module provides the skip/cache infrastructure for Hamilton nodes,
reusing the existing manifest tables and hashing functions from the
build system. It tracks execution results and persists manifests.

Design Principles
-----------------
1. Reuse existing compute_input_hash() from codeintel.build.hashing.
2. Reuse existing OutputManifest and BuildTracking from build system.
3. TargetRunRecord captures execution state for Hamilton observability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.hashing import compute_input_hash, compute_options_hash
from codeintel.build.manifest import OutputManifest

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.hamilton.io.dataset_ref import DatasetRef
    from codeintel.build.targets import OutputTarget
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetRunRecord:
    """Record of a Hamilton node execution for a target.

    This captures the execution state of a target run, including timing,
    row counts, and error information. Used for observability and as
    the return type from Hamilton node functions.

    Attributes
    ----------
    target
        Name of the target that was executed.
    plugin_name
        Name of the plugin that produced this target.
    status
        Execution status: "succeeded", "failed", or "skipped".
    input_hash
        Content-addressable hash of inputs (for cache validation).
    options_hash
        Hash of plugin configuration options, if any.
    duration_ms
        Execution duration in milliseconds.
    row_counts
        Mapping of table keys to row counts written.
    error
        Error message if execution failed.
    datasets
        Tuple of DatasetRef instances produced by this target (Phase 1).

    Examples
    --------
    >>> record = TargetRunRecord(
    ...     target="function_metrics",
    ...     plugin_name="analytics.function_metrics",
    ...     status="succeeded",
    ...     input_hash="a1b2c3d4",
    ...     duration_ms=1234.5,
    ...     row_counts={"analytics.function_metrics": 1500},
    ... )
    """

    target: str
    plugin_name: str
    status: str
    input_hash: str | None
    options_hash: str | None = None
    duration_ms: float = 0.0
    row_counts: Mapping[str, int] = field(default_factory=dict)
    error: str | None = None
    datasets: tuple[DatasetRef, ...] = ()

    @property
    def success(self) -> bool:
        """Return True if execution succeeded.

        Returns
        -------
        bool
            True if status is "succeeded".
        """
        return self.status == "succeeded"

    @property
    def skipped(self) -> bool:
        """Return True if execution was skipped.

        Returns
        -------
        bool
            True if status is "skipped".
        """
        return self.status == "skipped"

    def get_dataset(self, table_key: str) -> DatasetRef | None:
        """Get a specific dataset ref by table key.

        Parameters
        ----------
        table_key
            Fully-qualified table name to find.

        Returns
        -------
        DatasetRef | None
            The matching DatasetRef, or None if not found.

        Examples
        --------
        >>> record = TargetRunRecord(
        ...     target="test",
        ...     plugin_name="test.plugin",
        ...     status="succeeded",
        ...     input_hash="abc123",
        ...     datasets=(DatasetRef(table_key="test.table"),),
        ... )
        >>> ds = record.get_dataset("test.table")
        >>> ds is not None
        True
        """
        for ds in self.datasets:
            if ds.table_key == table_key:
                return ds
        return None


def compute_target_input_hash(
    *,
    target: OutputTarget,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
    options_hash: str | None = None,
) -> str:
    """Compute input hash for a target using existing infrastructure.

    Delegates to the existing compute_input_hash() function from the
    build system hashing module.

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

    Returns
    -------
    str
        16-character hex hash string.
    """
    return compute_input_hash(target, snapshot, gateway, options_hash)


def compute_target_options_hash(options: object | None) -> str | None:
    """Compute hash of plugin configuration options.

    Delegates to the existing compute_options_hash() function.

    Parameters
    ----------
    options
        Plugin options object (must be JSON-serializable).

    Returns
    -------
    str | None
        16-character hex hash, or None if no options.
    """
    return compute_options_hash(options)


def should_skip(
    *,
    gateway: StorageGateway,
    target: str,
    repo: str,
    commit: str,
    input_hash: str,
) -> bool:
    """Check if a target can be skipped based on existing manifest.

    Looks up the prior manifest for this target/repo/commit and compares
    the input hash. If they match, the target output is still valid.

    Parameters
    ----------
    gateway
        Storage gateway for manifest access.
    target
        Target name to check.
    repo
        Repository slug.
    commit
        Commit SHA.
    input_hash
        Current input hash to compare.

    Returns
    -------
    bool
        True if the target can be skipped (output is still valid).

    Examples
    --------
    >>> if should_skip(
    ...     gateway=gateway,
    ...     target="function_metrics",
    ...     repo="my-org/my-repo",
    ...     commit="abc123",
    ...     input_hash="a1b2c3d4",
    ... ):
    ...     print("Skipping - output is still valid")
    """
    prior = gateway.build.load_manifest(target=target, repo=repo, commit=commit)
    if prior is None:
        return False
    return prior.input_hash == input_hash


@dataclass(frozen=True)
class ManifestSaveRequest:
    """Parameters for saving a manifest record.

    Attributes
    ----------
    target
        Target name that was computed.
    repo
        Repository slug.
    commit
        Commit SHA.
    plugin
        Plugin name that produced the target.
    duration_ms
        Execution duration in milliseconds.
    input_hash
        Content-addressable hash of inputs.
    row_count
        Optional total row count across all tables.
    options_hash
        Optional hash of plugin options.
    """

    target: str
    repo: str
    commit: str
    plugin: str
    duration_ms: float
    input_hash: str
    row_count: int | None = None
    options_hash: str | None = None


def save_manifest(*, gateway: StorageGateway, request: ManifestSaveRequest) -> None:
    """Persist a manifest record for a completed target.

    Creates an OutputManifest and saves it via the gateway's build tracking.

    Parameters
    ----------
    gateway
        Storage gateway for manifest persistence.
    request
        Manifest save request containing all required fields.
    """
    manifest = OutputManifest(
        target=request.target,
        repo=request.repo,
        commit=request.commit,
        plugin=request.plugin,
        computed_at=datetime.now(tz=UTC),
        duration_ms=request.duration_ms,
        input_hash=request.input_hash,
        row_count=request.row_count,
        options_hash=request.options_hash,
    )
    gateway.build.save_manifest(manifest)
    log.debug(
        "build.hamilton.manifest.saved target=%s input_hash=%s",
        request.target,
        request.input_hash,
    )


__all__ = [
    "ManifestSaveRequest",
    "TargetRunRecord",
    "compute_target_input_hash",
    "compute_target_options_hash",
    "save_manifest",
    "should_skip",
]
