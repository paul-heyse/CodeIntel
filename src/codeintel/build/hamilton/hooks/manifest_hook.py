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
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.hashing import (
    compute_input_hash,
    compute_input_hash_with_deps,
    compute_options_hash,
)
from codeintel.build.manifest import OutputManifest
from codeintel.hamilton.records import TargetRunRecord

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.targets import OutputTarget
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def compute_target_input_hash(
    *,
    target: OutputTarget,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
    options_hash: str | None = None,
    manifests: Mapping[str, OutputManifest] | None = None,
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
    manifests
        Optional pre-loaded manifest index to avoid per-dependency DB calls.

    Returns
    -------
    str
        16-character hex hash string.
    """
    return compute_input_hash(target, snapshot, gateway, options_hash, manifests=manifests)


def compute_target_input_hash_with_deps(
    *,
    target: OutputTarget,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
    options_hash: str | None = None,
    manifests: Mapping[str, OutputManifest] | None = None,
) -> tuple[str, dict[str, str]]:
    """Compute input hash and dependency hash mapping.

    Delegates to compute_input_hash_with_deps() from the build system hashing
    module. Returns both the input hash and the individual dependency hashes
    for staleness explanation.

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
        Tuple of (input_hash, dep_hashes) where dep_hashes maps dependency
        names to their input hashes (or "MISSING" sentinel).
    """
    return compute_input_hash_with_deps(
        target, snapshot, gateway, options_hash, manifests=manifests
    )


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


@dataclass(frozen=True)
class SkipCheckRequest:
    """Input parameters for manifest skip evaluation."""

    gateway: StorageGateway
    target: str
    repo: str
    commit: str
    input_hash: str
    manifest_index: Mapping[str, OutputManifest] | None = None


def should_skip(
    request: SkipCheckRequest,
) -> bool:
    """Check if a target can be skipped based on existing manifest.

    Looks up the prior manifest for this target/repo/commit and compares
    the input hash. If they match, the target output is still valid.

    Parameters
    ----------
    request
        SkipCheckRequest containing gateway, target, repo, commit, input hash,
        and optional manifest index.

    Returns
    -------
    bool
        True if the target can be skipped (output is still valid).

    Examples
    --------
    >>> from codeintel.build.hamilton.hooks.manifest_hook import SkipCheckRequest, should_skip
    >>> request = SkipCheckRequest(
    ...     gateway=gateway,
    ...     target="function_metrics",
    ...     repo="my-org/my-repo",
    ...     commit="abc123",
    ...     input_hash="a1b2c3d4",
    ... )
    >>> if should_skip(request):
    ...     print("Skipping - output is still valid")
    """
    if request.manifest_index is not None:
        prior = request.manifest_index.get(request.target)
    else:
        prior = request.gateway.build.load_manifest(
            target=request.target,
            repo=request.repo,
            commit=request.commit,
        )
    if prior is None:
        return False
    return prior.input_hash == request.input_hash


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
    "SkipCheckRequest",
    "TargetRunRecord",
    "compute_target_input_hash",
    "compute_target_input_hash_with_deps",
    "compute_target_options_hash",
    "save_manifest",
    "should_skip",
]
