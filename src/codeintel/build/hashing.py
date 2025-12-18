"""Input hash computation for build system cache invalidation.

This module provides functions to compute content-addressable hashes
of a target's inputs, enabling cache invalidation when dependencies
or configuration change.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from codeintel.build.engine_version import get_build_engine_version

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.targets import OutputTarget
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.build_manifest import OutputManifest
    from codeintel.storage.gateway import StorageGateway


def compute_input_hash(
    target: OutputTarget,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
    options_hash: str | None = None,
    *,
    manifests: Mapping[str, OutputManifest] | None = None,
) -> str:
    """Compute content-addressable hash of a target's inputs.

    The input hash combines:
    - Repository and commit identifiers
    - Dependency input hashes (from their manifests) - cascades correctly
    - Plugin options hash (if provided)

    This enables cache invalidation: if the input hash matches a stored
    manifest, the target output is still valid.

    Parameters
    ----------
    target
        Target to compute hash for.
    snapshot
        Repository snapshot reference (provides repo/commit).
    gateway
        Storage gateway for loading dependency manifests.
    options_hash
        Optional hash of plugin configuration options.
    manifests
        Optional pre-loaded mapping of target names to manifests.
        If provided, avoids per-dependency DB round trips.

    Returns
    -------
    str
        16-character hex hash string.

    Examples
    --------
    >>> hash_value = compute_input_hash(target, snapshot, gateway)
    >>> len(hash_value)
    16
    """
    input_hash, _ = compute_input_hash_with_deps(
        target=target,
        snapshot=snapshot,
        gateway=gateway,
        options_hash=options_hash,
        manifests=manifests,
    )
    return input_hash


def compute_input_hash_with_deps(
    target: OutputTarget,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
    options_hash: str | None = None,
    *,
    manifests: Mapping[str, OutputManifest] | None = None,
) -> tuple[str, dict[str, str]]:
    """Compute input hash and return dependency hash mapping.

    Extended version of compute_input_hash that also returns the individual
    dependency hashes used in the computation. This enables "explain staleness"
    debugging by comparing current vs prior dependency hashes.

    Parameters
    ----------
    target
        Target to compute hash for.
    snapshot
        Repository snapshot reference (provides repo/commit).
    gateway
        Storage gateway for loading dependency manifests.
    options_hash
        Optional hash of plugin configuration options.
    manifests
        Optional pre-loaded mapping of target names to manifests.
        If provided, avoids per-dependency DB round trips.

    Returns
    -------
    tuple[str, dict[str, str]]
        Tuple of (input_hash, dep_hashes) where dep_hashes maps dependency
        names to their input hashes (or "MISSING" sentinel).

    Examples
    --------
    >>> hash_value, dep_hashes = compute_input_hash_with_deps(...)
    >>> len(hash_value)
    16
    >>> dep_hashes
    {'ast': 'abc123...', 'goids': 'def456...'}
    """
    hasher = hashlib.sha256()

    hasher.update(get_build_engine_version().encode("utf-8"))
    hasher.update(b"|")

    hasher.update(snapshot.repo.encode("utf-8"))
    hasher.update(b":")
    hasher.update(snapshot.commit.encode("utf-8"))
    hasher.update(b"|")

    hasher.update(target.name.encode("utf-8"))
    hasher.update(b"|")

    dep_hash_list: list[str] = []
    dep_hashes: dict[str, str] = {}
    for dep_name in sorted(target.dependencies):
        if manifests is not None:
            manifest = manifests.get(dep_name)
        else:
            manifest = gateway.build.load_manifest(
                target=dep_name,
                repo=snapshot.repo,
                commit=snapshot.commit,
            )

        if manifest is not None and manifest.input_hash is not None:
            dep_hash_list.append(f"{dep_name}:{manifest.input_hash}")
            dep_hashes[dep_name] = manifest.input_hash
        else:
            dep_hash_list.append(f"{dep_name}:MISSING")
            dep_hashes[dep_name] = "MISSING"

    hasher.update(",".join(dep_hash_list).encode("utf-8"))
    hasher.update(b"|")

    if options_hash is not None:
        hasher.update(options_hash.encode("utf-8"))

    return hasher.hexdigest()[:16], dep_hashes


def compute_options_hash(options: object | None) -> str | None:
    """Compute hash of plugin configuration options.

    Serializes the options to JSON and hashes the result. This allows
    detecting when plugin configuration has changed.

    Parameters
    ----------
    options
        Plugin options object (must be JSON-serializable).
        Returns None if options is None.

    Returns
    -------
    str | None
        16-character hex hash string, or None if no options.

    Examples
    --------
    >>> compute_options_hash({"threshold": 0.5})
    '7b226e616d65223a...'
    >>> compute_options_hash(None) is None
    True
    """
    if options is None:
        return None

    try:
        serialized = json.dumps(options, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(options)

    hasher = hashlib.sha256()
    hasher.update(serialized.encode("utf-8"))
    return hasher.hexdigest()[:16]


__all__ = [
    "compute_input_hash",
    "compute_input_hash_with_deps",
    "compute_options_hash",
]
