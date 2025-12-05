"""Input hash computation for build system cache invalidation.

This module provides functions to compute content-addressable hashes
of a target's inputs, enabling cache invalidation when dependencies
or configuration change.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.build.targets import OutputTarget
    from codeintel.storage.gateway import StorageGateway


def compute_input_hash(
    target: OutputTarget,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
    options_hash: str | None = None,
) -> str:
    """Compute content-addressable hash of a target's inputs.

    The input hash combines:
    - Repository and commit identifiers
    - Dependency output hashes (from their manifests)
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
    hasher = hashlib.sha256()

    # Include repo and commit
    hasher.update(snapshot.repo.encode("utf-8"))
    hasher.update(b":")
    hasher.update(snapshot.commit.encode("utf-8"))
    hasher.update(b"|")

    # Include target name
    hasher.update(target.name.encode("utf-8"))
    hasher.update(b"|")

    # Include dependency hashes (sorted for determinism)
    dep_hashes: list[str] = []
    for dep_name in sorted(target.dependencies):
        # Load manifest for dependency
        manifest = gateway.build.load_manifest(
            target=dep_name,
            repo=snapshot.repo,
            commit=snapshot.commit,
        )
        if manifest is not None and manifest.output_hash is not None:
            dep_hashes.append(f"{dep_name}:{manifest.output_hash}")
        else:
            # Dependency not computed or no output hash - use sentinel
            dep_hashes.append(f"{dep_name}:MISSING")

    hasher.update(",".join(dep_hashes).encode("utf-8"))
    hasher.update(b"|")

    # Include options hash if provided
    if options_hash is not None:
        hasher.update(options_hash.encode("utf-8"))

    # Return first 16 hex characters
    return hasher.hexdigest()[:16]


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

    # Serialize to JSON with sorted keys for determinism
    try:
        serialized = json.dumps(options, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # Fall back to str representation if not JSON-serializable
        serialized = str(options)

    hasher = hashlib.sha256()
    hasher.update(serialized.encode("utf-8"))
    return hasher.hexdigest()[:16]


__all__ = [
    "compute_input_hash",
    "compute_options_hash",
]
