"""Shared hash evaluation helpers for planning and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from codeintel.build.hashing import InputHashOptions, compute_input_hash_with_deps
from codeintel.core.config.settings import BuildSettings

if TYPE_CHECKING:

    from codeintel.build.targets import OutputTarget
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.build_manifest import OutputManifest
    from codeintel.storage.gateway import StorageGateway

HashStatus = Literal["missing", "current", "stale"]
HashReason = Literal[
    "no_manifest",
    "up_to_date",
    "options_hash_mismatch",
    "input_hash_mismatch",
]


@dataclass(frozen=True, slots=True)
class HashEvaluation:
    """Result of comparing computed hashes with stored manifests."""

    status: HashStatus
    reason: HashReason
    input_hash: str
    options_hash: str | None
    stored_hash: str | None
    dep_hashes: dict[str, str] = field(default_factory=dict)
    prior_dep_hashes: dict[str, str] = field(default_factory=dict)


def evaluate_hash_state(
    *,
    manifest: OutputManifest | None,
    input_hash: str,
    options_hash: str | None,
    dep_hashes: dict[str, str] | None = None,
    prior_dep_hashes: dict[str, str] | None = None,
) -> HashEvaluation:
    """Compare computed hash values to stored manifest state.

    Parameters
    ----------
    manifest
        Stored manifest for the target, if present.
    input_hash
        Current computed input hash.
    options_hash
        Current computed options hash.
    dep_hashes
        Current dependency hash mapping.
    prior_dep_hashes
        Prior dependency hashes from the stored manifest.

    Returns
    -------
    HashEvaluation
        Evaluation result with status and reason details.
    """
    resolved_dep_hashes = {} if dep_hashes is None else dict(dep_hashes)
    resolved_prior = {} if prior_dep_hashes is None else dict(prior_dep_hashes)

    if manifest is None:
        return HashEvaluation(
            status="missing",
            reason="no_manifest",
            input_hash=input_hash,
            options_hash=options_hash,
            stored_hash=None,
            dep_hashes=resolved_dep_hashes,
            prior_dep_hashes=resolved_prior,
        )

    stored_hash = manifest.input_hash
    if stored_hash == input_hash:
        return HashEvaluation(
            status="current",
            reason="up_to_date",
            input_hash=input_hash,
            options_hash=options_hash,
            stored_hash=stored_hash,
            dep_hashes=resolved_dep_hashes,
            prior_dep_hashes=resolved_prior or dict(manifest.dep_hashes or {}),
        )

    reason: HashReason = "input_hash_mismatch"
    if options_hash != manifest.options_hash:
        reason = "options_hash_mismatch"

    return HashEvaluation(
        status="stale",
        reason=reason,
        input_hash=input_hash,
        options_hash=options_hash,
        stored_hash=stored_hash,
        dep_hashes=resolved_dep_hashes,
        prior_dep_hashes=resolved_prior or dict(manifest.dep_hashes or {}),
    )


def compute_hash_evaluation(
    *,
    target: OutputTarget,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
    settings: BuildSettings,
    options: InputHashOptions | None = None,
) -> HashEvaluation:
    """Compute hash evaluation for a target.

    Parameters
    ----------
    target
        Target to evaluate.
    snapshot
        Repository snapshot reference.
    gateway
        Storage gateway for manifest lookup.
    settings
        Build settings for engine version hashing.
    options
        Optional hash options (options hash + manifest cache).

    Returns
    -------
    HashEvaluation
        Evaluation result with dependency hashes.
    """
    resolved_options = options or InputHashOptions()
    options_hash = resolved_options.options_hash
    manifests = resolved_options.manifests

    manifest = None
    if manifests is not None:
        manifest = manifests.get(target.name)
    if manifest is None:
        manifest = gateway.build.load_manifest(
            target=target.name,
            repo=snapshot.repo,
            commit=snapshot.commit,
        )

    input_hash, dep_hashes = compute_input_hash_with_deps(
        target,
        snapshot,
        gateway,
        settings=settings,
        options=resolved_options,
    )

    prior_dep_hashes: dict[str, str] = {}
    if manifest is not None and manifest.dep_hashes:
        prior_dep_hashes = dict(manifest.dep_hashes)

    return evaluate_hash_state(
        manifest=manifest,
        input_hash=input_hash,
        options_hash=options_hash,
        dep_hashes=dep_hashes,
        prior_dep_hashes=prior_dep_hashes,
    )


__all__ = [
    "HashEvaluation",
    "HashReason",
    "HashStatus",
    "compute_hash_evaluation",
    "evaluate_hash_state",
]
