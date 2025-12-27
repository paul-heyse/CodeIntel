"""Run record utilities used across Hamilton execution layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from codeintel.build.hash_evaluator import evaluate_hash_state
from codeintel.build.hashing import compute_target_options_hash
from codeintel.core.build_manifest import OutputManifest

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.hamilton.dag_catalog import TargetDescriptor
    from codeintel.build.hamilton.env import BuildEnv


class BuildManifestService(Protocol):
    """Protocol for manifest access in skip evaluation."""

    def load_manifest(
        self,
        *,
        target: str,
        repo: str,
        commit: str,
    ) -> OutputManifest | None:
        """Load an output manifest for a target snapshot."""
        ...


@dataclass(frozen=True)
class SkipCheckRequest:
    """Input parameters for manifest skip evaluation."""

    manifest_service: BuildManifestService
    target: str
    repo: str
    commit: str
    input_hash: str
    options_hash: str | None = None
    manifest_index: Mapping[str, OutputManifest] | None = None


def _resolve_manifest(request: SkipCheckRequest) -> OutputManifest | None:
    """Resolve a manifest from request inputs.

    Returns
    -------
    OutputManifest | None
        Cached manifest if available, otherwise None.
    """
    if request.manifest_index is not None:
        cached = request.manifest_index.get(request.target)
        if cached is not None:
            return cached
    return request.manifest_service.load_manifest(
        target=request.target,
        repo=request.repo,
        commit=request.commit,
    )


def should_skip(request: SkipCheckRequest) -> bool:
    """Return True if a target can be skipped based on manifest hashes.

    Returns
    -------
    bool
        True if the manifest is current, False otherwise.
    """
    manifest = _resolve_manifest(request)
    evaluation = evaluate_hash_state(
        manifest=manifest,
        input_hash=request.input_hash,
        options_hash=request.options_hash,
    )
    return evaluation.status == "current"


def options_hash_for_target(env: BuildEnv, target_name: str) -> str | None:
    """Compute the current configuration options hash for a target.

    Returns
    -------
    str | None
        Hash string for target options, or None when the target has no parameters.
    """
    params = env.config.parameters_for(target_name)
    return compute_target_options_hash(params)


def should_skip_native_target(
    env: BuildEnv,
    target: TargetDescriptor,
    input_hash: str,
    options_hash: str | None = None,
) -> bool:
    """Return True if a native target can be skipped based on manifest.

    Returns
    -------
    bool
        True when the target is safe to skip, False otherwise.
    """
    if target.name in env.force_targets:
        return False
    resolved_options_hash = options_hash
    if resolved_options_hash is None:
        resolved_options_hash = options_hash_for_target(env, target.name)

    request = SkipCheckRequest(
        manifest_service=env.manifest_service,
        target=target.name,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        input_hash=input_hash,
        options_hash=resolved_options_hash,
        manifest_index=env.manifest_index,
    )
    return should_skip(request)


__all__ = [
    "BuildManifestService",
    "SkipCheckRequest",
    "options_hash_for_target",
    "should_skip",
    "should_skip_native_target",
]
