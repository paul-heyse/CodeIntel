"""Helpers for working with build manifests in tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests._helpers.assertions import assert_target_ok, expect_true
from tests._helpers.hamilton_manifest_priming import ManifestPriming

if TYPE_CHECKING:
    from codeintel.build.hamilton.run_records import TargetRunRecord
    from codeintel.core.build_manifest import OutputManifest
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness


def load_manifest_index(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, OutputManifest]:
    """Load manifests into a target->manifest mapping.

    Returns
    -------
    dict[str, OutputManifest]
        Mapping of target name to manifest.
    """
    manifests = gateway.build.list_manifests(repo=repo, commit=commit)
    return {manifest.target: manifest for manifest in manifests}


def assert_succeeded(record: TargetRunRecord) -> None:
    """Assert that a TargetRunRecord succeeded."""
    assert_target_ok(record)
    expect_true(record.success, message="Expected record.success to be True.")


def prime_manifest(
    harness: HamiltonBuildHarness,
    *,
    spec: ManifestPriming.ManifestSpec,
) -> OutputManifest:
    """Insert a manifest for a target using the manifest priming helper.

    Returns
    -------
    OutputManifest
        Saved manifest record.
    """
    priming = ManifestPriming(harness)
    return priming.prime_manifest(spec)


def prime_modules_manifest(
    harness: HamiltonBuildHarness,
    *,
    file_state_hash: str,
    row_count: int | None = None,
    change_delta: dict[str, object] | None = None,
) -> OutputManifest:
    """Prime the modules manifest using the provided file state hash.

    Returns
    -------
    OutputManifest
        Saved manifest record for modules.
    """
    priming = ManifestPriming(harness)
    return priming.prime_modules_manifest(
        file_state_hash=file_state_hash,
        row_count=row_count,
        change_delta=change_delta,
    )


__all__ = [
    "assert_succeeded",
    "load_manifest_index",
    "prime_manifest",
    "prime_modules_manifest",
]
