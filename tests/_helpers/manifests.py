"""Helpers for working with build manifests in tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.run_records import compute_target_input_hash, options_hash_for_target
from codeintel.build.hashing import InputHashOptions
from tests._helpers.assertions import assert_target_ok
from tests._helpers.assertions.expectation_assertions import expect_true
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


def assert_skipped(record: TargetRunRecord) -> None:
    """Assert that a TargetRunRecord is skipped."""
    assert_target_ok(record, expected_status="skipped")
    expect_true(record.skipped, message="Expected record.skipped to be True.")


def assert_succeeded(record: TargetRunRecord) -> None:
    """Assert that a TargetRunRecord succeeded."""
    assert_target_ok(record)
    expect_true(record.success, message="Expected record.success to be True.")


def run_twice_and_assert_skip(
    harness: HamiltonBuildHarness,
    target: str,
    *,
    force_on_second: bool = False,
) -> tuple[TargetRunRecord, TargetRunRecord]:
    """Run a target twice and assert the second run skips unless forced.

    Returns
    -------
    tuple[TargetRunRecord, TargetRunRecord]
        Records from the first and second run.
    """
    first = harness.run_targets([target])
    first_record = harness.record(target, result=first)
    assert_succeeded(first_record)

    if force_on_second:
        harness.with_force_targets(target)

    second = harness.run_targets([target])
    second_record = harness.record(target, result=second)
    if force_on_second:
        assert_succeeded(second_record)
    else:
        assert_skipped(second_record)
    return first_record, second_record


def compute_input_hash(
    harness: HamiltonBuildHarness,
    target: str,
    *,
    file_state_hash: str | None = None,
    manifests: dict[str, OutputManifest] | None = None,
) -> tuple[str, str | None]:
    """Compute input and options hashes for a target.

    Returns
    -------
    tuple[str, str | None]
        Input hash and options hash for the target.

    Raises
    ------
    RuntimeError
        If the target cannot be found in the target graph.
    """
    env = harness.build_env()
    runtime = build_driver(config={"profile": env.profile})
    target_spec = runtime.graph.get(target)
    if target_spec is None:
        message = f"Target {target!r} not found in graph"
        raise RuntimeError(message)
    options_hash = options_hash_for_target(env, target)
    input_hash = compute_target_input_hash(
        target=target_spec,
        snapshot=env.snapshot,
        gateway=env.gateway,
        settings=env.settings,
        options=InputHashOptions(
            options_hash=options_hash,
            file_state_hash=file_state_hash,
            manifests=manifests,
        ),
    )
    return input_hash, options_hash


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
    "assert_skipped",
    "assert_succeeded",
    "compute_input_hash",
    "load_manifest_index",
    "prime_manifest",
    "prime_modules_manifest",
    "run_twice_and_assert_skip",
]
