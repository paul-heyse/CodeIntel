"""Tests for input hashing with catalog-based dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from codeintel.build.contracts import EMPTY_CONTRACT
from codeintel.build.hashing import InputHashOptions, compute_input_hash
from codeintel.core.build_manifest import OutputManifest
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import expect_equal, expect_not_equal
from tests._helpers.build import TEST_BUILD_SETTINGS, make_snapshot
from tests._helpers.catalog import make_target_descriptor


@dataclass(frozen=True)
class _FakeBuildAccessor:
    @staticmethod
    def load_manifest(target: str, repo: str, commit: str) -> OutputManifest | None:
        _ = (target, repo, commit)
        return None


@dataclass(frozen=True)
class _FakeGateway:
    build: _FakeBuildAccessor


def _make_manifest(target: str, input_hash: str) -> OutputManifest:
    return OutputManifest(
        target=target,
        repo="demo",
        commit="c1",
        impl_kind="native",
        computed_at=datetime.now(tz=UTC),
        duration_ms=1.0,
        input_hash=input_hash,
        output_hash=None,
    )


def test_input_hash_changes_with_dependency_manifest(tmp_path: Path) -> None:
    """Input hash should change when dependency manifest input_hash changes."""
    snapshot = make_snapshot(tmp_path, repo="demo", commit="c1")
    target = make_target_descriptor(
        name="main",
        module="analytics",
        contract=EMPTY_CONTRACT,
        dependencies=("dep",),
    )
    gateway = cast("StorageGateway", _FakeGateway(build=_FakeBuildAccessor()))

    dep_manifest_v1 = _make_manifest("dep", "hash_v1")
    dep_manifest_v2 = _make_manifest("dep", "hash_v2")

    hash_options_v1 = InputHashOptions(options_hash="opts", manifests={"dep": dep_manifest_v1})
    hash_options_v1_repeat = InputHashOptions(options_hash="opts", manifests={"dep": dep_manifest_v1})
    hash_options_v2 = InputHashOptions(options_hash="opts", manifests={"dep": dep_manifest_v2})

    hash_v1 = compute_input_hash(
        target,
        snapshot,
        gateway,
        settings=TEST_BUILD_SETTINGS,
        options=hash_options_v1,
    )
    hash_v1_repeat = compute_input_hash(
        target,
        snapshot,
        gateway,
        settings=TEST_BUILD_SETTINGS,
        options=hash_options_v1_repeat,
    )
    hash_v2 = compute_input_hash(
        target,
        snapshot,
        gateway,
        settings=TEST_BUILD_SETTINGS,
        options=hash_options_v2,
    )

    expect_equal(hash_v1, hash_v1_repeat)
    expect_not_equal(hash_v1, hash_v2)
