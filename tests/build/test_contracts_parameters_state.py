"""Tests for contracts, parameters, and state validation helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.build.contracts import ArtifactSpec, OutputContract, TableSchema
from codeintel.build.hashing import compute_input_hash
from codeintel.build.parameters import ParameterError, TargetParameters
from codeintel.build.state import BuildState, StateValidationOptions, StateValidator, TargetState
from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.config.datasets.primitives import Column
from codeintel.config.primitives import SnapshotRef
from codeintel.core.config.settings import BuildSettings, ExportAuditSettings
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_is_none,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.build import ManifestParams, sample_manifest
from tests._helpers.contracts import contract_for_keys

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.storage.gateway import StorageGateway

TEST_BUILD_SETTINGS = BuildSettings(
    engine_version="test",
    export_audit=ExportAuditSettings(),
)


class FakeBuildStore:
    """Fake build accessors for manifests."""

    def __init__(self, manifests: Mapping[str, object]) -> None:
        """Store manifest mapping."""
        self._manifests = manifests

    def load_manifest(self, target: str, repo: str, commit: str) -> object | None:
        """Return manifest for target if present.

        Returns
        -------
        object | None
            Stored manifest or None if absent.
        """
        _ = (repo, commit)
        return self._manifests.get(target)

    def list_manifests(self, repo: str, commit: str) -> list[object]:
        """List all manifests for repo/commit.

        Returns
        -------
        list[object]
            Manifests stored in this fake.
        """
        _ = (repo, commit)
        return list(self._manifests.values())


def _make_gateway(manifests: Mapping[str, object]) -> StorageGateway:
    """Create gateway with fake build store.

    Returns
    -------
    StorageGateway
        Gateway exposing a build attribute.
    """
    return cast("StorageGateway", SimpleNamespace(build=FakeBuildStore(manifests)))


def _make_target(name: str, dependencies: tuple[str, ...] = ()) -> OutputTarget:
    """Create a minimal OutputTarget for tests.

    Returns
    -------
    OutputTarget
        Target with provided dependencies.
    """
    return OutputTarget(
        name=name,
        module="analytics",
        contract=contract_for_keys((f"core.{name}",)),
        dependencies=dependencies,
        description=f"{name} target",
    )


def _snapshot() -> SnapshotRef:
    """Create a snapshot reference for tests.

    Returns
    -------
    SnapshotRef
        Snapshot with deterministic repo and commit.
    """
    repo_root = Path("repo-root")
    return SnapshotRef(repo="r", commit="c", repo_root=repo_root)


def test_output_contract_accessors_and_validation() -> None:
    """OutputContract exposes keys, lookups, and detects duplicates."""
    table = TableSchema(schema="core", name="items", columns=[Column("id", "INTEGER")])
    artifact = ArtifactSpec(name="index", path_template="{build_dir}/index.scip")
    contract = OutputContract(tables=(table,), artifacts=(artifact,))

    expect_equal(contract.table_keys, ("core.items",))
    expect_equal(contract.artifact_names, ("index",))
    expect_true(contract.get_table("core.items") is table)
    expect_is_none(contract.get_table("core.missing"))
    expect_true(contract.get_artifact("index") is artifact)
    expect_is_none(contract.get_artifact("missing"))
    expect_equal(contract.validate(), [])

    duplicate = OutputContract(
        tables=(table, table),
        artifacts=(artifact, ArtifactSpec(name="index", path_template="path")),
    )
    errors = duplicate.validate()
    expect_true("Duplicate table key" in errors[0])
    expect_true(any("Duplicate artifact name" in err for err in errors))

    empty_columns = OutputContract(tables=(TableSchema(schema="core", name="empty", columns=[]),))
    empty_errors = empty_columns.validate()
    expect_true(any("has no columns" in err for err in empty_errors))


def test_target_parameters_access_and_merge() -> None:
    """TargetParameters enforces types and supports merging."""
    params = TargetParameters({"count": 5, "name": "test"})
    expect_equal(params.get_typed("count", int), 5)
    expect_equal(params.get_typed("name", str), "test")
    expect_equal(params.get_typed("missing", str, default="fallback"), "fallback")
    expect_is_none(params.get_optional("missing", int))
    expect_true(params.has("count"))

    other = TargetParameters({"count": 10, "flag": True})
    merged = params.merge(other)
    expect_equal(merged.get_typed("count", int), 10)
    expect_true(merged.get_typed("flag", bool))
    expect_equal(params.get_typed("count", int), 5)

    with pytest.raises(ParameterError):
        params.get_typed("missing", int)
    with pytest.raises(ParameterError):
        params.get_typed("name", int)
    with pytest.raises(ParameterError):
        params.get_optional("count", str)


def test_target_parameters_empty_singleton() -> None:
    """EMPTY_PARAMETERS creates an empty parameter set."""
    empty = TargetParameters.empty()
    expect_equal(empty.key_set(), frozenset())
    expect_false(empty.has("anything"))


def test_state_validator_missing_and_current() -> None:
    """StateValidator marks missing targets and current when hashes match."""
    target = _make_target("single")
    snapshot = _snapshot()

    current_hash = compute_input_hash(
        target,
        snapshot,
        _make_gateway({}),
        settings=TEST_BUILD_SETTINGS,
    )
    manifest = sample_manifest(
        target="single",
        params=ManifestParams(input_hash=current_hash),
    )
    gateway = _make_gateway({"single": manifest})
    graph = TargetGraph()
    graph.register(target)
    validator = StateValidator(
        graph,
        gateway,
        snapshot,
        options=StateValidationOptions(settings=TEST_BUILD_SETTINGS),
    )

    state = validator.validate().get("single")
    expect_equal(state.status, "current")
    expect_true(state.manifest is manifest)
    expect_equal(state.blocking_deps, ())
    expect_is_none(state.blocking_reason)

    missing_validator = StateValidator(
        graph,
        _make_gateway({}),
        snapshot,
        options=StateValidationOptions(settings=TEST_BUILD_SETTINGS),
    )
    missing_state = missing_validator.validate().get("single")
    expect_equal(missing_state.status, "missing")
    expect_is_none(missing_state.blocking_reason)


def test_state_validator_stale_and_blocked_propagation() -> None:
    """Staleness triggers cascade and dependency blocking."""
    root = _make_target("root")
    leaf = _make_target("leaf", dependencies=("root",))
    snapshot = _snapshot()

    root_manifest = replace(sample_manifest(target="root"), input_hash="old")
    manifests = {"root": root_manifest, "leaf": sample_manifest("leaf")}
    graph = TargetGraph()
    graph.register(root)
    graph.register(leaf)
    validator = StateValidator(
        graph,
        _make_gateway(manifests),
        snapshot,
        options=StateValidationOptions(settings=TEST_BUILD_SETTINGS),
    )

    result = validator.validate()
    root_state = result.get("root")
    leaf_state = result.get("leaf")

    expect_equal(root_state.status, "stale")
    expect_is_not_none(root_state.blocking_reason)
    expect_equal(leaf_state.status, "blocked")
    expect_equal(leaf_state.blocking_deps, ("root",))
    expect_is_not_none(leaf_state.blocking_reason)


def test_build_state_helpers() -> None:
    """BuildState helper methods filter by status and check currentness."""
    states: dict[str, TargetState] = {
        "missing": TargetState(
            name="missing",
            status="missing",
            manifest=None,
        ),
        "stale": TargetState(
            name="stale",
            status="stale",
            manifest=None,
            blocking_reason="input_hash_mismatch",
        ),
        "current": TargetState(
            name="current",
            status="current",
            manifest=None,
            current_hash="hash",
        ),
    }
    build_state = BuildState(repo="r", commit="c", targets=states)

    expect_equal(build_state.by_status("missing"), ("missing",))
    expect_equal(build_state.by_status("stale"), ("stale",))
    expect_equal(build_state.by_status("current"), ("current",))
    expect_equal(build_state.by_status("blocked"), ())
    expect_true(build_state.is_current("current"))
    expect_false(build_state.is_current("absent"))

    with pytest.raises(KeyError):
        build_state.get("absent")
