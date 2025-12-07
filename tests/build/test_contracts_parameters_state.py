"""Tests for contracts, parameters, and state validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from codeintel.build.contracts import ArtifactSpec, OutputContract, TableSchema
from codeintel.build.parameters import ParameterError, TargetParameters
from codeintel.build.state import (
    DatabaseState,
    StalenessReason,
    StateValidator,
    TargetState,
)
from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.config.datasets.primitives import Column
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from tests._helpers.build import sample_manifest


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
        plugin=f"{name}_plugin",
        tables=(f"core.{name}",),
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

    assert contract.table_keys == ("core.items",)
    assert contract.artifact_names == ("index",)
    assert contract.get_table("core.items") is table
    assert contract.get_table("core.missing") is None
    assert contract.get_artifact("index") is artifact
    assert contract.get_artifact("missing") is None
    assert contract.validate() == []

    duplicate = OutputContract(
        tables=(table, table),
        artifacts=(artifact, ArtifactSpec(name="index", path_template="path")),
    )
    errors = duplicate.validate()
    assert "Duplicate table key" in errors[0]
    assert any("Duplicate artifact name" in err for err in errors)

    empty_columns = OutputContract(tables=(TableSchema(schema="core", name="empty", columns=[]),))
    empty_errors = empty_columns.validate()
    assert any("has no columns" in err for err in empty_errors)


def test_target_parameters_access_and_merge() -> None:
    """TargetParameters enforces types and supports merging."""
    params = TargetParameters({"count": 5, "name": "test"})
    assert params.get("count", int) == 5
    assert params.get("name", str) == "test"
    assert params.get("missing", str, default="fallback") == "fallback"
    assert params.get_optional("missing", int) is None
    assert params.has("count") is True

    other = TargetParameters({"count": 10, "flag": True})
    merged = params.merge(other)
    assert merged.get("count", int) == 10
    assert merged.get("flag", bool) is True
    assert params.get("count", int) == 5  # original unchanged

    with pytest.raises(ParameterError):
        params.get("missing", int)
    with pytest.raises(ParameterError):
        params.get("name", int)
    with pytest.raises(ParameterError):
        params.get_optional("count", str)


def test_target_parameters_empty_singleton() -> None:
    """EMPTY_PARAMETERS creates an empty parameter set."""
    empty = TargetParameters.empty()
    assert empty.keys() == frozenset()
    assert empty.has("anything") is False


def test_state_validator_missing_and_computed() -> None:
    """StateValidator marks missing targets and computed when hashes match."""
    target = _make_target("single")
    snapshot = _snapshot()

    # Compute manifest with matching input hash
    manifest = sample_manifest(target="single")
    gateway = _make_gateway({"single": manifest})
    graph = TargetGraph()
    graph.register(target)
    validator = StateValidator(graph, gateway, snapshot)

    state = validator.validate().get("single")
    assert state.status == "computed"
    assert state.manifest is manifest
    assert state.blocking_deps == ()
    assert state.staleness_reason is None

    # Missing manifest path
    missing_validator = StateValidator(graph, _make_gateway({}), snapshot)
    missing_state = missing_validator.validate().get("single")
    assert missing_state.status == "missing"
    assert missing_state.staleness_reason is None


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
    validator = StateValidator(graph, _make_gateway(manifests), snapshot)

    result = validator.validate()
    root_state = result.get("root")
    leaf_state = result.get("leaf")

    assert root_state.status == "stale"
    assert root_state.staleness_reason is not None
    assert leaf_state.status == "blocked"
    assert leaf_state.blocking_deps == ("root",)
    assert leaf_state.staleness_reason is not None


def test_database_state_helpers() -> None:
    """DatabaseState helper methods filter by status and check currentness."""
    states: dict[str, TargetState] = {
        "missing": TargetState(
            name="missing",
            status="missing",
            manifest=None,
            staleness_reason=None,
            blocking_deps=(),
            current_input_hash=None,
        ),
        "stale": TargetState(
            name="stale",
            status="stale",
            manifest=None,
            staleness_reason=StalenessReason(
                kind="input_hash_mismatch",
                details="hash changed",
            ),
            blocking_deps=(),
            current_input_hash=None,
        ),
        "computed": TargetState(
            name="computed",
            status="computed",
            manifest=None,
            staleness_reason=None,
            blocking_deps=(),
            current_input_hash="hash",
        ),
    }
    db_state = DatabaseState(repo="r", commit="c", targets=states)

    assert db_state.missing_targets() == ("missing",)
    assert db_state.stale_targets() == ("stale",)
    assert db_state.computed_targets() == ("computed",)
    assert db_state.blocked_targets() == ()
    assert db_state.is_target_current("computed") is True
    assert db_state.is_target_current("absent") is False

    with pytest.raises(KeyError):
        db_state.get("absent")
