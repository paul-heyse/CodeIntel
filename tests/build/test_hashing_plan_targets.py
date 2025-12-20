"""Tests for hashing, manifest models, and target graph behaviors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pytest

from codeintel.build.contracts import OutputContract
from codeintel.build.errors import CycleDetectedError
from codeintel.build.hashing import compute_input_hash, compute_options_hash
from codeintel.build.settings import BuildSettings
from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.config.datasets.primitives import Column, TableSchema
from codeintel.core.build_manifest import BuildRunRecord, OutputManifest
from tests._helpers import make_snapshot
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_none,
    expect_is_not_none,
    expect_not_equal,
    expect_true,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class _FakeBuildAccessor:
    manifests: dict[str, OutputManifest]

    def load_manifest(self, target: str, repo: str, commit: str) -> OutputManifest | None:
        _ = (repo, commit)
        return self.manifests.get(target)


@dataclass
class _FakeGateway:
    build: _FakeBuildAccessor


def test_compute_input_hash_differentiates_dependency_hashes(tmp_path: Path) -> None:
    """Input hash uses repo/commit/target/dependency input_hashes deterministically.

    Note: Uses input_hash (not output_hash) for cascade semantics - changes in
    upstream inputs propagate correctly through the dependency chain.
    """
    dep_manifest = OutputManifest(
        target="dep",
        repo="demo",
        commit="c1",
        plugin="p",
        computed_at=datetime.now(tz=UTC),
        duration_ms=1.0,
        input_hash="in",
        output_hash="out-hash",
    )
    gateway = _FakeGateway(build=_FakeBuildAccessor({"dep": dep_manifest}))
    target = OutputTarget(
        name="main",
        module="analytics",
        dependencies=("dep", "missing"),
    )
    snapshot = make_snapshot(tmp_path, repo="demo", commit="c1")
    settings = BuildSettings(
        engine_version="test",
        export_audit_log_path=None,
        export_audit_table_enabled=False,
    )

    hash1 = compute_input_hash(
        target,
        snapshot,
        cast("Any", gateway),
        options_hash="opts",
        settings=settings,
    )
    hash2 = compute_input_hash(
        target,
        snapshot,
        cast("Any", gateway),
        options_hash="opts",
        settings=settings,
    )

    expect_equal(hash1, hash2)

    combined = b"test|demo:c1|main|dep:in,missing:MISSING|opts"
    expect_equal(hash1, hashlib.sha256(combined).hexdigest()[:16])


def test_compute_options_hash_json_and_fallback() -> None:
    """Options hash handles JSON-able and non-JSON objects."""
    options_hash = compute_options_hash({"a": 1, "b": [2, 3]})
    expect_is_not_none(options_hash)
    expect_equal(len(options_hash or ""), 16)

    class Unserializable:
        def __str__(self) -> str:
            return "repr"

    fallback_hash = compute_options_hash(Unserializable())
    expect_is_not_none(fallback_hash)
    expect_not_equal(fallback_hash, options_hash)
    expect_is_none(compute_options_hash(None))


def test_build_run_record_to_dict_handles_none_fields() -> None:
    """BuildRunRecord to_dict preserves None for optional fields."""
    record = BuildRunRecord(
        run_id="run-1",
        repo="demo",
        commit="c1",
        requested_targets=("a",),
        computed_targets=(),
        skipped_targets=(),
        started_at=datetime.now(tz=UTC),
        status="running",
    )
    payload = record.to_dict()
    expect_is_none(payload["completed_at"])
    expect_is_none(payload["error_summary"])


def test_target_graph_validation_and_topology() -> None:
    """TargetGraph validates missing deps, cycles, and ordering."""
    graph = TargetGraph()
    graph.register(OutputTarget(name="a", module="ingestion"))
    graph.register(OutputTarget(name="b", module="graphs", dependencies=("a",)))
    graph.register(OutputTarget(name="c", module="analytics", dependencies=("b", "missing")))

    errors = graph.validate()
    expect_in("missing", errors[0])

    graph_valid = TargetGraph()
    graph_valid.register(OutputTarget(name="a", module="ingestion"))
    graph_valid.register(OutputTarget(name="b", module="graphs", dependencies=("a",)))
    graph_valid.register(OutputTarget(name="c", module="analytics", dependencies=("b",)))
    order = graph_valid.topological_order(("c",))
    expect_equal(order[0], "a")

    cyclic_graph = TargetGraph()
    cyclic_graph.register(OutputTarget(name="cycle1", module="export", dependencies=("cycle2",)))
    cyclic_graph.register(OutputTarget(name="cycle2", module="export", dependencies=("cycle1",)))
    errors_cycle = cyclic_graph.validate()
    expect_true(any("Dependency cycle detected" in err for err in errors_cycle))
    with pytest.raises(CycleDetectedError, match="cycle"):
        cyclic_graph.topological_order(("cycle1",))


def test_target_table_keys_and_execution_duration() -> None:
    """table_keys prefers contract tables, else falls back to legacy tables."""
    contract_target = OutputTarget(
        name="contracted",
        module="ingestion",
        contract=OutputContract(
            tables=(
                TableSchema(
                    schema="core",
                    name="t",
                    columns=[Column("id", "INTEGER", nullable=False)],
                ),
            )
        ),
    )
    legacy_target = OutputTarget(
        name="legacy",
        module="graphs",
        contract=OutputContract.simple(table_keys=("core.legacy",)),
    )
    expect_equal(contract_target.table_keys, ("core.t",))
    expect_equal(legacy_target.table_keys, ("core.legacy",))
    expect_true(legacy_target.estimated_duration_ms >= 0)
