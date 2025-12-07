"""Tests for hashing, manifest models, plans, and target graph behaviors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from codeintel.build import operations
from codeintel.build.contracts import OutputContract
from codeintel.build.hashing import compute_input_hash, compute_options_hash
from codeintel.build.manifest import BuildRunRecord, OutputManifest
from codeintel.build.operations import OperationTargets
from codeintel.build.plan import BuildPlan, PlanGenerator, PlanStage, PlanStep, format_duration
from codeintel.build.resolver import ResolutionResult
from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.config.datasets.primitives import Column, TableSchema
from tests._helpers import make_snapshot, sample_target_graph


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
    """Input hash uses repo/commit/target/dependency hashes deterministically."""
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
        plugin="plugin",
        dependencies=("dep", "missing"),
    )
    snapshot = make_snapshot(tmp_path, repo="demo", commit="c1")

    hash1 = compute_input_hash(target, snapshot, cast("Any", gateway), options_hash="opts")
    hash2 = compute_input_hash(target, snapshot, cast("Any", gateway), options_hash="opts")
    # Deterministic and includes both dependency states
    assert hash1 == hash2
    combined = b"demo:c1|main|dep:out-hash,missing:MISSING|opts"
    assert hash1 == hashlib.sha256(combined).hexdigest()[:16]


def test_compute_options_hash_json_and_fallback() -> None:
    """Options hash handles JSON-able and non-JSON objects."""
    options_hash = compute_options_hash({"a": 1, "b": [2, 3]})
    assert options_hash is not None
    assert len(options_hash) == 16

    class Unserializable:
        def __str__(self) -> str:
            return "repr"

    fallback_hash = compute_options_hash(Unserializable())
    assert fallback_hash is not None
    assert fallback_hash != options_hash
    assert compute_options_hash(None) is None


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
    assert payload["completed_at"] is None
    assert payload["error_summary"] is None


def test_build_plan_summary_and_formatting() -> None:
    """Plan summary includes stages, skipped, and blocked targets."""
    stage = PlanStage(
        module="ingestion",
        steps=(
            PlanStep(
                target="modules",
                module="ingestion",
                plugin="repo_scan",
                estimated_duration_ms=None,
                dependencies=(),
                reason="fresh start",
            ),
        ),
    )
    plan = BuildPlan(
        requested_targets=("modules",),
        stages=(stage,),
        skipped_targets=("ast",),
        blocked_targets=("graphs",),
    )
    summary = plan.format_summary()
    assert "Build Plan for: modules" in summary
    assert "Skipped: 1 targets" in summary
    assert "Blocked: 1 targets" in summary

    assert format_duration(500) == ", ~500ms"
    assert format_duration(2000) == ", ~2s"
    assert not format_duration(None)


def test_plan_generator_warns_on_missing_reason(caplog: pytest.LogCaptureFixture) -> None:
    """PlanGenerator logs when a resolution reason is missing."""
    graph = sample_target_graph()
    resolution = ResolutionResult(
        requested=("modules",),
        to_compute=("modules",),
        to_skip=(),
        blocked=(),
        reasons={},
    )
    caplog.set_level("WARNING")
    plan = PlanGenerator(graph).generate(resolution)
    assert plan.total_steps == 1
    assert any("has no resolution reason" in rec.message for rec in caplog.records)


def test_target_graph_validation_and_topology() -> None:
    """TargetGraph validates missing deps, cycles, and ordering."""
    graph = TargetGraph()
    graph.register(OutputTarget(name="a", module="ingestion", plugin="p"))
    graph.register(OutputTarget(name="b", module="graphs", plugin="p", dependencies=("a",)))
    graph.register(
        OutputTarget(name="c", module="analytics", plugin="p", dependencies=("b", "missing"))
    )

    errors = graph.validate()
    assert "missing" in errors[0]

    graph_valid = TargetGraph()
    graph_valid.register(OutputTarget(name="a", module="ingestion", plugin="p"))
    graph_valid.register(OutputTarget(name="b", module="graphs", plugin="p", dependencies=("a",)))
    graph_valid.register(
        OutputTarget(name="c", module="analytics", plugin="p", dependencies=("b",))
    )
    order = graph_valid.topological_order(("c",))
    assert order[0] == "a"

    cyclic_graph = TargetGraph()
    cyclic_graph.register(
        OutputTarget(name="cycle1", module="export", plugin="p", dependencies=("cycle2",))
    )
    cyclic_graph.register(
        OutputTarget(name="cycle2", module="export", plugin="p", dependencies=("cycle1",))
    )
    errors_cycle = cyclic_graph.validate()
    assert any("Cycle detected" in err for err in errors_cycle)
    with pytest.raises(ValueError, match="Cycle detected"):
        cyclic_graph.topological_order(("cycle1",))


def test_target_table_keys_and_execution_duration() -> None:
    """table_keys prefers contract tables, else falls back to legacy tables."""
    contract_target = OutputTarget(
        name="contracted",
        module="ingestion",
        plugin="p",
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
        plugin="p",
        tables=("core.legacy",),
    )
    assert contract_target.table_keys == ("core.t",)
    assert legacy_target.table_keys == ("core.legacy",)
    assert legacy_target.estimated_duration_ms >= 0


def test_operations_resolve_missing_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operations resolution skips datasets/graphs without mappings."""
    # Override caches to avoid loading real registry
    monkeypatch.setattr(operations, "_TABLE_TO_TARGET", {})
    monkeypatch.setattr(operations, "_GRAPH_TO_TARGET", {})

    unresolved_datasets = operations.resolve_targets_for_operation(
        cast(
            "Any",
            type(
                "Op",
                (),
                {
                    "id": "op",
                    "required_datasets": ("core.unknown",),
                    "required_graphs": ("unknown_graph",),
                },
            )(),
        )
    )

    assert not unresolved_datasets.data_targets
    assert not unresolved_datasets.graph_targets


def test_operations_targets_dataclass_repr() -> None:
    """OperationTargets aggregates graph and data targets."""
    targets = OperationTargets(
        operation_id="op",
        required_targets=frozenset({"a", "b"}),
        graph_targets=frozenset({"a"}),
        data_targets=frozenset({"b"}),
    )
    assert "op" in repr(targets)
