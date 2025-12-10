"""Tests for hashing, manifest models, plans, and target graph behaviors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from codeintel.build.contracts import OutputContract
from codeintel.build.hashing import compute_input_hash, compute_options_hash
from codeintel.build.manifest import BuildRunRecord, OutputManifest
from codeintel.build.operations import OperationTargets
from codeintel.build.plan import BuildPlan, PlanGenerator, PlanStage, PlanStep, format_duration
from codeintel.build.resolver import ResolutionResult
from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.config.datasets.primitives import Column, TableSchema
from codeintel.serving.operations.catalog import DataSourceType, Operation
from tests._helpers import make_snapshot, sample_target_graph
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_none,
    expect_is_not_none,
    expect_not_equal,
    expect_true,
)
from tests._helpers.operations_registry import OperationRegistryBuilder


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
    expect_equal(hash1, hash2)
    combined = b"demo:c1|main|dep:out-hash,missing:MISSING|opts"
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
    expect_in("Build Plan for: modules", summary)
    expect_in("Skipped: 1 targets", summary)
    expect_in("Blocked: 1 targets", summary)

    expect_equal(format_duration(500), ", ~500ms")
    expect_equal(format_duration(2000), ", ~2s")
    expect_false(bool(format_duration(None)))


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
    expect_equal(plan.total_steps, 1)
    expect_true(any("has no resolution reason" in rec.message for rec in caplog.records))


def test_target_graph_validation_and_topology() -> None:
    """TargetGraph validates missing deps, cycles, and ordering."""
    graph = TargetGraph()
    graph.register(OutputTarget(name="a", module="ingestion", plugin="p"))
    graph.register(OutputTarget(name="b", module="graphs", plugin="p", dependencies=("a",)))
    graph.register(
        OutputTarget(name="c", module="analytics", plugin="p", dependencies=("b", "missing"))
    )

    errors = graph.validate()
    expect_in("missing", errors[0])

    graph_valid = TargetGraph()
    graph_valid.register(OutputTarget(name="a", module="ingestion", plugin="p"))
    graph_valid.register(OutputTarget(name="b", module="graphs", plugin="p", dependencies=("a",)))
    graph_valid.register(
        OutputTarget(name="c", module="analytics", plugin="p", dependencies=("b",))
    )
    order = graph_valid.topological_order(("c",))
    expect_equal(order[0], "a")

    cyclic_graph = TargetGraph()
    cyclic_graph.register(
        OutputTarget(name="cycle1", module="export", plugin="p", dependencies=("cycle2",))
    )
    cyclic_graph.register(
        OutputTarget(name="cycle2", module="export", plugin="p", dependencies=("cycle1",))
    )
    errors_cycle = cyclic_graph.validate()
    expect_true(any("Cycle detected" in err for err in errors_cycle))
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
        contract=OutputContract.simple(table_keys=("core.legacy",)),
    )
    expect_equal(contract_target.table_keys, ("core.t",))
    expect_equal(legacy_target.table_keys, ("core.legacy",))
    expect_true(legacy_target.estimated_duration_ms >= 0)


def test_operations_resolve_missing_mapping() -> None:
    """Operations resolution skips datasets/graphs without mappings."""
    graph = sample_target_graph()
    builder = OperationRegistryBuilder(targets=graph.all_targets)

    op = Operation(
        id="op",
        category="tests",
        summary="missing mappings",
        description=None,
        http_method=None,
        http_path=None,
        tool_name=None,
        output_model_name="None",
        backend_method="",
        data_source=DataSourceType.TABLE,
        source_name="core.unknown",
        repository_method=None,
        required_datasets=("core.unknown",),
        required_graphs=("unknown_graph",),
        exposed_datasets=(),
        supports_pagination=False,
        default_limit=None,
        max_limit=None,
    )
    targets = builder.build_targets_for_operation(op)

    expect_false(bool(targets.data_targets))
    expect_false(bool(targets.graph_targets))


def test_operations_targets_dataclass_repr() -> None:
    """OperationTargets aggregates graph and data targets."""
    targets = OperationTargets(
        operation_id="op",
        required_targets=frozenset({"a", "b"}),
        graph_targets=frozenset({"a"}),
        data_targets=frozenset({"b"}),
    )
    expect_in("op", repr(targets))
