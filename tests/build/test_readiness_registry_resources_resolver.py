"""Readiness, registry, resolver, and resources edge-case tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import duckdb
import pytest

from codeintel.build.contracts import OutputContract
from codeintel.build.hashing import compute_input_hash
from codeintel.build.readiness import DatabaseReadinessView, TargetReadinessView
from codeintel.build.registry import (
    build_target_graph,
    derive_schemas_from_targets,
    get_target_by_table,
)
from codeintel.build.resolver import BuildResolver
from codeintel.build.resources import TargetExecution, TargetResources
from codeintel.build.state import DatabaseState, StalenessReason, TargetState
from codeintel.build.targets import OutputTarget, TargetGraph, TargetOptions
from codeintel.config.datasets.primitives import Column, TableSchema
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.build import ManifestParams, sample_manifest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.build.state import TargetStatus
    from codeintel.storage.datasets import DatasetRegistry
    from codeintel.storage.gateway.accessors import (
        AnalyticsTables,
        CoreTables,
        DocsViews,
        GraphTables,
    )
    from codeintel.storage.gateway.config import StorageConfig
    from codeintel.storage.ibis_adapter import IbisGateway
    from codeintel.storage.tracking import PipelineRunTracking
    from codeintel.storage.tracking.asset_tracking import AssetTracking
    from codeintel.storage.tracking.build_tracking import BuildTracking
    from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

_DURATION_THRESHOLD_MS = 5000


@dataclass
class FakeBuildStore:
    """Fake build store for readiness tests."""

    manifests: Mapping[str, object]

    def load_manifest(self, target: str, repo: str, commit: str) -> object | None:
        """Return manifest for target if present.

        Returns
        -------
        object | None
            Manifest or None when absent.
        """
        _ = (repo, commit)
        return self.manifests.get(target)

    def list_manifests(self, repo: str, commit: str) -> list[object]:
        """List all manifests for repo/commit.

        Returns
        -------
        list[object]
            Stored manifests.
        """
        _ = (repo, commit)
        return list(self.manifests.values())


class _FakeIbisGateway:
    """Minimal ibis gateway stub for readiness tests."""

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        """Return backing DuckDB connection."""
        return self._con

    def table(self, name: str) -> duckdb.DuckDBPyRelation:
        """
        Return a relation for a table.

        Returns
        -------
        DuckDBPyRelation
            Relation bound to the requested table.
        """
        return self._con.table(name)

    def view(self, name: str) -> duckdb.DuckDBPyRelation:
        """
        Return a relation for a view.

        Returns
        -------
        DuckDBPyRelation
            Relation bound to the requested view.
        """
        return self.table(name)

    def sql(self, raw_sql: str) -> duckdb.DuckDBPyRelation:
        """
        Execute raw SQL via DuckDB.

        Returns
        -------
        DuckDBPyRelation
            Relation produced by the SQL statement.
        """
        return self._con.sql(raw_sql)


class FakeGateway(StorageGateway):
    """Minimal StorageGateway implementation for readiness tests."""

    def __init__(self, build: FakeBuildStore) -> None:
        self.build = cast("BuildTracking", build)
        placeholder = SimpleNamespace()
        self.analytics = cast("AnalyticsTables", placeholder)
        self.assets = cast("AssetTracking", placeholder)
        self.config = cast("StorageConfig", placeholder)
        self.core = cast("CoreTables", placeholder)
        self.datasets = cast("DatasetRegistry", placeholder)
        self.docs = cast("DocsViews", placeholder)
        self.graph = cast("GraphTables", placeholder)
        self.runs = cast("PipelineRunTracking", placeholder)
        self.policy = cast("DuckDBPolicyBackend", placeholder)
        self._con = duckdb.connect(":memory:")
        self.ibis = cast("IbisGateway", _FakeIbisGateway(self._con))
        self.executions: list[tuple[str, tuple[object, ...] | None]] = []

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        """Return the fake connection.

        Returns
        -------
        DuckDBPyConnection
            Connection stub used in tests.
        """
        return self._con

    def close(self) -> None:
        """Close the fake connection."""
        self._con.close()

    def execute(
        self, sql: str, params: Sequence[object] | None = None
    ) -> duckdb.DuckDBPyConnection:
        """Proxy execution to the fake connection.

        Returns
        -------
        DuckDBPyConnection
            Connection stub used in tests.
        """
        normalized_params = tuple(params) if params is not None else None
        self.executions.append((sql, normalized_params))
        return self._con.execute(sql, params)

    def table(self, name: str) -> duckdb.DuckDBPyRelation:
        """Proxy table access to the fake connection.

        Returns
        -------
        DuckDBPyRelation
            Relation stub used in tests.
        """
        return self._con.table(name)


def _gateway(manifests: Mapping[str, object]) -> StorageGateway:
    """Create a gateway exposing build accessors.

    Returns
    -------
    StorageGateway
        Gateway with build accessor.
    """
    return FakeGateway(FakeBuildStore(manifests))


def _snapshot(tmp_path_factory: pytest.TempPathFactory | None = None) -> SnapshotRef:
    """Create a snapshot reference rooted at a temporary path.

    Returns
    -------
    SnapshotRef
        Snapshot reference for tests.
    """
    repo_root = tmp_path_factory.mktemp("repo") if tmp_path_factory else Path("repo-root")
    return SnapshotRef(repo="org/repo", commit="abc123", repo_root=repo_root)


def _target(name: str, dependencies: tuple[str, ...] = (), duration: int = 1000) -> OutputTarget:
    """Create a minimal target with controllable duration.

    Returns
    -------
    OutputTarget
        Target configured for tests.
    """
    return OutputTarget.from_tables(
        name=name,
        module="analytics",
        plugin=f"{name}_plugin",
        tables=(f"core.{name}",),
        options=TargetOptions(
            dependencies=dependencies,
            execution=TargetExecution(cpu_intensive=False, max_runtime_ms=duration),
            resources=TargetResources(),
            description=name,
        ),
    )


def test_target_readiness_current_is_ready(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Target is ready when manifest exists and hash matches."""
    snapshot = _snapshot(tmp_path_factory)
    target = _target("solo")
    gateway = _gateway({})
    current_hash = compute_input_hash(target, snapshot, gateway)
    manifest = replace(
        sample_manifest("solo", ManifestParams(input_hash="x")), input_hash=current_hash
    )
    gateway = _gateway({"solo": manifest})

    view = TargetReadinessView(
        target,
        TargetGraphWithTargets(target),
        gateway,
        snapshot,
    )

    readiness = view.readiness
    expect_true(readiness.is_ready)
    expect_equal(readiness.self_status, "current")
    expect_equal(readiness.action_needed.kind, "none")


def test_readiness_blocked_dependency_reports_chain(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Stale dependency blocks downstream target with run_first action."""
    snapshot = _snapshot(tmp_path_factory)
    root = _target("root", duration=2000)
    leaf = _target("leaf", dependencies=("root",))

    stale_manifest = sample_manifest("root", ManifestParams(input_hash="old"))
    gateway = _gateway({"root": stale_manifest})
    graph = TargetGraphWithTargets(root, leaf)
    view = TargetReadinessView(
        leaf,
        graph,
        gateway,
        snapshot,
    )

    readiness = view.readiness
    expect_false(readiness.is_ready)
    expect_equal(readiness.action_needed.kind, "run_first")
    expect_equal(readiness.action_needed.target, "root")
    expect_equal(readiness.ultimate_bottleneck, "root")
    expect_equal(readiness.blocker_chain[0].blocked_by, "root")
    expect_equal(readiness.estimated_time_to_ready_ms, root.estimated_duration_ms)


def test_database_readiness_bottlenecks_and_summary(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """DatabaseReadinessView reports bottlenecks and summaries."""
    snapshot = _snapshot(tmp_path_factory)
    root = _target("root")
    leaf = _target("leaf", dependencies=("root",))
    manifest = sample_manifest("root", ManifestParams(input_hash="old"))
    gateway = _gateway({"root": manifest})
    graph = TargetGraph()
    graph.register(root)
    graph.register(leaf)

    db_view = DatabaseReadinessView(graph, gateway, snapshot)

    expect_in("root", db_view.bottlenecks())
    summary = db_view.summary()
    expect_true(summary["blocked"] >= 1)
    formatted = db_view.format_summary()
    expect_in("Readiness for", formatted)
    expect_in("Bottlenecks", formatted)


def test_readiness_runnable_when_never_computed(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Never-computed target with satisfied deps can run."""
    snapshot = _snapshot(tmp_path_factory)
    target = _target("fresh")
    gateway = _gateway({})

    view = TargetReadinessView(
        target,
        TargetGraphWithTargets(target),
        gateway,
        snapshot,
    )

    readiness = view.readiness
    expect_equal(readiness.action_needed.kind, "run")
    expect_equal(readiness.action_needed.target, "fresh")
    expect_is_not_none(readiness.action_needed.command)


def test_registry_derives_schemas_and_detects_duplicates(caplog: pytest.LogCaptureFixture) -> None:
    """Schema derivation captures duplicates and returns mapping."""
    caplog.set_level("WARNING")
    table = TableSchema(schema="core", name="items", columns=[Column("id", "INTEGER")])
    t1 = OutputTarget(
        name="one", module="analytics", plugin="p1", contract=OutputContract(tables=(table,))
    )
    t2 = OutputTarget(
        name="two", module="analytics", plugin="p2", contract=OutputContract(tables=(table,))
    )

    schemas = derive_schemas_from_targets((t1, t2))

    expect_equal(schemas["core.items"], table)
    expect_true(any("Duplicate schema" in rec.message for rec in caplog.records))


def test_registry_build_target_graph_validation_error() -> None:
    """build_target_graph raises when targets have missing deps."""
    bad_target = OutputTarget.from_tables(
        name="bad",
        module="analytics",
        plugin="p",
        tables=("core.bad",),
        options=TargetOptions(dependencies=("missing_dep",)),
    )

    with pytest.raises(ValueError, match="missing_dep") as excinfo:
        build_target_graph((bad_target,))

    expect_in("missing_dep", str(excinfo.value))


def test_registry_get_target_by_table() -> None:
    """get_target_by_table returns producer target for table."""
    target = OutputTarget.from_tables(
        name="producer",
        module="analytics",
        plugin="p",
        tables=("core.produced",),
    )

    found = get_target_by_table("core.produced", targets=(target,))

    expect_true(found is target)


def _state_with_status(graph: TargetGraph, status: TargetStatus = "computed") -> DatabaseState:
    """Create DatabaseState with uniform status for all targets.

    Returns
    -------
    DatabaseState
        State with all targets at the given status.
    """
    targets = {
        name: TargetState(
            name=name,
            status=status,
            manifest=None,
            staleness_reason=StalenessReason(kind="dependency_blocked", details="blocked")
            if status == "blocked"
            else None,
            blocking_deps=(),
            current_input_hash=None,
        )
        for name in graph
    }
    return DatabaseState(repo="org/repo", commit="abc123", targets=targets)


def test_resolver_unknown_goal_raises_keyerror() -> None:
    """Unknown goal raises KeyError during validation."""
    graph = TargetGraph()
    graph.register(_target("known"))
    resolver = BuildResolver(graph, _state_with_status(graph))

    with pytest.raises(KeyError):
        resolver.resolve(["unknown"])


def test_resolver_cycle_detection_raises() -> None:
    """Cycle in graph triggers ValueError during resolution."""
    graph = TargetGraph()
    a = _target("a", dependencies=("b",))
    b = _target("b", dependencies=("a",))
    graph.register(a)
    graph.register(b)
    resolver = BuildResolver(graph, _state_with_status(graph))

    with pytest.raises(ValueError, match="Cycle detected"):
        resolver.resolve(["a"])


def test_target_resources_and_execution_helpers() -> None:
    """Resource and execution helpers return expected values."""
    resources = TargetResources(tools=("tool",))
    expect_true(resources.requires_any_tool())
    expect_false(TargetResources().requires_any_tool())

    execution = TargetExecution(
        cpu_intensive=True, io_intensive=True, memory_intensive=True, max_runtime_ms=10000
    )
    expect_equal(execution.estimated_duration_ms(), 10000)

    execution_light = TargetExecution(
        cpu_intensive=True, io_intensive=False, memory_intensive=False, max_runtime_ms=60000
    )
    expect_true(execution_light.estimated_duration_ms() > _DURATION_THRESHOLD_MS)


@dataclass
class TargetGraphWithTargets(TargetGraph):
    """TargetGraph populated at construction for convenience."""

    def __init__(self, *targets: OutputTarget) -> None:
        """Register provided targets on creation."""
        super().__init__()
        for target in targets:
            self.register(target)
