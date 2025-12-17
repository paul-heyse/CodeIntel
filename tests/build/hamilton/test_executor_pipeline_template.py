"""Executor pipeline template tests.

This module validates that the reusable executor pipeline template in
``codeintel.build.hamilton.templates.materialize_template`` can be instantiated via
Hamilton's ``@subdag`` decorator and produces correct TargetRunRecords for
success, failure, and skip scenarios.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, cast

import hamilton.driver as h_driver
from hamilton.function_modifiers import source, subdag, tag, value

from codeintel.build.contracts import OutputContract
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.templates import materialize_template
from codeintel.build.hamilton.templates.materialize_template import executor_materialize
from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.build import make_build_config, make_build_paths
from tests._helpers.fakes.fake_providers import FakeProviders

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Protocol

    from codeintel.build.providers import Providers
    from codeintel.storage.gateway import StorageGateway

    class _EphemeralExtractModule(Protocol):
        t__goids__extract: object
        t__goids: object

# Keep types available for Hamilton's runtime type resolution
_HAMILTON_TYPE_HINTS = (TargetRunRecord,)


@dataclass
class MockComputeResult:
    """Mock compute result implementing ComputeResult protocol."""

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


def _make_env(
    *,
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    force_targets: frozenset[str] | None = None,
) -> BuildEnv:
    """Create a BuildEnv for testing.

    Returns
    -------
    BuildEnv
        Build environment configured for testing.
    """
    paths = make_build_paths(snapshot.repo_root)
    config = make_build_config()
    providers = cast("Providers", FakeProviders.defaults())
    return BuildEnv(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        providers=providers,
        config=config,
        force_targets=force_targets or frozenset({"goids"}),
    )


def _make_graph() -> TargetGraph:
    """Create a minimal TargetGraph with goids target.

    Returns
    -------
    TargetGraph
        Target graph with goids target registered.
    """
    graph = TargetGraph()
    graph.register(
        OutputTarget(
            name="goids",
            module="graphs",
            contract=OutputContract.simple(table_keys=("core.goids", "core.goid_crosswalk")),
        )
    )
    return graph


def test_compute_result_protocol() -> None:
    """Verify MockComputeResult has required ComputeResult attributes."""
    result = MockComputeResult(success=True, table_counts={"core.goids": 10})
    # Can't use isinstance with Protocol that has non-method members
    # Instead, verify the attributes exist
    expect_true(hasattr(result, "success"), message="MockComputeResult should have success attr")
    expect_true(
        hasattr(result, "table_counts"), message="MockComputeResult should have table_counts attr"
    )
    expect_true(hasattr(result, "error"), message="MockComputeResult should have error attr")


def test_executor_materialize_success(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify executor_materialize produces succeeded record on success."""
    repo = "test/repo"
    commit = "abc123"
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=tmp_path / "repo")
    env = _make_env(gateway=fresh_gateway, snapshot=snapshot)
    graph = _make_graph()

    compute_result = MockComputeResult(
        success=True,
        table_counts={"core.goids": 100, "core.goid_crosswalk": 50},
    )

    record = executor_materialize(env, graph, "goids", compute_result)

    expect_equal(record.status, expected="succeeded", label="record.status")
    expect_equal(record.target, expected="goids", label="record.target")
    expect_equal(
        record.row_counts.get("core.goids"),
        expected=100,
        label="record.row_counts[core.goids]",
    )
    expect_equal(
        record.row_counts.get("core.goid_crosswalk"),
        expected=50,
        label="record.row_counts[core.goid_crosswalk]",
    )


def test_executor_materialize_failure(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify executor_materialize produces failed record on failure."""
    repo = "test/repo"
    commit = "abc123"
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=tmp_path / "repo")
    env = _make_env(gateway=fresh_gateway, snapshot=snapshot)
    graph = _make_graph()

    compute_result = MockComputeResult(
        success=False,
        error="GOID extraction failed: syntax error",
    )

    record = executor_materialize(env, graph, "goids", compute_result)

    expect_equal(record.status, expected="failed", label="record.status")
    expect_equal(record.target, expected="goids", label="record.target")
    expect_true(
        record.error is not None and "GOID extraction failed" in record.error,
        message=f"Expected error message containing 'GOID extraction failed', got: {record.error}",
    )


def test_executor_materialize_failure_default_error(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify executor_materialize uses default error message when error is None."""
    repo = "test/repo"
    commit = "abc123"
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=tmp_path / "repo")
    env = _make_env(gateway=fresh_gateway, snapshot=snapshot)
    graph = _make_graph()

    compute_result = MockComputeResult(
        success=False,
        error=None,
    )

    record = executor_materialize(env, graph, "goids", compute_result)

    expect_equal(record.status, expected="failed", label="record.status")
    expect_true(
        record.error is not None and "goids computation failed" in record.error,
        message=f"Expected default error message, got: {record.error}",
    )


def _build_subdag_module(compute_result: MockComputeResult) -> ModuleType:
    """Build an ephemeral Hamilton module using materialize_template via @subdag.

    Returns
    -------
    ModuleType
        Ephemeral Hamilton module with executor_pipeline wired via @subdag.
    """
    mod = ModuleType("tests.build.hamilton._executor_pipeline_case")
    mod.__doc__ = "Ephemeral module for testing materialize_template via @subdag."
    sys.modules[mod.__name__] = mod

    # Capture compute_result in closure
    captured_result = compute_result

    @tag(domain="graphs", target="goids", node_type="tool")
    def t__goids__extract(env: BuildEnv) -> MockComputeResult:
        """Return the captured mock compute result.

        Returns
        -------
        MockComputeResult
            Mock result for testing.
        """
        # Use env to satisfy Hamilton's requirement for inputs
        _ = env
        return captured_result

    @tag(domain="graphs", target="goids", node_type="materialize")
    @subdag(
        materialize_template,
        inputs={
            "env": source("env"),
            "graph": source("graph"),
            "target_name": value("goids"),
            "compute_result": source("t__goids__extract"),
        },
    )
    def t__goids(executor_record: TargetRunRecord) -> TargetRunRecord:
        """Return the subDAG-produced record.

        Returns
        -------
        TargetRunRecord
            Target execution record produced by the executor pipeline.
        """
        return executor_record

    # Set module ownership for Hamilton discovery
    t__goids__extract.__module__ = mod.__name__
    t__goids.__module__ = mod.__name__

    module_namespace = cast("_EphemeralExtractModule", mod)
    module_namespace.t__goids__extract = t__goids__extract
    module_namespace.t__goids = t__goids
    return mod


def test_executor_pipeline_via_subdag_success(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify materialize_template works via @subdag with successful compute."""
    repo = "test/repo"
    commit = "abc123"
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=tmp_path / "repo")
    env = _make_env(gateway=fresh_gateway, snapshot=snapshot)
    graph = _make_graph()

    compute_result = MockComputeResult(
        success=True,
        table_counts={"core.goids": 42, "core.goid_crosswalk": 21},
    )
    module = _build_subdag_module(compute_result)

    driver = h_driver.Builder().with_modules(module).build()
    results = driver.execute(["t__goids"], inputs={"env": env, "graph": graph})
    record = cast("TargetRunRecord", results["t__goids"])

    expect_equal(record.status, expected="succeeded", label=f"record.error={record.error}")
    expect_equal(record.target, expected="goids", label="record.target")
    expect_equal(
        record.row_counts.get("core.goids"),
        expected=42,
        label="record.row_counts[core.goids]",
    )


def test_executor_pipeline_via_subdag_failure(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Verify materialize_template works via @subdag with failed compute."""
    repo = "test/repo"
    commit = "abc123"
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=tmp_path / "repo")
    env = _make_env(gateway=fresh_gateway, snapshot=snapshot)
    graph = _make_graph()

    compute_result = MockComputeResult(
        success=False,
        error="Test failure",
    )
    module = _build_subdag_module(compute_result)

    driver = h_driver.Builder().with_modules(module).build()
    results = driver.execute(["t__goids"], inputs={"env": env, "graph": graph})
    record = cast("TargetRunRecord", results["t__goids"])

    expect_equal(record.status, expected="failed", label="record.status")
    expect_true(
        record.error is not None and "Test failure" in record.error,
        message=f"Expected error to contain 'Test failure', got: {record.error}",
    )
